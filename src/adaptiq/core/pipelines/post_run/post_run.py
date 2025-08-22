import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from adaptiq.core.abstract.integrations import BaseConfig, BaseLogParser
from adaptiq.core.entities import (
    Outputs,
    PostRunResults,
    ProcessedLogs,
    ReconciliationResults,
    Stats,
    ValidationData,
    ValidationResults,
)
from adaptiq.core.pipelines.post_run.tools import PostRunReconciler, PostRunValidator


class PostRunPipeline:
    """
    AdaptiqPostRunOrchestrator orchestrates the entire workflow of:
    1. Capturing the agent's execution trace using AdaptiqAgentTracer,
    2. Parsing the logs using AdaptiqLogParser
    3. Validating the parsed logs using AdaptiqPostRunValidator

    This class serves as a high-level interface to the entire pipeline,
    providing methods to execute the full workflow or individual stages.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        base_log_parser: BaseLogParser,
        output_dir: str,
        feedback: Optional[str] = None,
    ):
        """
        Initialize the AdaptiqPostRunOrchestrator.

        Args:
            base_config: BaseConfig instance containing configuration data.
            base_log_parser: BaseLogParser instance for parsing logs.
            output_dir: Directory where output files will be saved.
            feedback: Optional human feedback to guide the post-run analysis.
        """

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-PostRun")

        self.base_config = base_config
        self.configuration = base_config.get_config()
        self.llm = self.base_config.get_llm_instance()
        self.embedding = self.base_config.get_embeddings_instance()
        self.output_dir = output_dir
        self.feedback = feedback

        self.agent_name = self.configuration.agent_modifiable_config.agent_name
        self.report_path = self.configuration.report_config.output_path

        self.old_prompt = base_config.get_prompt(get_newest=True)

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Other components will be initialized as needed
        self.log_parser = base_log_parser
        self.validator = None
        self.reconciler = None

        # Paths for output files
        self.parsed_logs_path = os.path.join(output_dir, "parsed_logs.json")
        self.validated_logs_path = os.path.join(output_dir, "validated_logs.json")
        self.validation_summary_path = os.path.join(
            output_dir, "validation_summary.json"
        )

    def parse_logs(self) -> ProcessedLogs:
        """
        Parse logs using AdaptiqLogParser.

        Returns:
            ProcessedLogs: Parsed logs containing state-action-reward mappings

        Raises:
            FileNotFoundError: If raw logs are not provided and can't be loaded.
        """
        self.logger.info("Starting log parsing...")

        try:

            # Initialize and run parser
            parsed_data = self.log_parser.parse_logs()

            self.logger.info(
                f"Log parsing completed, generated {len(parsed_data.processed_logs)} state-action-reward mappings"
            )
            self.logger.info(f"Parsed logs saved to {self.parsed_logs_path}")

            return parsed_data

        except FileNotFoundError as e:
            self.logger.error(f"Failed to parse logs: {e}")
            raise

    def validate_parsed_logs(
        self,
        raw_logs: List[Dict[str, Any]],
        parsed_logs: ProcessedLogs,
    ) -> Tuple[ProcessedLogs, ValidationResults]:
        """
        Validate the parsed logs using AdaptiqPostRunValidator.

        Args:
            raw_logs: List of raw log entries as dictionaries
            parsed_logs: ProcessedLogs containing parsed log entries

        Returns:
            Tuple containing:
            - List of corrected logs with validated rewards
            - Dictionary with validation results and summary
        """
        self.logger.info("Starting validation of parsed logs...")

        # Initialize and run validator
        self.validator = PostRunValidator(
            raw_logs=raw_logs, parsed_logs=parsed_logs, llm=self.llm
        )

        corrected_logs, validation_results = self.validator.run_validation_pipeline()

        return ProcessedLogs(processed_logs=corrected_logs), validation_results

    def reconciliate_logs(self, parsed_logs: ProcessedLogs) -> ReconciliationResults:
        """
        Reconciliate logs using PostRunReconciler.

        Args:
            parsed_logs: ProcessedLogs containing parsed log entries

        Returns:
            ReconciliationResults: The results of the reconciliation process
        """
        # Ensure output directory exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use pathlib for proper path construction
        # parsed_logs_file = output_dir / "parsed_logs.json"
        warmed_qtable_file = output_dir / "adaptiq_q_table.json"

        self.reconciler = PostRunReconciler(
            parsed_logs=parsed_logs,
            warmed_qtable_file=str(warmed_qtable_file),
            llm=self.llm,
            embeddings=self.embedding,
            old_prompt=self.old_prompt,
            agent_name=self.agent_name,
            feedback=self.feedback,
            report_path=self.report_path,
        )

        result = self.reconciler.run_process()

        results_file = output_dir / "results.json"
        self.reconciler.save_results(results=result, output_file=str(results_file))

        return result

    def get_raw_logs(self, trace_output) -> Dict[str, Any]:
        """
        Get the agent trace.

        Returns:
            Dict containing the raw logs from the agent trace.
        """
        self.logger.info("Starting agent trace retrieval...")

        try:
            raw_logs: Dict[str, Any] = json.loads(trace_output)
            return raw_logs
        except Exception as e:
            self.logger.error(f"Failed to save raw logs to file: {e}")

    def execute_post_run_pipeline(self) -> PostRunResults:
        """
        Run the complete pipeline: agent execution, log parsing, and validation.

        Returns:
            PostRunResults: Results of the post-run pipeline including validation and reconciliation data.
        """
        self.logger.info("Starting full Adaptiq pipeline execution...")

        trace_output = self.base_config.get_agent_trace()
        raw_logs = self.get_raw_logs(trace_output=trace_output)

        # Parse logs
        parsed_logs: ProcessedLogs = self.parse_logs()

        # Step 3: Validate parsed logs
        corrected_logs, validation_results = self.validate_parsed_logs(
            raw_logs, parsed_logs
        )

        # Step 4: Reconciliate logs
        reconciliated_data: ReconciliationResults = self.reconciliate_logs(parsed_logs)

        # Prepare pipeline results
        validation_output = ValidationData(
            outputs=Outputs(
                parsed_logs_path=self.parsed_logs_path,
                validated_logs_path=self.validated_logs_path,
                validation_summary_path=self.validation_summary_path,
            ),
            stats=Stats(
                parsed_entries_count=(
                    len(parsed_logs.processed_logs) if parsed_logs else 0
                ),
                validation_results=validation_results,
            ),
        )

        pipeline_results = PostRunResults(
            validation_data=validation_output, reconciliation_results=reconciliated_data
        )

        self.logger.info("Full pipeline execution completed successfully")

        return pipeline_results

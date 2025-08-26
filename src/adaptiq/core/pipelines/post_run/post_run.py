import logging
import os
from pathlib import Path
from typing import Optional, Tuple

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

from adaptiq.core.pipelines.post_run.tools import PostRunReconciler


class PostRunPipeline:
    """
    AdaptiqPostRunOrchestrator orchestrates the entire workflow of:
    1. Capturing the agent's execution trace using AdaptiqAgentTracer,
    2. Parsing the logs and validating them using AdaptiqLogParser

    This class serves as a high-level interface to the entire pipeline,
    providing methods to execute the full workflow or individual stages.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        base_log_parser: BaseLogParser,
        output_path: str,
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
        self.output_dir = output_path
        self.feedback = feedback

        self.agent_name = self.configuration.agent_modifiable_config.agent_name
        self.report_path = self.configuration.report_config.output_path

        self.old_prompt = base_config.get_prompt(get_newest=True)

        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.log_parser = base_log_parser
        self.log_parser.set_embeddings(self.embedding)
        self.validator = None
        self.reconciler = None

        # Paths for output files
        self.parsed_logs_path = os.path.join(output_path, "parsed_logs.json")
        self.validated_logs_path = os.path.join(output_path, "validated_logs.json")
        self.validation_summary_path = os.path.join(
            output_path, "validation_summary.json"
        )

    def parse_logs(self) -> Tuple[ProcessedLogs, ValidationResults]:
        """
        Parse logs using AdaptiqLogParser.

        Returns:
            Tuple[ProcessedLogs, ValidationResults]: Parsed logs containing state-action-reward mappings

        Raises:
            FileNotFoundError: If raw logs are not provided and can't be loaded.
        """
        self.logger.info("Starting log parsing...")

        try:

            # Initialize and run parser
            parsed_data, validation_results = self.log_parser.parse_logs()

            self.logger.info(
                f"Log parsing completed, generated {len(parsed_data.processed_logs)} state-action-reward mappings"
            )

            self.logger.info(f"Parsed logs saved to {self.parsed_logs_path}")

            return parsed_data, validation_results

        except FileNotFoundError as e:
            self.logger.error(f"Failed to parse logs: {e}")
            raise

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

    def execute_post_run_pipeline(self) -> PostRunResults:
        """
        Run the complete pipeline: agent execution, log parsing, and validation.

        Returns:
            PostRunResults: Results of the post-run pipeline including validation and reconciliation data.
        """
        self.logger.info("Starting full Adaptiq pipeline execution...")

        # Parse logs
        parsed_logs, validation_results = self.parse_logs()

        # Reconciliate logs
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

import json
import logging
from pathlib import Path
from typing import Any, Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from adaptiq.core.entities import (
    ProcessedLogs,
    ReconciliationResults,
    ReconciliationSummary,
)
from adaptiq.core.pipelines.post_run.tools.post_run_updater import PostRunUpdater
from adaptiq.core.pipelines.post_run.tools.prompt_engineer import PromptEngineer
from adaptiq.core.q_table import StateActionExtractor, StateMapper

logger = logging.getLogger(__name__)


class PostRunReconciler:
    """
    Orchestrator class that coordinates the entire Adaptiq reconciliation pipeline.

    This class manages the flow between:
    1. StateActionExtractor - processes execution data
    2. StateMapper - matches states with Q-table
    3. PostRunUpdater - updates Q-table based on classifications
    4. AdaptiqPromptEngineer - generates improvement reports
    """

    def __init__(
        self,
        parsed_logs: ProcessedLogs,
        warmed_qtable_file: str,
        llm: BaseChatModel,
        embeddings: Embeddings,
        old_prompt: str = None,
        agent_name: str = None,
        feedback: str = None,
        report_path: str = None,
    ):
        """
        Initialize the orchestrator with file paths and configuration.

        Args:
            parsed_logs: ProcessedLogs containing execution data for extraction
            warmed_qtable_file: Path to JSON file containing the warmed Q-table
            llm: BaseChatModel instance for language model interactions
            embeddings: Embeddings instance for text embeddings
            old_prompt: Optional prompt string for reference
            agent_name: Optional name of the agent
            feedback: Optional human feedback for prompt evaluation
            report_path: Optional path to save the report
        """
        self.parsed_logs = parsed_logs
        self.warmed_qtable_file = Path(warmed_qtable_file)
        self.embedding_model = "text-embedding-3-small"
        self.old_prompt = old_prompt
        self.agent_name = agent_name
        self.llm = llm
        self.embedding = embeddings
        self.report_path = Path(report_path) if report_path else None
        self.feedback = feedback
        self.alpha = 0.8
        self.gamma = 0.8
        self.similarity_threshold = 0.7

        # Validate file existence
        self._validate_files()

        # Initialize components (will be done lazily)
        self.extractor = None
        self.mapper = None
        self.post_run_updater = None
        self.prompt_engineer = None

        logger.info("PostRunReconciler initialized successfully")

    def _validate_files(self):
        """Validate that all required files exist."""
        files_to_check = [
            (self.warmed_qtable_file, "Warmed Q-table file"),
        ]

        for file_path, description in files_to_check:
            if not file_path.exists():
                raise FileNotFoundError(f"{description} not found: {file_path}")

        logger.info("All required files validated successfully")

    def _load_json_file(self, file_path: Path) -> Any:
        """Load and return data from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Successfully loaded JSON file: %s", file_path)
            return data
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in file %s: %s", file_path, e)
            raise
        except Exception as e:
            logger.error("Error loading file %s: %s", file_path, e)
            raise

    def _initialize_extractor(self):
        """Initialize the StateActionExtractor if not already done."""
        if self.extractor is None:
            self.extractor = StateActionExtractor(
                llm=self.llm,
            )
            logger.info("StateActionExtractor initialized")

    def _initialize_mapper(self, warmed_qtable_data: Dict):
        """Initialize the StateMapper if not already done."""
        if self.mapper is None:
            self.mapper = StateMapper(
                warmed_qtable_data=warmed_qtable_data,
                llm=self.llm,
            )
            logger.info("StateMapper initialized")

    def _initialize_post_run_updater(self):
        """Initialize the PostRunUpdater if not already done."""
        if self.post_run_updater is None:
            self.post_run_updater = PostRunUpdater(
                embeddings=self.embedding,
                alpha=self.alpha,
                gamma=self.gamma,
                similarity_threshold=self.similarity_threshold,
            )
            logger.info("PostRunUpdater initialized")

    def _initialize_prompt_engineer(self):
        """Initialize the AdaptiqPromptEngineer if not already done."""
        if self.prompt_engineer is None:
            self.prompt_engineer = PromptEngineer(
                llm=self.llm,
                report_path=self.report_path,
                old_prompt=self.old_prompt,
                agent_name=self.agent_name,
                feedback=str(self.feedback),
            )
            logger.info("AdaptiqPromptEngineer initialized")

    def run_process(self) -> ReconciliationResults:
        """
        Run the complete reconciliation pipeline.

        Returns:
            ReconciliationResults: Results of the reconciliation process
        """
        logger.info("Starting Adaptiq reconciliation pipeline")

        try:
            # Step 1: Load all required data
            logger.info("Step 1: Loading input data files")

            warmed_qtable_data = self._load_json_file(self.warmed_qtable_file)

            # Step 2: Extract state-action pairs from execution data
            logger.info("Step 2: Extracting state-action pairs")
            self._initialize_extractor()
            extracted_data = self.extractor.process_batch(self.parsed_logs)
            logger.info("Extracted %d state-action pairs", len(extracted_data))

            # Step 3: Map states to Q-table states
            logger.info("Step 3: Mapping states to Q-table")
            self._initialize_mapper(warmed_qtable_data)
            state_classifications = self.mapper.classify_states(extracted_data)
            logger.info("Classified %d states", len(state_classifications))

            # Log classification summary
            known_states_count = sum(
                1 for c in state_classifications if c.classification.is_known_state
            )
            logger.info(
                "Found %d known states out of %d total",
                known_states_count,
                len(state_classifications),
            )

            # Step 4: Update Q-table based on classifications and rewards
            logger.info("Step 4: Updating Q-table")
            self._initialize_post_run_updater()
            updated_qtable, q_insights = self.post_run_updater.process_data(
                state_classifications_data=state_classifications,
                reward_execs_data=self.parsed_logs.processed_logs,
                q_table_data=warmed_qtable_data,
            )
            logger.info("Q-table updated successfully")
            # Step 5: Generate prompt engineering report
            logger.info("Step 5: Generating prompt engineering report")
            self._initialize_prompt_engineer()
            report_content = self.prompt_engineer.generate_and_save_report(
                q_insights=q_insights,
            )
            logger.info("Prompt engineering report generated and saved")

            # Compile results
            results = ReconciliationResults(
                pipeline_status="completed",
                extracted_data=extracted_data,
                state_classifications=state_classifications,
                updated_qtable=updated_qtable,
                report_content=report_content,
                summary=ReconciliationSummary(
                    total_extracted_pairs=len(extracted_data),
                    total_classified_states=len(state_classifications),
                    known_states_found=known_states_count,
                    unknown_states_found=len(state_classifications)
                    - known_states_count,
                    task_key=self.prompt_engineer.task_name,
                    new_prompt=self.prompt_engineer.new_prompt,
                ),
            )

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error("Pipeline failed with error: %s", e)
            return ReconciliationResults(
                pipeline_status="failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    def save_results(self, results: ReconciliationResults, output_file: str):
        """
        Save pipeline results to a JSON file.

        Args:
            results: ReconciliationResults object containing the pipeline output
            output_file: Path to save the results
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info("Results saved to: %s", output_path)
        except Exception as e:
            logger.error("Error saving results to %s: %s", output_path, e)
            raise

import json
import logging
from pathlib import Path
from typing import Any, Dict

from adaptiq.core.q_table.state_mapper import StateMapper
from adaptiq.core.q_table.state_action_extractor import StateActionExtractor
from adaptiq.core.pipelines.post_run.tools.post_run_updater import PostRunUpdater
from adaptiq.core.pipelines.post_run.tools.prompt_engineer import PromptEngineer

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
        execution_data_file: str,
        warmed_qtable_file: str,
        reward_execs_file: str,
        model_name: str,
        api_key: str,
        provider: str,
        old_prompt: str = None,
        agent_name: str = None,
        feedback: str = None,
        report_path: str = None
    ):
        """
        Initialize the orchestrator with file paths and configuration.

        Args:
            execution_data_file: Path to JSON file containing execution data for extraction
            warmed_qtable_file: Path to JSON file containing the warmed Q-table
            reward_execs_file: Path to JSON file containing reward execution data
            config_file: Path to YAML config file for AdaptiqPromptEngineer
            feedback: Human feedback for prompt evaluation
        """
        self.execution_data_file = Path(execution_data_file)
        self.warmed_qtable_file = Path(warmed_qtable_file)
        self.reward_execs_file = Path(reward_execs_file)
        self.embedding_model = "text-embedding-3-small"
        self.old_prompt = old_prompt
        self.agent_name = agent_name
        self.report_path = Path(report_path) if report_path else None

        self.feedback = feedback

        self.api_key = api_key
        if not self.api_key:
            raise ValueError("API key must be provided for the reconciliation pipeline")
        
        self.model_name = model_name
        self.provider = provider

        # Validate file existence
        self._validate_files()

        # Store parameters for component initialization
        self.extractor_params = {
            "model": self.model_name,
            "api_key": self.api_key,
            "provider": self.provider,
        }
        self.mapper_params = {
            "llm_model_name_for_reconciliation": self.model_name,
            "llm_api_key": self.api_key,
            "provider": self.provider,
        }
        self.post_run_updater_params = {
            "api_key": self.api_key,
            "model": self.embedding_model,
            "provider": self.provider,
            "alpha": 0.8,
            "gamma": 0.8,
            "similarity_threshold": 0.7,
        }

        # Initialize components (will be done lazily)
        self.extractor = None
        self.mapper = None
        self.post_run_updater = None
        self.prompt_engineer = None

        logger.info("PostRunReconciler initialized successfully")


    def _validate_files(self):
        """Validate that all required files exist."""
        files_to_check = [
            (self.execution_data_file, "Execution data file"),
            (self.warmed_qtable_file, "Warmed Q-table file"),
            (self.reward_execs_file, "Reward executions file")
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
                provider=self.extractor_params["provider"],
                model=self.extractor_params["model"],
                api_key=self.extractor_params["api_key"],
            )
            logger.info("StateActionExtractor initialized")

    def _initialize_mapper(self, warmed_qtable_data: Dict):
        """Initialize the StateMapper if not already done."""
        if self.mapper is None:
            self.mapper = StateMapper(
                warmed_qtable_data=warmed_qtable_data,
                provider=self.mapper_params["provider"],
                llm_model_name_for_reconciliation=self.mapper_params[
                    "llm_model_name_for_reconciliation"
                ],
                llm_api_key=self.mapper_params["llm_api_key"],
            )
            logger.info("StateMapper initialized")

    def _initialize_post_run_updater(self):
        """Initialize the PostRunUpdater if not already done."""
        if self.post_run_updater is None:
            self.post_run_updater = PostRunUpdater(
                provider=self.post_run_updater_params["provider"],
                api_key=self.post_run_updater_params["api_key"],
                model=self.post_run_updater_params["model"],
                alpha=self.post_run_updater_params["alpha"],
                gamma=self.post_run_updater_params["gamma"],
                similarity_threshold=self.post_run_updater_params["similarity_threshold"],
            )
            logger.info("PostRunUpdater initialized")

    def _initialize_prompt_engineer(self):
        """Initialize the AdaptiqPromptEngineer if not already done."""
        if self.prompt_engineer is None:
            self.prompt_engineer = PromptEngineer(
                model_name=self.model_name,
                api_key=self.api_key,
                provider=self.provider,
                report_path=self.report_path,
                old_prompt=self.old_prompt,
                agent_name=self.agent_name,
                feedback=str(self.feedback)
            )
            logger.info("AdaptiqPromptEngineer initialized")

    def run_process(self) -> Dict[str, Any]:
        """
        Run the complete reconciliation pipeline.

        Returns:
            Dict containing results from each stage of the pipeline
        """
        logger.info("Starting Adaptiq reconciliation pipeline")

        try:
            # Step 1: Load all required data
            logger.info("Step 1: Loading input data files")
            execution_data = self._load_json_file(self.execution_data_file)
            warmed_qtable_data = self._load_json_file(self.warmed_qtable_file)
            reward_execs_data = self._load_json_file(self.reward_execs_file)

            # Step 2: Extract state-action pairs from execution data
            logger.info("Step 2: Extracting state-action pairs")
            self._initialize_extractor()
            extracted_data = self.extractor.process_batch(execution_data)
            logger.info("Extracted %d state-action pairs", len(extracted_data))

            # Step 3: Map states to Q-table states
            logger.info("Step 3: Mapping states to Q-table")
            self._initialize_mapper(warmed_qtable_data)
            state_classifications = self.mapper.classify_states(extracted_data)
            logger.info("Classified %d states", len(state_classifications))

            # Log classification summary
            known_states_count = sum(
                1
                for c in state_classifications
                if c["classification"]["is_known_state"]
            )
            logger.info(
                "Found %d known states out of %d total",
                known_states_count,
                len(state_classifications)
            )

            # Step 4: Update Q-table based on classifications and rewards
            logger.info("Step 4: Updating Q-table")
            self._initialize_post_run_updater()
            updated_qtable = self.post_run_updater.process_data(
                state_classifications_data=state_classifications,
                reward_execs_data=reward_execs_data,
                q_table_data=warmed_qtable_data,
            )
            logger.info("Q-table updated successfully")

            # Step 5: Generate prompt engineering report
            logger.info("Step 5: Generating prompt engineering report")
            self._initialize_prompt_engineer()
            report_content = self.prompt_engineer.generate_and_save_report(
                q_table_output=updated_qtable,
            )
            logger.info("Prompt engineering report generated and saved")

            # Compile results
            results = {
                "pipeline_status": "completed",
                "extracted_data": extracted_data,
                "state_classifications": state_classifications,
                "updated_qtable": updated_qtable,
                "report_content": report_content,
                "summary": {
                    "total_extracted_pairs": len(extracted_data),
                    "total_classified_states": len(state_classifications),
                    "known_states_found": known_states_count,
                    "unknown_states_found": len(state_classifications)
                    - known_states_count,
                    "task_key": self.prompt_engineer.task_name,
                    "new_prompt": self.prompt_engineer.new_prompt,
                },
            }

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error("Pipeline failed with error: %s", e)
            return {
                "pipeline_status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save pipeline results to a JSON file.

        Args:
            results: Results dictionary from run_pipeline()
            output_file: Path to save the results
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Results saved to: %s", output_path)
        except Exception as e:
            logger.error("Error saving results to %s: %s", output_path, e)
            raise



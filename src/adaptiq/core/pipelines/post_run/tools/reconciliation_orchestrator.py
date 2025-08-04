import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from adaptiq.core.q_table.state_mapper import StateMapper
from adaptiq.core.q_table.state_action_extractor import StateActionExtractor
from adaptiq.core.pipelines.post_run.tools.reconciliation import Reconciliation
from adaptiq.core.pipelines.post_run.tools.prompt_engineer import PromptEngineer

logger = logging.getLogger(__name__)

class ReconciliationOrchestrator:
    """
    Orchestrator class that coordinates the entire Adaptiq reconciliation pipeline.

    This class manages the flow between:
    1. StateActionExtractor - processes execution data
    2. StateMapper - matches states with Q-table
    3. AdaptiqQtablePostrunUpdate - updates Q-table based on classifications
    4. AdaptiqPromptEngineer - generates improvement reports
    """

    def __init__(
        self,
        execution_data_file: str,
        warmed_qtable_file: str,
        reward_execs_file: str,
        config_file: str,
        feedback: str = None,
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
        self.config_file = Path(config_file)
        self.embedding_model = "text-embedding-3-small"

        self.config = self._load_config(config_path=self.config_file)
        self.feedback = feedback

        # Extract key configuration
        self.llm_config = self.config.get("llm_config", {})
        self.agent_config = self.config.get("agent_modifiable_config", {})

        self.api_key = self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model_name = self.llm_config.get("model_name")
        self.provider = self.llm_config.get("provider")

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
        self.Reconciliation_params = {
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
        self.reconciliation = None
        self.prompt_engineer = None

        logger.info("AdaptiqReconciliationOrchestrator initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and parse the ADAPTIQ configuration YAML file

        Args:
            config_path: Path to the configuration file

        Returns:
            dict: The parsed configuration
        """
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise

    def _validate_files(self):
        """Validate that all required files exist."""
        files_to_check = [
            (self.execution_data_file, "Execution data file"),
            (self.warmed_qtable_file, "Warmed Q-table file"),
            (self.reward_execs_file, "Reward executions file"),
            (self.config_file, "Configuration file"),
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
            logger.info(f"Successfully loaded JSON file: {file_path}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
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

    def _initialize_Reconciliation(self):
        """Initialize the AdaptiqQtablePostrunUpdate if not already done."""
        if self.reconciliation is None:
            self.reconciliation = Reconciliation(
                provider=self.Reconciliation_params["provider"],
                api_key=self.Reconciliation_params["api_key"],
                model=self.Reconciliation_params["model"],
                alpha=self.Reconciliation_params["alpha"],
                gamma=self.Reconciliation_params["gamma"],
                similarity_threshold=self.Reconciliation_params["similarity_threshold"],
            )
            logger.info("AdaptiqQtablePostrunUpdate initialized")

    def _initialize_prompt_engineer(self):
        """Initialize the AdaptiqPromptEngineer if not already done."""
        if self.prompt_engineer is None:
            self.prompt_engineer = PromptEngineer(
                main_config_path=str(self.config_file), feedback=str(self.feedback)
            )
            logger.info("AdaptiqPromptEngineer initialized")

    def run_pipeline(self) -> Dict[str, Any]:
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
            logger.info(f"Extracted {len(extracted_data)} state-action pairs")

            # Step 3: Map states to Q-table states
            logger.info("Step 3: Mapping states to Q-table")
            self._initialize_mapper(warmed_qtable_data)
            state_classifications = self.mapper.classify_states(extracted_data)
            logger.info(f"Classified {len(state_classifications)} states")

            # Log classification summary
            known_states_count = sum(
                1
                for c in state_classifications
                if c["classification"]["is_known_state"]
            )
            logger.info(
                f"Found {known_states_count} known states out of {len(state_classifications)} total"
            )

            # Step 4: Update Q-table based on classifications and rewards
            logger.info("Step 4: Updating Q-table")
            self._initialize_Reconciliation()
            updated_qtable = self.reconciliation.process_data(
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
            logger.error(f"Pipeline failed with error: {e}")
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
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")
            raise


def adaptiq_reconciliation_pipeline(
    config_path: str, output_path: str, feedback: str = None
) -> Any:
    """Execute full reconciliation pipeline workflow."""
    # Ensure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use pathlib for proper path construction
    execution_data_file = output_dir / "parsed_logs.json"
    warmed_qtable_file = output_dir / "adaptiq_q_table.json"
    reward_execs_file = output_dir / "parsed_logs.json"

    # Check if required files exist, create empty ones if they don't
    if not execution_data_file.exists():
        # Create an empty JSON file or copy from elsewhere if needed
        execution_data_file.write_text("[]")  # or appropriate empty structure
        print(f"Created empty execution data file: {execution_data_file}")

    reconciliation_orchestrator = ReconciliationOrchestrator(
        execution_data_file=str(execution_data_file),
        warmed_qtable_file=str(warmed_qtable_file),
        reward_execs_file=str(reward_execs_file),
        config_file=config_path,
        feedback=feedback,
    )

    result = reconciliation_orchestrator.run_pipeline()

    # Fix the save_results call - output_path should be a file path, not directory
    results_file = output_dir / "results.json"  # or whatever filename you want
    reconciliation_orchestrator.save_results(
        results=result, output_file=str(results_file)
    )

    return result

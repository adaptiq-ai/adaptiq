import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from adaptiq.core.abstract.integrations import BaseConfig, BaseLogParser
from adaptiq.core.pipelines.post_run.tools import PostRunValidator
from adaptiq.core.pipelines.post_run.tools import PostRunReconciler


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
    feedback: Optional[str] = None
    ):
        """
        Initialize the AdaptiqPostRunOrchestrator.

        Args:
            config_path (str): Path to the agent configuration file in YAML format.
            output_dir (str): Directory for saving outputs (logs, parsed data, validation results).
            model_name (str): Name of the LLM model to use for validation.
            api_key (Optional[str]): API key for the LLM service. If None, will try to use environment variable.
            validate_results (bool): Whether to perform validation after parsing.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file cannot be parsed.
        """

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-PostRun")

        self.base_config = base_config
        self.configuration = base_config.get_config()
        self.output_dir = output_dir
        self.feedback = feedback

        self.api_key = self.configuration.llm_config.api_key
        self.model_name = self.configuration.llm_config.model_name.value
        self.provider = self.configuration.llm_config.provider.value
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
        self.raw_logs_path = os.path.join(output_dir, "raw_logs.json")
        self.parsed_logs_path = os.path.join(output_dir, "parsed_logs.json")
        self.validated_logs_path = os.path.join(output_dir, "validated_logs.json")
        self.validation_summary_path = os.path.join(
            output_dir, "validation_summary.json"
        )

    def parse_logs(
        self, raw_logs: Optional[str] = None
    ) -> Dict[Tuple[Tuple[str, str, str, str], str], float]:
        """
        Parse logs using AdaptiqLogParser.

        Args:
            raw_logs (Optional[str]): Raw logs as string. If None, will use logs from run_agent() if available,
                                     or try to load from raw_logs_path.

        Returns:
            Dict: The parsed log data with state-action-reward mappings.

        Raises:
            FileNotFoundError: If raw logs are not provided and can't be loaded.
        """
        self.logger.info("Starting log parsing...")
        try:
            # If raw_logs is None, try to load from file
            if raw_logs is None:
                if not os.path.exists(self.raw_logs_path):
                    raise FileNotFoundError(
                        f"Raw logs file not found at {self.raw_logs_path}"
                    )

                
            with open(self.raw_logs_path, "r", encoding="utf-8") as f:
                if self.raw_logs_path.endswith(".json"):
                    # Load as JSON
                    raw_content = json.load(f)
                    # Convert back to string if needed for parser
                    raw_logs = json.dumps(raw_content)
                else:
                    # Load as text
                    raw_logs = f.read()

        except Exception as e:
            self.logger.error(f"Failed to load raw logs from file: {e}")
            raise

        # Initialize parser with temporary file path for raw data
        temp_raw_logs_path = os.path.join(self.output_dir, "temp_raw_logs.json")

        try:
            # Write raw logs to temporary file for parser
            with open(temp_raw_logs_path, "w", encoding="utf-8") as f:
                f.write(raw_logs)

            # Initialize and run parser
            parsed_data = self.log_parser.parse_logs()

            self.logger.info(
                f"Log parsing completed, generated {len(parsed_data)} state-action-reward mappings"
            )
            self.logger.info(f"Parsed logs saved to {self.parsed_logs_path}")

            return parsed_data

        finally:
            # Clean up temporary file
            if os.path.exists(temp_raw_logs_path):
                os.remove(temp_raw_logs_path)

    def validate_parsed_logs(
        self,
        raw_logs: Optional[List[Dict[str, Any]]] = None,
        parsed_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate the parsed logs using AdaptiqPostRunValidator.

        Args:
            raw_logs (Optional[List[Dict]]): Raw logs as a list of dictionaries.
                                            If None, will try to load from raw_logs_path.
            parsed_logs (Optional[List[Dict]]): Parsed logs as a list of dictionaries.
                                              If None, will try to load from parsed_logs_path.

        Returns:
            Tuple containing:
            - List of corrected logs with validated rewards
            - Dictionary with validation results and summary

        Raises:
            FileNotFoundError: If logs are not provided and can't be loaded.
            ValueError: If API key is not provided and can't be found in environment.
        """
        self.logger.info("Starting validation of parsed logs...")

        if self.api_key is None:
            raise ValueError("API key is required for validation but none was provided")

        # Load raw logs if not provided
        if raw_logs is None:
            if not os.path.exists(self.raw_logs_path):
                raise FileNotFoundError(
                    f"Raw logs file not found at {self.raw_logs_path}"
                )

            try:
                with open(self.raw_logs_path, "r", encoding="utf-8") as f:
                    raw_logs = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load raw logs from file: {e}")
                raise

        # Load parsed logs if not provided
        if parsed_logs is None:
            if not os.path.exists(self.parsed_logs_path):
                raise FileNotFoundError(
                    f"Parsed logs file not found at {self.parsed_logs_path}"
                )

            try:
                with open(self.parsed_logs_path, "r", encoding="utf-8") as f:
                    parsed_logs = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load parsed logs from file: {e}")
                raise

        # Initialize and run validator
        self.validator = PostRunValidator(
            raw_logs=raw_logs,
            parsed_logs=parsed_logs,
            model_name=self.model_name,
            api_key=self.api_key,
            provider=self.provider,
        )

        corrected_logs, validation_results = self.validator.run_validation_pipeline()

        # Save validated logs and validation summary
        try:
            with open(self.validated_logs_path, "w", encoding="utf-8") as f:
                json.dump(corrected_logs, f, indent=2, ensure_ascii=False)

            with open(self.validation_summary_path, "w", encoding="utf-8") as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Validated logs saved to {self.validated_logs_path}")
            self.logger.info(
                f"Validation summary saved to {self.validation_summary_path}"
            )

        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")

        return corrected_logs, validation_results

    def reconciliate_logs(self) -> Dict[str, Any]:
        """
        Reconciliate logs using PostRunReconciler.

        This method is a placeholder for future implementation of log reconciliation.
        Currently, it does not perform any operations.
        """
        # Ensure output directory exists
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use pathlib for proper path construction
        execution_data_file = output_dir / "parsed_logs.json"
        warmed_qtable_file = output_dir / "adaptiq_q_table.json"
        reward_execs_file = output_dir / "parsed_logs.json"

        # Check if required files exist, create empty ones if they don't
        if not execution_data_file.exists():
            # Create an empty JSON file or copy from elsewhere if needed
            execution_data_file.write_text("[]")  # or appropriate empty structure
            print("Created empty execution data file: %s", execution_data_file)

        self.reconciler = PostRunReconciler(
            execution_data_file=str(execution_data_file),
            warmed_qtable_file=str(warmed_qtable_file),
            reward_execs_file=str(reward_execs_file),
            model_name=self.model_name,
            api_key=self.api_key,
            provider=self.provider,
            old_prompt=self.old_prompt,
            agent_name=self.agent_name,
            feedback=self.feedback,
            report_path=self.report_path,
        )

        result = self.reconciler.run_process()
        
        results_file = output_dir / "results.json"  # or whatever filename you want
        self.reconciler.save_results(
            results=result, output_file=str(results_file)
        )

        return result

    def save_logs_in_raw(self, trace_output) -> str:
        """
        Get the agent trace and save it to raw logs file.

        Returns:
            str: The execution trace as text.
        """
        self.logger.info("Starting agent trace retrieval...")

        if not trace_output:
            self.logger.warning("Agent trace retrieval produced empty trace output")
        else:
            self.logger.info(
                f"Agent trace retrieval completed, captured {len(trace_output)} characters of trace"
            )

            # Save raw trace to file
            try:
                # Try to parse as JSON first
                try:
                    json_data = json.loads(trace_output)
                    with open(self.raw_logs_path, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    # If not valid JSON, save as text
                    with open(self.raw_logs_path, "w", encoding="utf-8") as f:
                        f.write(trace_output)

                self.logger.info(f"Raw logs saved to {self.raw_logs_path}")
            except Exception as e:
                self.logger.error(f"Failed to save raw logs to file: {e}")

        return trace_output

    def execute_post_run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline: agent execution, log parsing, and validation.

        Returns:
            Dict containing paths to all output files and summary information.
        """
        self.logger.info("Starting full Adaptiq pipeline execution...")

        # Step 1: Run agent
        trace_output = self.base_config.get_agent_trace()
        self.save_logs_in_raw(trace_output=trace_output)
        # Step 2: Parse logs
        parsed_data = self.parse_logs(trace_output)

        # Step 3: Validate parsed logs (if enabled)
        validation_results = None
        corrected_logs = None


        # Load the raw and parsed logs from files
        with open(self.raw_logs_path, "r", encoding="utf-8") as f:
            raw_logs = json.load(f)

        with open(self.parsed_logs_path, "r", encoding="utf-8") as f:
            parsed_logs = json.load(f)

        _ , validation_results = self.validate_parsed_logs(
            raw_logs, parsed_logs
        )

        # Prepare pipeline results
        validation_results = {
            "outputs": {
                "raw_logs_path": self.raw_logs_path,
                "parsed_logs_path": self.parsed_logs_path,
            },
            "stats": {
                "raw_log_size": (
                    os.path.getsize(self.raw_logs_path)
                    if os.path.exists(self.raw_logs_path)
                    else 0
                ),
                "parsed_entries_count": len(parsed_data) if parsed_data else 0,
            },
        }


        validation_results["outputs"][
            "validated_logs_path"
        ] = self.validated_logs_path
        validation_results["outputs"][
            "validation_summary_path"
        ] = self.validation_summary_path

        if validation_results and "summary" in validation_results:
            validation_results["stats"]["validation_summary"] = validation_results[
                "summary"
            ]
        
        # Step 4: Reconciliate logs
        reconciliated_data = self.reconciliate_logs()

        pipeline_results = {
            "validation_results": validation_results,
            "reconciliation_results": reconciliated_data
        }

        self.logger.info("Full pipeline execution completed successfully")

        return pipeline_results



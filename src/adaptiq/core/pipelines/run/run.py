from datetime import datetime
import json
import logging

import os
from typing import Any, Dict, Optional, List
from adaptiq.core.abstract.integrations import BaseConfig, BasePromptParser, BaseLogParser
from adaptiq.core.entities import PostRunResults, PreRunResults
from adaptiq.core.pipelines import PreRunPipeline
from adaptiq.core.pipelines import PostRunPipeline
from adaptiq.core.reporting.aggregation import Aggregator
from adaptiq.core.entities import AdaptiQConfig

class AdaptiqRun:
    """
    Unified pipeline class that orchestrates both ADAPTIQ's pre-run and post-run modules:
    
    1. init_run: Executes the complete pre-run pipeline including:
       - Prompt Parsing
       - Hypothetical State Generation
       - Scenario Simulation
       - Q-table Initialization
       - Prompt Analysis and Estimation
    
    2. start_run: Executes the complete post-run pipeline including:
       - Log Parsing
       - Log Validation
       - Log Reconciliation
    
    This unified interface provides a single entry point for the complete ADAPTIQ workflow.
    """

    def __init__(
        self,
        base_config: BaseConfig,
        base_prompt_parser: BasePromptParser,
        base_log_parser: BaseLogParser,
        current_dir: str,
        template: str = "crew-ai",
        feedback: Optional[str] = None,
        prompt_auto_update: bool = False,
        save_results: bool = True,
        allow_pipeline: bool = True,
        

    ):
        """
        Initialize the unified RunPipeline with all required components.

        Args:
            base_config: An instance of BaseConfig (or its subclasses like CrewConfig, OpenAIConfig, etc.)
            base_prompt_parser: An instance of BasePromptParser for prompt parsing functionality
            base_log_parser: An instance of BaseLogParser for log parsing functionality
            output_path: Path where output files will be saved
            feedback: Optional feedback for post-run reconciliation
            validate_results: Whether to perform validation during post-run
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-RunPipeline")

        # Store configuration and components
        self.base_config = base_config
        self.adaptiq_config : AdaptiQConfig = self.base_config.get_config()
        self.base_prompt_parser = base_prompt_parser
        self.base_log_parser = base_log_parser
        self.current_dir = current_dir 
        self.output_path = current_dir + "/results"
        self.feedback = feedback
        self.save_results = save_results
        self.template =template
        self.prompt_auto_update = prompt_auto_update
        self.results_path = self.output_path  + "/adaptiq_results.json"
        self.allow_pipeline = allow_pipeline

        # Initialize pipeline components
        self.pre_run_pipeline = None
        self.post_run_pipeline = None
        self.aggerator = None

        # Results storage
        self.pre_run_results: PreRunResults = None
        self.post_run_results : PostRunResults = None

        self.logger.info("RunPipeline initialized successfully")

    def start_pre_run(self) -> PreRunResults:
        """
        Execute the complete pre-run pipeline to prepare the agent for execution.
        
        This includes:
        - Prompt parsing and analysis
        - Hypothetical state generation
        - Scenario simulation
        - Q-table initialization
        - Prompt optimization

        Args:
            save_results: Whether to save the results to files

        Returns:
            Dictionary containing all pre-run results including:
            - parsed_steps: List of parsed task steps
            - hypothetical_states: Generated state-action pairs
            - simulated_scenarios: Simulated execution scenarios
            - q_table_size: Size of initialized Q-table
            - prompt_analysis: Analysis of current prompt
            - new_prompt: Optimized system prompt

        Raises:
            Exception: If pre-run pipeline execution fails
        """
        self.logger.info("Starting init_run - Pre-run Pipeline Execution...")

        try:
            # Initialize the pre-run pipeline
            self.pre_run_pipeline = PreRunPipeline(
                base_config=self.base_config,
                base_prompt_parser=self.base_prompt_parser,
                output_path=self.output_path
            )

            # Execute the complete pre-run pipeline
            self.pre_run_results = self.pre_run_pipeline.execute_pre_run_pipeline(
                save_results=self.save_results
            )

            self.logger.info("init_run completed successfully")
            self.logger.info(
                "Pre-run results: %d parsed steps, %d hypothetical states, "
                "%d simulated scenarios, Q-table size: %d",
                len(self.pre_run_results.parsed_steps),
                len(self.pre_run_results.hypothetical_states),
                len(self.pre_run_results.simulated_scenarios),
                self.pre_run_results.q_table_size
            )

            return self.pre_run_results

        except Exception as e:
            self.logger.error("init_run failed: %s", str(e))
            raise

    def start_post_run(self) -> PostRunResults:
        """
        Execute the complete post-run pipeline to analyze agent execution results.
        
        This includes:
        - Log parsing from agent execution traces
        - Log validation and correction
        - Log reconciliation and analysis

        Returns:
            Dictionary containing all post-run results including:
            - validation_results: Results from log validation
            - reconciliation_results: Results from log reconciliation

        Raises:
            Exception: If post-run pipeline execution fails
        """
        self.logger.info("Starting start_run - Post-run Pipeline Execution...")

        try:
            # Initialize the post-run pipeline
            self.post_run_pipeline = PostRunPipeline(
                base_config=self.base_config,
                base_log_parser=self.base_log_parser,
                output_dir=self.output_path,
                feedback=self.feedback
            )

            # Execute the complete post-run pipeline
            self.post_run_results = self.post_run_pipeline.execute_post_run_pipeline()

            self.logger.info("start_run completed successfully")

            return self.post_run_results

        except Exception as e:
            self.logger.error("start_run failed: %s", str(e))
            raise

    def aggregate_run(
        self,
        agent_metrics: List[Dict] = None, 
        should_send_report: bool = False,
        ) -> Dict[str, Any]:
        """
        Aggregate results from post-run pipeline.

        Returns:
            Boolean indicating success of aggregation.
        """
        self.logger.info("Starting aggregation of run results...")

        if not self.post_run_results:
            raise ValueError("Post-run results must be available for aggregation.")

        # Initialize aggregator
        self.aggerator = Aggregator(
            config_data= self.adaptiq_config,
            original_prompt=self.base_config.get_prompt(get_newest=True),
        )

        reconciliation_results = self.post_run_results.reconciliation_results

        aggregated_results_status = self.aggerator.aggregate_results(
            agent_metrics=agent_metrics,
            validation_results=self.post_run_results.validation_data.stats.validation_results,
            reconciliation_results=reconciliation_results,
            should_send_report=should_send_report,
        )

        self.logger.info("Aggregation completed successfully")
        return aggregated_results_status
    
    def _verify_pre_run(self) -> bool:
        try:

            if os.path.isfile(self.results_path):
                return True
            else:
                self.logger.warning(f"File does not exist: {self.results_path}")
                return False

        except (OSError, TypeError) as e:
            self.logger.error(f"Error verifying results_path '{self.results_path}': {e}")
            return None

    def update_prompt(self, new_prompt: str, type: str):
            # update in memory config
            self.base_config.set_active_prompt(new_prompt=new_prompt)
            prompts_files = self.adaptiq_config.report_config.prompts_path
            prompts_path = os.path.join(self.current_dir, prompts_files)

            # save to prompt configuration if crew-ai template
            if self.template == "crew-ai" and self.prompt_auto_update:
                try:
                    task_path: str = self.adaptiq_config.agent_modifiable_config.prompt_configuration_file_path
                    self.base_config.update_instructions_within_file(
                        file_path=os.path.join(self.current_dir, task_path),
                        key="description"
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to update prompt in YAML file: {e}") from e

            # Save prompt into prompts file
            try:
                # load existing prompts if file exists and has content
                prompts_data = []
                if os.path.exists(prompts_path) and os.path.getsize(prompts_path) > 0:
                    with open(prompts_path, "r", encoding="utf-8") as f:
                        try:
                            prompts_data = json.load(f)
                        except json.JSONDecodeError:
                            prompts_data = []  # reset if corrupted

                # append new prompt with timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                prompts_data.append({
                    "timestamp": timestamp,
                    "type":type,
                    "prompt": new_prompt
                })

                # write back
                with open(prompts_path, "w", encoding="utf-8") as f:
                    json.dump(prompts_data, f, indent=4, ensure_ascii=False)

            except Exception as e:
                raise RuntimeError(f"Failed to save prompt in JSON file: {e}") from e
            
    def init_run(self,  func: callable, *args, **kwargs, ):
        """
        Executes a provided function with pre- and post-run prompt updates.
        Catches exceptions and raises a RuntimeError with the original error message.
        """

        try:
            results = None
            
            if self.allow_pipeline:
                if not self._verify_pre_run():
                    self.start_pre_run()
                    self.update_prompt(new_prompt=self.get_pre_run_prompt(), type="pre-run")

                start_time = datetime.now()
                #Exec the agent's main func
                results = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                self.logger.info(f"Function {func.__name__} completed in {duration:.3f}s")

            else:
                results = func(*args, **kwargs)
            
            return results

        except Exception as e:
            raise RuntimeError(f"Run execution failed: {e}") from e
        
    def run(self, agent_metrics: List[Dict[str, Any]]):
        try:
            if self.allow_pipeline:
                # order here matters ecause the aggregation need the old prompt
                self.start_post_run()
                self.aggregate_run(agent_metrics=agent_metrics)
                self.update_prompt(new_prompt=self.get_post_run_prompt(), type="post-run")     
        except Exception as e:
            raise RuntimeError(f"Run execution failed: {e}") from e

    def get_pre_run_results(self) -> Optional[PreRunResults]:
        """
        Get the results from the pre-run pipeline execution.

        Returns:
            Dictionary with pre-run results or None if init_run hasn't been called yet
        """
        return self.pre_run_results

    def get_post_run_results(self) -> Optional[PostRunResults]:
        """
        Get the results from the post-run pipeline execution.

        Returns:
            PostRunResults object or None if start_run hasn't been called yet
        """
        return self.post_run_results

    def get_pre_run_prompt(self) -> Optional[str]:
        """
        Get the optimized prompt generated during the pre-run phase.

        Returns:
            The optimized prompt string or None if init_run hasn't been executed
        """
        if self.pre_run_results:
            return self.pre_run_results.new_prompt
        return None
    
    def get_post_run_prompt(self) -> Optional[str]:
        """
        Get the prompt used during the post-run phase.

        Returns:
            The prompt string used in post-run or None if start_run hasn't been executed
        """
        if self.post_run_results:
            return self.post_run_results.reconciliation_results.summary.new_prompt
        return None

    # def get_pipeline_status(self) -> Dict[str, Any]:
    #     """
    #     Get the current status of both pipeline components.

    #     Returns:
    #         Dictionary with status information for both pipelines
    #     """
    #     status = {
    #         "pre_run_pipeline": {
    #             "initialized": self.pre_run_pipeline is not None,
    #             "completed": self.pre_run_results is not None,
    #             "results_available": bool(self.pre_run_results)
    #         },
    #         "post_run_pipeline": {
    #             "initialized": self.post_run_pipeline is not None,
    #             "completed": self.post_run_results is not None,
    #             "results_available": bool(self.post_run_results)
    #         }
    #     }

    #     # Add detailed status from pre-run pipeline if available
    #     if self.pre_run_pipeline:
    #         status["pre_run_details"] = self.pre_run_pipeline.get_status_summary()

    #     return status
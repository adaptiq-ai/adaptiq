import logging

from typing import Any, Dict, Optional, List
from adaptiq.core.abstract.integrations import BaseConfig, BasePromptParser, BaseLogParser
from adaptiq.core.pipelines import PreRunPipeline
from adaptiq.core.pipelines import PostRunPipeline
from adaptiq.core.reporting.aggregation import Aggregator


class RunPipeline:
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
        output_path: str,
        feedback: Optional[str] = None,
        validate_results: bool = True
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
        self.base_prompt_parser = base_prompt_parser
        self.base_log_parser = base_log_parser
        self.output_path = output_path
        self.feedback = feedback
        self.validate_results = validate_results

        # Initialize pipeline components
        self.pre_run_pipeline = None
        self.post_run_pipeline = None
        self.aggerator = None

        # Results storage
        self.pre_run_results = None
        self.post_run_results = None

        self.logger.info("RunPipeline initialized successfully")

    def init_run(self, save_results: bool = True) -> Dict[str, Any]:
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
                save_results=save_results
            )

            self.logger.info("init_run completed successfully")
            self.logger.info(
                "Pre-run results: %d parsed steps, %d hypothetical states, "
                "%d simulated scenarios, Q-table size: %d",
                len(self.pre_run_results.get("parsed_steps", [])),
                len(self.pre_run_results.get("hypothetical_states", [])),
                len(self.pre_run_results.get("simulated_scenarios", [])),
                self.pre_run_results.get("q_table_size", 0)
            )

            return self.pre_run_results

        except Exception as e:
            self.logger.error("init_run failed: %s", str(e))
            raise

    def start_run(self) -> Dict[str, Any]:
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
                feedback=self.feedback,
                validate_results=self.validate_results
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
        run_number: int = None
        ) -> Dict[str, Any]:
        """
        Aggregate results from post-run pipeline.

        Returns:
            Boolean indicating success of aggregation.
        """
        self.logger.info("Starting aggregation of run results...")

        if not self.pre_run_results or not self.post_run_results:
            raise ValueError("Both pre-run and post-run results must be available for aggregation.")

        # Initialize aggregator
        self.aggerator = Aggregator(
            config_data=self.base_config.get_config(),
            original_prompt=self.base_config.get_prompt(),
        )

        validation_summary_path = self.post_run_results.get(
            "validation_results", {}
        ).get("outputs", {}).get("validation_summary_path")

        reconciliation_results = self.post_run_results.get(
            "reconciliation_results", {}
        )

        aggregated_results_status = self.aggerator.aggregate_results(
            agent_metrics=agent_metrics,
            validation_summary_path=validation_summary_path,
            reconciliation_results=reconciliation_results,
            should_send_report=should_send_report,
            run_number=run_number
        )

        self.logger.info("Aggregation completed successfully")
        return aggregated_results_status

    def get_pre_run_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the results from the pre-run pipeline execution.

        Returns:
            Dictionary with pre-run results or None if init_run hasn't been called yet
        """
        return self.pre_run_results

    def get_post_run_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the results from the post-run pipeline execution.

        Returns:
            Dictionary with post-run results or None if start_run hasn't been called yet
        """
        return self.post_run_results

    def get_optimized_prompt(self) -> Optional[str]:
        """
        Get the optimized prompt generated during the pre-run phase.

        Returns:
            The optimized prompt string or None if init_run hasn't been executed
        """
        if self.pre_run_results and "new_prompt" in self.pre_run_results:
            return self.pre_run_results["new_prompt"]
        return None

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of both pipeline components.

        Returns:
            Dictionary with status information for both pipelines
        """
        status = {
            "pre_run_pipeline": {
                "initialized": self.pre_run_pipeline is not None,
                "completed": self.pre_run_results is not None,
                "results_available": bool(self.pre_run_results)
            },
            "post_run_pipeline": {
                "initialized": self.post_run_pipeline is not None,
                "completed": self.post_run_results is not None,
                "results_available": bool(self.post_run_results)
            }
        }

        # Add detailed status from pre-run pipeline if available
        if self.pre_run_pipeline:
            status["pre_run_details"] = self.pre_run_pipeline.get_status_summary()

        return status
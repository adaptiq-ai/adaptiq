import logging
from typing import Any, Dict, List

from adaptiq.core.reporting.aggregation.helpers import DataProcessor, MetricsCalculator, ReportBuilder
from adaptiq.core.reporting.monitoring.adaptiq_logger import AdaptiqLogger


class Aggregator:
    """
    AdaptiqAggregator class for tracking and aggregating metrics across multiple LLM runs.

    This class provides:
      - Tracking of token usage (input/output) for pre, post, and reconciliation steps.
      - Calculation of average and total costs based on configurable model pricing.
      - Measurement of execution time and error rates per run.
      - Aggregation of performance scores (rewards) and summary statistics.
      - Construction of per-run and overall project reports in JSON format.
      - Support for multiple LLM providers (OpenAI, Google) and dynamic config loading.

    Designed for use in LLM evaluation, benchmarking, and reporting pipelines.
    """
    # TODO: Unify the config to generic data model (from base config)
    # TODO: Add Start Aggregation that process incoming results and manage all logic inside the aggregator
    def __init__(self, config_data: Dict[str, Any], original_prompt: str):
        """Initialize the aggregator with pricing information for different models."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-Aggregator")
        self.original_prompt = original_prompt
        
        self.config_data = config_data
        self.email = self.config_data.get("email", "")

        # Initialize run tracking
        self._run_count = 0
        self._default_run_mode = True
        self.task_name = None

        # TODO: Move it to independant file 
        # Define pricing information
        self.pricings = {
            "openai": {
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
                "gpt-4.1": {"input": 0.002, "output": 0.008},
            }
        }

        # Initialize metrics calculator and report builder
        self.data_processor = DataProcessor()
        self.metrics_calculator = MetricsCalculator(self.config_data, self.pricings)
        self.report_builder = ReportBuilder(self.config_data)
        self.tracer = AdaptiqLogger.setup()

    def increment_run_count(self) -> int:
        """
        Increment and return the number of times the CLI run command has been executed.
        Returns:
            int: The current run count.
        """
        self._run_count += 1
        self.metrics_calculator.set_run_count(self._run_count)
        return self._run_count

    def calculate_avg_reward(
        self,
        validation_summary_path: str = None,
        simulated_scenarios: List = None,
        reward_type: str = "execution",
    ) -> float:
        """
        Calculate and update the running average reward across all runs.

        Args:
            validation_summary_path (str, optional): Path to the validation_summary.json file (for execution rewards).
            simulated_scenarios (list, optional): List of simulated scenario dicts (for simulation rewards).
            reward_type (str): "execution" or "simulation" to select which reward to calculate.

        Returns:
            float: The running average reward value, or 0.0 if none found.
        """
        return self.metrics_calculator.calculate_avg_reward(
            validation_summary_path, simulated_scenarios, reward_type
        )

    def update_avg_run_tokens(
        self,
        pre_input: int,
        pre_output: int,
        post_input: int,
        post_output: int,
        recon_input: int,
        recon_output: int,
        default_run_mode: bool = True,
    ):
        """
        Update the running sum for input/output tokens for each token type.

        Args:
            pre_input (int): Input tokens for pre_tokens.
            pre_output (int): Output tokens for pre_tokens.
            post_input (int): Input tokens for post_tokens.
            post_output (int): Output tokens for post_tokens.
            recon_input (int): Input tokens for recon_tokens.
            recon_output (int): Output tokens for recon_tokens.
            default_run_mode (bool): Whether this is a default mode (True) or not (False).
        """
        self._default_run_mode = default_run_mode
        return self.metrics_calculator.update_avg_run_tokens(
            pre_input, pre_output, post_input, post_output, 
            recon_input, recon_output, default_run_mode
        )

    def get_avg_run_tokens(self) -> tuple:
        """
        Get the overall average tokens: for each token type, average input/output, then average all three.
        Also return the average input tokens and average output tokens per run.

        Returns:
            tuple: (overall_avg, avg_input_tokens, avg_output_tokens)
        """
        return self.metrics_calculator.get_avg_run_tokens()

    def calculate_avg_cost(self) -> float:
        """
        Calculate the average cost based on avg_input and avg_output tokens,
        using the pricing info from self.pricings and model/provider from self.config.

        Returns:
            float: The average cost for the current averages.
        """
        return self.metrics_calculator.calculate_avg_cost()

    def calculate_current_run_cost(
        self, total_input_tokens: int, total_output_tokens: int
    ) -> float:
        """
        Calculate the cost for the current run based on total input and output tokens.

        Args:
            total_input_tokens (int): Total input tokens for the run.
            total_output_tokens (int): Total output tokens for the run.

        Returns:
            float: The cost for the current run.
        """
        return self.metrics_calculator.calculate_current_run_cost(
            total_input_tokens, total_output_tokens
        )

    def update_avg_run_time(self, run_time_seconds: float):
        """
        Update the running average of execution time per run.

        Args:
            run_time_seconds (float): The execution time for this run in seconds.
        """
        return self.metrics_calculator.update_avg_run_time(run_time_seconds)

    def get_avg_run_time(self) -> float:
        """
        Get the average execution time per run in seconds.

        Returns:
            float: The average run time in seconds.
        """
        return self.metrics_calculator.get_avg_run_time()

    def update_error_count(self, errors_this_run: int):
        """
        Update the running sum of errors across runs.
        Args:
            errors_this_run (int): Number of errors in this run.
        """
        return self.metrics_calculator.update_error_count(errors_this_run)

    def get_avg_errors(self) -> float:
        """
        Get the average number of errors per run.
        Returns:
            float: Average errors per run.
        """
        return self.metrics_calculator.get_avg_errors()

    def calculate_performance_score(self) -> float:
        """
        Calculate the performance score for the current run using internal state.

        Returns:
            float: The calculated performance score.
        """
        return self.metrics_calculator.calculate_performance_score()

    def parse_log_file(self, log_file_path: str, task_name: str) -> List[Dict[str, Any]]:
        """
        Parse a JSON log file and extract tool usage information.

        Args:
            log_file_path (str): Path to the JSON log file
            task_name (str): Task name to include in input_data

        Returns:
            List[Dict]: List of dictionaries containing tool usage information
        """
        return self.data_processor.parse_log_file(log_file_path, task_name)

    def estimate_prompt_tokens(
        self,  suggested_prompt: str, model_name: str = "gpt-4"
    ) -> tuple:
        """
        Estimate token counts for original and suggested prompts.

        Args:
            original_prompt (str): The original prompt text
            suggested_prompt (str): The suggested/optimized prompt text
            model_name (str): The model name for token encoding (default: "gpt-4")

        Returns:
            tuple: (original_tokens, suggested_tokens)
        """
        return self.metrics_calculator.estimate_prompt_tokens(
            self.original_prompt, suggested_prompt, model_name
        )

    def build_project_result(self) -> Dict:
        """
        Build a project overview JSON structure from the config and run count.
        Returns:
            dict: The project overview.
        """
        # Get metrics for summary
        avg_reward = (
            round(self.metrics_calculator.get_reward_sum() / self._run_count, 3) 
            if self._run_count else 0.0
        )
        overall_avg_tokens, _, _ = self.get_avg_run_tokens()
        total_cost = (
            round(self.calculate_avg_cost() * self._run_count, 3)
            if self._run_count else 0.0
        )
        avg_time = round(self.get_avg_run_time(), 2)
        avg_errors = self.get_avg_errors()
        error_rate = round((avg_errors / self._run_count) * 100, 1) if self._run_count else 0.0

        # Build summary metrics
        summary_metrics = self.report_builder.build_summary_metrics(
            total_runs=self._run_count,
            avg_reward=avg_reward,
            overall_avg_tokens=overall_avg_tokens,
            total_cost=total_cost,
            avg_time=avg_time,
            error_rate=error_rate,
        )

        return self.report_builder.build_project_result(self._run_count, summary_metrics)

    def build_run_summary(
        self,
        run_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        status: str,
        issues: List,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: List = None,
    ) -> Dict:
        """
        Build a summary JSON for a single run.
        """
        # Get task name and original prompt
        prompt_file_path = self.config_data.get("agent_modifiable_config", {}).get(
            "prompt_configuration_file_path", "N/A"
        )

        task_name = "Under-Fixing (Dev msg)"


        # Calculate token totals from metrics calculator
        pre = self.metrics_calculator.run_tokens["pre_tokens"]
        post = self.metrics_calculator.run_tokens["post_tokens"]
        recon = self.metrics_calculator.run_tokens["recon_tokens"]
        
        total_input_tokens = int(pre["input"] + post["input"] + recon["input"])
        total_output_tokens = int(pre["output"] + post["output"] + recon["output"])
        total_tokens = total_input_tokens + total_output_tokens

        # Set last run data for performance calculation
        self.metrics_calculator.set_last_run_data(
            reward, run_time_seconds or 0, self.original_prompt, suggested_prompt
        )

        # Calculate performance score and current run cost
        performance_score = self.calculate_performance_score()
        current_run_cost = self.calculate_current_run_cost(total_input_tokens, total_output_tokens)

        return self.report_builder.build_run_summary(
            run_number=self._run_count,
            run_name=run_name,
            task_name=task_name,
            reward=reward,
            api_calls=api_calls,
            suggested_prompt=suggested_prompt,
            original_prompt=self.original_prompt,
            status=status,
            issues=issues,
            performance_score=performance_score,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            current_run_cost=current_run_cost,
            error=error,
            memory_usage=memory_usage,
            run_time_seconds=run_time_seconds,
            execution_logs=execution_logs,
        )

    def add_run_summary(
        self,
        run_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        status: str,
        issues: List,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: List = None,
    ):
        """
        Build and add a run summary to the runs list.
        """
        # Get task name and original prompt
        prompt_file_path = self.config_data.get("agent_modifiable_config", {}).get(
            "prompt_configuration_file_path", "N/A"
        )
        

        task_name = "Under-Fixing (Dev msg)"

        # Calculate token totals from metrics calculator
        pre = self.metrics_calculator.run_tokens["pre_tokens"]
        post = self.metrics_calculator.run_tokens["post_tokens"]
        recon = self.metrics_calculator.run_tokens["recon_tokens"]
        
        total_input_tokens = int(pre["input"] + post["input"] + recon["input"])
        total_output_tokens = int(pre["output"] + post["output"] + recon["output"])
        total_tokens = total_input_tokens + total_output_tokens

        # Set last run data for performance calculation
        self.metrics_calculator.set_last_run_data(
            reward, run_time_seconds or 0, self.original_prompt, suggested_prompt
        )

        # Calculate performance score and current run cost
        performance_score = self.calculate_performance_score()
        current_run_cost = self.calculate_current_run_cost(total_input_tokens, total_output_tokens)

        self.report_builder.add_run_summary(
            run_number=self._run_count,
            run_name=run_name,
            task_name=task_name,
            reward=reward,
            api_calls=api_calls,
            suggested_prompt=suggested_prompt,
            original_prompt=self.original_prompt,
            status=status,
            issues=issues,
            performance_score=performance_score,
            total_tokens=total_tokens,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            current_run_cost=current_run_cost,
            error=error,
            memory_usage=memory_usage,
            run_time_seconds=run_time_seconds,
            execution_logs=execution_logs,
        )

    def get_runs_report(self) -> List[Dict]:
        """
        Get the report containing all runs.
        """
        return self.report_builder.get_runs_report()

    def create_error_info(
        self,
        exception,
        error_type: str = "pipeline_execution_error",
        severity: str = "Critical",
        include_stack_trace: bool = True,
    ) -> Dict:
        """
        Create error information dictionary from an exception.

        Args:
            exception (Exception): The caught exception
            error_type (str): Type of error (default: "pipeline_execution_error")
            severity (str): Severity level (default: "Critical")
            include_stack_trace (bool): Whether to include stack trace (default: True)

        Returns:
            dict: Error information dictionary
        """
        return self.report_builder.create_error_info(
            exception, error_type, severity, include_stack_trace
        )

    def build_run_details(
        self,
        exec_time: float,
        reward: float,
        timestamp: str = None,
        task_name: str = None,
        suggested_prompt: str = None,
        memory_usage: float = None,
        api_calls: int = None,
        error: str = None,
        execution_logs: List = None,
    ) -> Dict:
        """
        Build a detailed prompt analysis JSON for a single run.

        Args:
            exec_time (float): Execution time in seconds
            reward (float): Reward for this run
            timestamp (str, optional): Timestamp. If None, uses current UTC.
            task_name (str, optional): Task name
            original_prompt (str, optional): Original prompt text
            suggested_prompt (str, optional): Suggested/optimized prompt text
            memory_usage (float, optional): Memory usage in MB
            api_calls (int, optional): Number of API calls
            error (str, optional): Error information
            execution_logs (list, optional): List of execution log dicts.

        Returns:
            dict: The prompt analysis JSON.
        """
        # Get token totals
        pre = self.metrics_calculator.run_tokens["pre_tokens"]
        post = self.metrics_calculator.run_tokens["post_tokens"]
        recon = self.metrics_calculator.run_tokens["recon_tokens"]
        
        total_input_tokens = int(pre["input"] + post["input"] + recon["input"])
        total_output_tokens = int(pre["output"] + post["output"] + recon["output"])

        # Get log file path for tools (if not in default mode)
        log_file_path = None
        if not self._default_run_mode:
            log_file_path = (
                self.config_data.get("framework_adapter", {})
                .get("settings", {})
                .get("log_source", {})
                .get("path")
            )

        # Parse tools if needed
        tools_used = []
        if not self._default_run_mode and log_file_path and task_name:
            tools_used = self.parse_log_file(log_file_path, task_name)

        return self.report_builder.build_run_details(
            run_number=self._run_count,
            exec_time=exec_time,
            reward=reward,
            timestamp=timestamp,
            task_name=task_name,
            original_prompt=self.original_prompt,
            suggested_prompt=suggested_prompt,
            memory_usage=memory_usage or 0,
            api_calls=api_calls or 0,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            error=error,
            execution_logs=execution_logs,
            tools_used=tools_used,
            reward_sum=self.metrics_calculator.get_reward_sum(),
            run_count=self._run_count,
            input_price=self.metrics_calculator.input_price,
            default_run_mode=self._default_run_mode,
            log_file_path=log_file_path,
        )

    def send_run_results(self, data: Dict) -> bool:
        """
        Send the run results as a JSON payload to the configured project report endpoint.

        This method posts the provided data to the URL specified in self.url_report.
        It is used to deliver per-run or project summary results for further processing,
        storage, or notification (such as emailing the report to the user).

        Args:
            data (dict): The JSON payload containing run or project results.

        Returns:
            bool: True if the request was successful (HTTP 201), False otherwise.
        """
        return self.data_processor.send_run_results(data)

    def save_json_report(
        self, data: Dict[str, Any], filename: str = "default_run.json"
    ) -> str:
        """
        Save JSON data to a file in the reports_data folder.

        Args:
            data (Dict[str, Any]): The data to save as JSON
            filename (str): Name of the JSON file (default: "default_run.json")

        Returns:
            str: The absolute path of the saved file

        Raises:
            OSError: If there's an error creating the directory or writing the file
            TypeError: If the data is not JSON serializable
        """
        return self.data_processor.save_json_report(data, filename)

    def merge_json_reports(self, new_json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge new JSON data with existing default_run.json report.

        Args:
            new_json_data (Dict[str, Any]): New JSON data containing runs to merge

        Returns:
            Dict[str, Any]: Merged report with averaged summary metrics and combined runs

        Raises:
            FileNotFoundError: If default_run.json is not found
            ValueError: If project names don't match
            json.JSONDecodeError: If JSON file is corrupted
        """
        return self.data_processor.merge_json_reports(new_json_data)

    def reset_tracking(self):
        """Reset all tracking variables."""
        self._run_count = 0
        self.task_name = None
        self.metrics_calculator.reset_tracking()
        self.report_builder.clear_runs()

    # Additional helper methods for backward compatibility
    def get_run_count(self) -> int:
        """Get the current run count."""
        return self._run_count

    def get_config(self) -> Dict:
        """Get the current configuration."""
        return self.config_data

    def get_email(self) -> str:
        """Get the email from configuration."""
        return self.email

    def set_task_name(self, task_name: str):
        """Set the current task name."""
        self.task_name = task_name

    def set_last_run_data(
        self, reward: float, run_time_seconds: float = 0.0, suggested_prompt: str = ""
    ):
        """
        Set the last run data for performance calculation.
        
        Args:
            reward (float): Reward for the last run
            run_time_seconds (float): Execution time in seconds
            original_prompt (str): Original prompt text
            suggested_prompt (str): Suggested/optimized prompt text
        """
        self.metrics_calculator.set_last_run_data(
            reward, run_time_seconds, self.original_prompt, suggested_prompt
        )

    def aggregate_results(
    self, 
    agent_metrics: List[Dict] = None, 
    validation_summary_path: str = None,
    reconciliation_results: Dict = None,
    should_send_report: bool = False,
    run_number: int = None
    ) -> bool:
        """
        Aggregate results from agent metrics and build comprehensive reports.
        
        Args:
            agent_metrics (List[Dict], optional): List of agent execution metrics
            validation_summary_path (str, optional): Path to validation summary file
            reconciliation_results (Dict, optional): Results from reconciliation pipeline
            should_send_report (bool): Whether to send the report to endpoint
            run_number (int, optional): Run number for logging prefix
            
        Returns:
            bool: True if successful, False if error occurred
        """
        run_prefix = f"[RUN {run_number}] " if run_number is not None else ""
        
        try:
            # Process agent metrics and calculate totals
            total_execution_time = 0
            total_peak_memory = 0
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_successful_requests = 0

            if agent_metrics:
                logging.info("%sProcessing %s crew metrics entries...", run_prefix, len(agent_metrics))

                for i, metrics in enumerate(agent_metrics):
                    execution_time_seconds = metrics.get("execution_time_seconds", 0)
                    peak_memory_mb = metrics.get("peak_memory_mb", 0)
                    prompt_tokens = metrics.get("prompt_tokens", 0)
                    completion_tokens = metrics.get("completion_tokens", 0)
                    successful_requests = metrics.get("successful_requests", 0)

                    # Accumulate totals
                    total_execution_time += execution_time_seconds
                    total_peak_memory = max(total_peak_memory, peak_memory_mb)
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    total_successful_requests += successful_requests

                    logging.info(
                        "%sMetrics %s/%s: time=%.2fs, tokens=%s, requests=%s",
                        run_prefix, i + 1, len(agent_metrics), execution_time_seconds,
                        prompt_tokens + completion_tokens, successful_requests
                    )

            logging.info(
                "%sAgent token stats: input=%s, output=%s, calls=%s",
                run_prefix, total_prompt_tokens, total_completion_tokens, total_successful_requests
            )
            logging.info(
                "%sTotal execution time: %.2f seconds, Peak memory usage: %.2f MB",
                run_prefix, total_execution_time, total_peak_memory
            )

            # Calculate average reward
            avg_reward = self.calculate_avg_reward(
                validation_summary_path=validation_summary_path, 
                reward_type="execution"
            )
            
            new_prompt = reconciliation_results.get("summary", {}).get("new_prompt", "") if reconciliation_results else ""

            # Process each crew metrics entry and add to aggregator
            if agent_metrics:
                for i, metrics in enumerate(agent_metrics):
                    execution_time_seconds = metrics.get("execution_time_seconds", 0)
                    peak_memory_mb = metrics.get("peak_memory_mb", 0)
                    prompt_tokens = metrics.get("prompt_tokens", 0)
                    completion_tokens = metrics.get("completion_tokens", 0)
                    successful_requests = metrics.get("successful_requests", 0)
                    execution_count = metrics.get("execution_count", i + 1)

                    # Update aggregator with each run's data
                    self.increment_run_count()

                    # Update token statistics for this specific run
                    self.update_avg_run_tokens(
                        pre_input=0,
                        pre_output=0,
                        post_input=prompt_tokens,
                        post_output=completion_tokens,
                        recon_input=0,
                        recon_output=0,
                        default_run_mode=False,
                    )

                    # Update run time and error count
                    self.update_avg_run_time(execution_time_seconds)
                    self.update_error_count(0)

                    # Add run summary to the aggregator
                    self.add_run_summary(
                        run_name=f"{run_prefix}Execution-{execution_count}",
                        reward=avg_reward,
                        api_calls=successful_requests,
                        suggested_prompt=new_prompt,
                        status="completed",
                        issues=[],
                        error=None,
                        memory_usage=peak_memory_mb,
                        run_time_seconds=execution_time_seconds,
                        execution_logs=self.tracer.get_logs(),
                    )

                    logging.info("%sAdded run summary for execution %s", run_prefix, execution_count)
            else:
                # Handle case where no crew metrics provided - add single summary
                logging.info("%sNo crew metrics provided, adding single run summary...", run_prefix)

                self.increment_run_count()
                self.update_avg_run_tokens(
                    pre_input=0, pre_output=0, post_input=0, post_output=0,
                    recon_input=0, recon_output=0, default_run_mode=False,
                )
                self.update_avg_run_time(0)
                self.update_error_count(0)

                self.add_run_summary(
                    run_name=f"{run_prefix}Single-Run",
                    reward=avg_reward,
                    api_calls=0,
                    suggested_prompt=new_prompt,
                    status="completed",
                    issues=[],
                    error=None,
                    memory_usage=0,
                    run_time_seconds=0,
                    execution_logs=self.tracer.get_logs(),
                )

            # Send results if requested
            if should_send_report:
                logging.info("%sBuilding and sending comprehensive project results...", run_prefix)

                # Build project result JSON (now contains ALL runs)
                project_result = self.build_project_result()

                # Merge old with new result then save the new report
                merged_result = self.merge_json_reports(new_json_data=project_result)
                self.save_json_report(merged_result)

                # Send results to endpoint if email is configured
                if self.email != "":
                    success = self.send_run_results(merged_result)
                    if success:
                        logging.info("%sSuccessfully sent comprehensive run results to reporting endpoint", run_prefix)
                    else:
                        logging.warning("%sFailed to send run results to reporting endpoint", run_prefix)
                else:
                    logging.info("%sResults are successfully saved locally", run_prefix)
            else:
                logging.info("%sRun summaries added to aggregator - report will be sent when all runs complete", run_prefix)

            return True

        except Exception as e:
            logging.error("%sError during aggregation: %s", run_prefix, str(e))
            
            # Handle errors for all runs if agent_metrics provided
            if agent_metrics:
                for i, metrics in enumerate(agent_metrics):
                    execution_count = metrics.get("execution_count", i + 1)

                    # Update aggregator with error information for each run
                    self.increment_run_count()
                    self.update_error_count(1)
                    self.update_avg_run_time(0)

                    # Add failed run summary for each execution
                    self.add_run_summary(
                        run_name=f"{run_prefix}Execution-{execution_count}",
                        reward=0.0,
                        api_calls=0,
                        suggested_prompt="",
                        status="failed",
                        issues=["Aggregation failed"],
                        error=str(e),
                        memory_usage=0,
                        run_time_seconds=0,
                        execution_logs=self.tracer.get_logs(),
                    )
            else:
                # Handle single failed run
                self.increment_run_count()
                self.update_error_count(1)
                self.update_avg_run_time(0)

                self.add_run_summary(
                    run_name=f"{run_prefix}Single-Run",
                    reward=0.0,
                    api_calls=0,
                    suggested_prompt="",
                    status="failed",
                    issues=["Aggregation failed"],
                    error=str(e),
                    memory_usage=0,
                    run_time_seconds=0,
                    execution_logs=self.tracer.get_logs(),
                )

            # Send results even for failed runs if requested
            if should_send_report:
                logging.info("%sBuilding and sending project results for failed run...", run_prefix)
                project_result = self.build_project_result()
                self.send_run_results(project_result)
            else:
                logging.info("%sFailed run summary added to aggregator - report will be sent when all runs complete", run_prefix)

            return False
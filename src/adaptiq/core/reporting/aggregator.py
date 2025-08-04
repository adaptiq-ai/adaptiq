import datetime
import json
import logging
import os
import re
import traceback
import uuid
from copy import deepcopy
from typing import Any, Dict, List

import requests
import tiktoken
import yaml


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

    def __init__(self, config_path: str):
        """Initialize the aggregator with pricing information for different models."""
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-Aggregator")

        self.config_path = config_path
        self.config = self._load_config(self.config_path)
        self.email = self.config.get("email", "")
        self._run_count = 0
        self._reward_sum = 0.0

        self.avg_input_tokens = 0.0
        self.avg_output_tokens = 0.0
        self.avg_input = 0.0
        self.avg_output = 0.0
        self.overall_avg = 0.0

        self._total_run_time = 0.0
        self._total_errors = 0

        self.input_price = 0.0
        self.output_price = 0.0

        self.runs = []
        self._default_run_mode = True

        self.task_name = None
        self.url_report = "https://api.getadaptiq.io/projects"

        self.pricings = {
            "openai": {
                "gpt-4o": {"input": 0.0025, "output": 0.010},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4.1": {"input": 0.00185, "output": 0.0074},
                "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
                "gpt-4.1-nano": {"input": 0.00005, "output": 0.0002},
            }
        }

        self.run_tokens = {
            "pre_tokens": {"input": 0.0, "output": 0.0},
            "post_tokens": {"input": 0.0, "output": 0.0},
            "recon_tokens": {"input": 0.0, "output": 0.0},
        }

        # Initialize tracking variables
        self.reset_tracking()

    def reset_tracking(self):
        """Reset all tracking variables."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_history = []

        self.avg_input_tokens = 0.0
        self.avg_output_tokens = 0.0

    def _load_config(self, config_path: str) -> Dict:
        """
        Load and parse the ADAPTIQ configuration YAML file

        Args:
            config_path: Path to the configuration file

        Returns:
            dict: The parsed configuration
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.logger.info("Successfully loaded configuration from %s", config_path)
            return config
        except Exception as e:
            self.logger.error("Failed to load configuration: %s", str(e))
            raise

    def increment_run_count(self) -> int:
        """
        Increment and return the number of times the CLI run command has been executed.
        Returns:
            int: The current run count.
        """
        if not hasattr(self, "_run_count"):
            self._run_count = 0
        self._run_count += 1
        return self._run_count

    def calculate_avg_reward(
        self,
        validation_summary_path: str = None,
        simulated_scenarios: list = None,
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
        try:
            if reward_type == "simulation" and simulated_scenarios is not None:
                # Calculate average of reward_sim from simulated scenarios
                rewards = [
                    scenario.get("reward_sim")
                    for scenario in simulated_scenarios
                    if "reward_sim" in scenario
                ]
                if not rewards or self._run_count == 0:
                    return 0.0
                avg_this_run = sum(rewards) / len(rewards)
                self._reward_sum += avg_this_run
                return self._reward_sum / self._run_count

            elif reward_type == "execution" and validation_summary_path is not None:
                with open(validation_summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                rewards = [
                    entry["corrected_entry"]["reward_exec"]
                    for entry in data.get("validations", [])
                    if "corrected_entry" in entry
                    and "reward_exec" in entry["corrected_entry"]
                ]
                if not rewards or self._run_count == 0:
                    return 0.0
                avg_this_run = sum(rewards) / len(rewards)
                self._reward_sum += avg_this_run
                return self._reward_sum / self._run_count

            else:
                self.logger.error(
                    "Invalid arguments for calculate_avg_reward: must provide either validation_summary_path or simulated_scenarios."
                )
                return (
                    self._reward_sum / self._run_count if self._run_count > 0 else 0.0
                )

        except (OSError, json.JSONDecodeError, TypeError) as e:
            self.logger.error("Failed to calculate average reward: %s", e)
            return self._reward_sum / self._run_count if self._run_count > 0 else 0.0

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
            run_mode (bool): Whether this is a default mode (True) or not (False).
        """
        if self._run_count == 0:
            return  # Avoid division by zero

        # Update cumulative sums
        self.run_tokens["pre_tokens"]["input"] += pre_input
        self.run_tokens["pre_tokens"]["output"] += pre_output
        self.run_tokens["post_tokens"]["input"] += post_input
        self.run_tokens["post_tokens"]["output"] += post_output
        self.run_tokens["recon_tokens"]["input"] += recon_input
        self.run_tokens["recon_tokens"]["output"] += recon_output

        # Calculate this run's average input and output tokens
        if default_run_mode:
            avg_input_this_run = pre_input
            avg_output_this_run = pre_output
        else:
            avg_input_this_run = (
                pre_input + post_input + recon_input
            ) / self._run_count
            avg_output_this_run = (
                pre_output + post_output + recon_output
            ) / self._run_count

        # Add to running sums for averages
        self.avg_input_tokens += avg_input_this_run
        self.avg_output_tokens += avg_output_this_run

    def get_avg_run_tokens(self) -> tuple:
        """
        Get the overall average tokens: for each token type, average input/output, then average all three.
        Also return the average input tokens and average output tokens per run.
        Args:
            default_run_mode (bool): Whether this is a default mode (True) or not (False).
        Returns:
            tuple: (overall_avg, avg_input_tokens, avg_output_tokens)
        """
        if self._run_count == 0:
            return 0.0, 0.0, 0.0

        avg_pre = (
            self.run_tokens["pre_tokens"]["input"]
            + self.run_tokens["pre_tokens"]["output"]
        ) / self._run_count
        avg_post = (
            self.run_tokens["post_tokens"]["input"]
            + self.run_tokens["post_tokens"]["output"]
        ) / self._run_count
        avg_recon = (
            self.run_tokens["recon_tokens"]["input"]
            + self.run_tokens["recon_tokens"]["output"]
        ) / self._run_count

        if self._default_run_mode:
            self.overall_avg = avg_pre
        else:
            self.overall_avg = (avg_pre + avg_post + avg_recon) / self._run_count

        self.avg_input = self.avg_input_tokens / self._run_count
        self.avg_output = self.avg_output_tokens / self._run_count

        return self.overall_avg, self.avg_input, self.avg_output

    def calculate_avg_cost(self) -> float:
        """
        Calculate the average cost based on avg_input and avg_output tokens,
        using the pricing info from self.pricings and model/provider from self.config.

        Returns:
            float: The average cost for the current averages.
        """
        provider = self.config.get("llm_config", {}).get("provider")
        model = self.config.get("llm_config", {}).get("model_name")
        if not provider or not model:
            self.logger.error("Provider or model not found in config.")
            return 0.0

        pricing = self.pricings.get(provider, {}).get(model)
        if not pricing:
            self.logger.error(
                "Pricing not found for provider '%s' and model '%s'.", provider, model
            )
            return 0.0

        input_price = pricing.get("input", 0.0)
        output_price = pricing.get("output", 0.0)

        avg_cost = ((self.avg_input / 1000) * input_price) + (
            (self.avg_output / 1000) * output_price
        )
        return avg_cost

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
        provider = self.config.get("llm_config", {}).get("provider")
        model = self.config.get("llm_config", {}).get("model_name")
        if not provider or not model:
            self.logger.error("Provider or model not found in config.")
            return 0.0

        pricing = self.pricings.get(provider, {}).get(model)
        if not pricing:
            self.logger.error(
                "Pricing not found for provider '%s' and model '%s'.", provider, model
            )
            return 0.0

        input_price = pricing.get("input", 0.0)
        self.input_price = input_price
        output_price = pricing.get("output", 0.0)

        cost = ((total_input_tokens / 1000) * input_price) + (
            (total_output_tokens / 1000) * output_price
        )
        return cost

    def update_avg_run_time(self, run_time_seconds: float):
        """
        Update the running average of execution time per run.

        Args:
            run_time_seconds (float): The execution time for this run in seconds.
        """
        if not hasattr(self, "_total_run_time"):
            self._total_run_time = 0.0
        self._total_run_time += run_time_seconds

    def get_avg_run_time(self) -> float:
        """
        Get the average execution time per run in seconds.

        Returns:
            float: The average run time in seconds.
        """
        if self._run_count == 0 or not hasattr(self, "_total_run_time"):
            return 0.0
        return self._total_run_time / self._run_count

    def update_error_count(self, errors_this_run: int):
        """
        Update the running sum of errors across runs.
        Args:
            errors_this_run (int): Number of errors in this run.
        """
        if not hasattr(self, "_total_errors"):
            self._total_errors = 0
        self._total_errors += errors_this_run

    def get_avg_errors(self) -> float:
        """
        Get the average number of errors per run.
        Returns:
            float: Average errors per run.
        """
        if self._run_count == 0 or not hasattr(self, "_total_errors"):
            return 0.0
        return self._total_errors / self._run_count

    def calculate_performance_score(self) -> float:
        """
        Calculate the performance score for the current run using internal state.

        Returns:
            float: The calculated performance score.
        """
        # Use last run's reward (assume it's stored in self._last_reward or similar)
        reward = getattr(self, "_last_reward", 0.0)
        exec_time = getattr(self, "_last_run_time", 0.0)
        avg_errors = self.get_avg_errors()
        error_rate = (
            round((avg_errors / self._run_count) * 100, 1) if self._run_count else 0.0
        )

        # Calculate detail_added from last run's prompts if available
        original_prompt = getattr(self, "_last_original_prompt", "")
        suggested_prompt = getattr(self, "_last_suggested_prompt", "")
        orig_len = len(original_prompt) if original_prompt else 0
        sugg_len = len(suggested_prompt) if suggested_prompt else 0
        detail_added = ((sugg_len - orig_len) / orig_len) * 100 if orig_len > 0 else 0.0

        # Normalize metrics (example: reward out of 1, detail_added out of 100, error_rate out of 100, exec_time out of 60s)
        reward_norm = reward  # assuming reward is already 0-1 or 0-100
        detail_norm = min(max(detail_added / 100, 0), 1)
        error_norm = 1 - min(max(error_rate / 100, 0), 1)
        exec_time_norm = 1 - min(exec_time / 60, 1)  # 1 is best, 0 is worst if >60s

        # Weighted sum (adjust weights as needed)
        performance_score = round(
            0.5 * reward_norm
            + 0.2 * detail_norm
            + 0.2 * exec_time_norm
            + 0.1 * error_norm,
            3,
        )
        return performance_score

    def extract_report_fields(self, markdown_text):
        """
        Extracts timestamp, task name, original prompt, and suggested enhanced prompt from a markdown report.

        Args:
            markdown_text (str): The markdown report as a string.

        Returns:
            tuple: (timestamp, task_name, original_prompt, suggested_prompt)
        """
        # Timestamp (ISO format)
        timestamp_match = re.search(r"Date:\s*([0-9T:\.\-]+)", markdown_text)
        timestamp = timestamp_match.group(1) if timestamp_match else None

        # Task name
        task_match = re.search(
            r"# Prompt Engineering Report for Task: ([^\s\(]+)", markdown_text
        )
        self.task_name = task_match.group(1) if task_match else None

        # Original prompt (inside ```text ... ```)
        orig_prompt_match = re.search(
            r"## Original Prompt for Task.*?\n```text\n(.*?)\n```",
            markdown_text,
            re.DOTALL,
        )
        original_prompt = (
            orig_prompt_match.group(1).strip() if orig_prompt_match else None
        )

        # Suggested enhanced prompt (inside ```text ... ```)
        sugg_prompt_match = re.search(
            r"## Suggested Enhanced Prompt for Task.*?\n```text\n(.*?)\n```",
            markdown_text,
            re.DOTALL,
        )
        suggested_prompt = (
            sugg_prompt_match.group(1).strip() if sugg_prompt_match else None
        )

        return timestamp, self.task_name, original_prompt, suggested_prompt

    def parse_log_file(
        self, log_file_path: str, task_name: str
    ) -> List[Dict[str, Any]]:
        """
        Parse a JSON log file and extract tool usage information.

        Args:
            log_file_path (str): Path to the JSON log file
            task_name (str): Task name to include in input_data

        Returns:
            List[Dict]: List of dictionaries containing tool usage information
        """
        tools_used = []

        if not log_file_path:
            self.logger.error("Log file path is not provided.")
            return tools_used

        try:
            with open(log_file_path, "r", encoding="utf-8") as file:
                log_data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error reading JSON log file: {e}")
            return tools_used

        # Filter only AgentAction entries (tool usage)
        agent_actions = [
            entry for entry in log_data if entry.get("type") == "AgentAction"
        ]

        for i, action in enumerate(agent_actions):
            # Extract tool information
            tool_name = action.get("tool", "Unknown Tool")
            tool_result = action.get("result", "")
            timestamp = action.get("timestamp", "")

            # Calculate duration
            duration = "0s"  # Default
            if i < len(agent_actions) - 1:
                try:
                    current_time = datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S"
                    )
                    next_timestamp = agent_actions[i + 1].get("timestamp", "")
                    next_time = datetime.datetime.strptime(
                        next_timestamp, "%Y-%m-%d %H:%M:%S"
                    )
                    duration_seconds = (next_time - current_time).total_seconds()
                    duration = f"{duration_seconds:.2f}s"
                except (ValueError, TypeError):
                    duration = "N/A"

            # Use regex to check for "error" or "Error" in tool_result
            error_pattern = re.compile(r"\berror\b", re.IGNORECASE)
            has_error = bool(error_pattern.search(tool_result))

            # Determine status based on result
            status = "failed" if has_error else "success"

            # Set error message
            error_message = tool_result if has_error else None

            # Set output data
            output_data = (
                {"status": "completed", "result": tool_result}
                if status == "success"
                else {"error": "Tool execution failed"}
            )

            tool_info = {
                "name": tool_name.strip(),
                "status": status,
                "duration": duration,
                "error_message": error_message,
                "input_data": {
                    "task": task_name,
                    "timeout": 30,
                    "tool_input": action.get("tool_input", {}),
                },
                "output_data": output_data,
            }

            tools_used.append(tool_info)

        return tools_used

    def estimate_prompt_tokens(
        self, original_prompt, suggested_prompt, model_name="gpt-4"
    ):
        """
        Estimate token counts for original and suggested prompts.

        Args:
            original_prompt (str): The original prompt text
            suggested_prompt (str): The suggested/optimized prompt text
            model_name (str): The model name for token encoding (default: "gpt-4")

        Returns:
            tuple: (original_tokens, suggested_tokens)
        """
        try:
            # Get the appropriate encoding for the model
            encoding = tiktoken.encoding_for_model(model_name)

            # Encode and count tokens for original prompt
            original_tokens = 0
            if original_prompt:
                original_encoded = encoding.encode(original_prompt)
                original_tokens = len(original_encoded)

            # Encode and count tokens for suggested prompt
            suggested_tokens = 0
            if suggested_prompt:
                suggested_encoded = encoding.encode(suggested_prompt)
                suggested_tokens = len(suggested_encoded)

            return original_tokens, suggested_tokens

        except Exception as e:
            print(f"Error estimating tokens: {e}")
            # Fallback to rough estimation (4 chars per token)
            original_tokens = len(original_prompt) // 4 if original_prompt else 0
            suggested_tokens = len(suggested_prompt) // 4 if suggested_prompt else 0

            return original_tokens, suggested_tokens

    def build_project_result(self) -> dict:
        """
        Build a project overview JSON structure from the config and run count.
        Returns:
            dict: The project overview.
        """
        return {
            "email": self.config.get("email", ""),
            "overview": {
                "project_name": self.config.get("project_name", ""),
                "metadata": {
                    "agent_type": self.config.get("agent_modifiable_config", {}).get(
                        "agent_name", ""
                    ),
                    "total_runs_analyzed": getattr(self, "_run_count", 0),
                    "model": self.config.get("llm_config", {}).get("model_name", ""),
                },
                "summary_metrics": self.build_summary_metrics(),
            },
            "runs": self.get_runs_report(),
        }

    def build_summary_metrics(self) -> list:
        """
        Build the summaryMetrics JSON structure for reporting.

        Returns:
            list: List of summary metric dictionaries.
        """
        # Get values from aggregator
        total_runs = self._run_count
        avg_reward = (
            round(self._reward_sum / self._run_count, 3) if self._run_count else 0.0
        )
        overall_avg_tokens, _, _ = self.get_avg_run_tokens()
        total_cost = (
            round(self.calculate_avg_cost() * self._run_count, 3)
            if self._run_count
            else 0.0
        )
        avg_time = round(self.get_avg_run_time(), 2)
        avg_errors = self.get_avg_errors()
        error_rate = round((avg_errors / total_runs) * 100, 1) if total_runs else 0.0

        return [
            {
                "id": "total_runs",
                "icon": "hash",
                "label": "Total Runs",
                "description": "Executions analyzed",
                "value": total_runs,
                "unit": None,
            },
            {
                "id": "avg_reward",
                "icon": "target",
                "label": "Avg Reward",
                "description": "Performance score",
                "value": avg_reward,
                "unit": None,
            },
            {
                "id": "avg_tokens",
                "icon": "token",
                "label": "Avg Tokens",
                "description": "Token usage",
                "value": int(overall_avg_tokens),
                "unit": None,
            },
            {
                "id": "total_cost",
                "icon": "dollar",
                "label": "Total Cost",
                "description": "Cumulative spend",
                "value": total_cost,
                "unit": "$",
            },
            {
                "id": "avg_time",
                "icon": "clock",
                "label": "Avg Time",
                "description": "Execution duration",
                "value": avg_time,
                "unit": "s",
            },
            {
                "id": "error_rate",
                "icon": "error_triangle",
                "label": "Error Rate",
                "description": "Average failures",
                "value": error_rate,
                "unit": "%",
            },
        ]

    def build_run_summary(
        self,
        run_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        status: str,
        issues: list,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: list = None,
    ) -> dict:
        """
        Build a summary JSON for a single run.
        """
        run_id = str(uuid.uuid4())
        run_number = self._run_count
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        prompt_file_path = self.config.get("agent_modifiable_config", {}).get(
            "prompt_configuration_file_path", "N/A"
        )
        original_prompt_config = self._load_config(prompt_file_path)
        task_name = next(iter(original_prompt_config))
        original_prompt = original_prompt_config[task_name].get("description", "")

        # Calculate total tokens for this run (sum of last added tokens)
        pre = self.run_tokens["pre_tokens"]
        post = self.run_tokens["post_tokens"]
        recon = self.run_tokens["recon_tokens"]
        total_tokens = int(
            pre["input"]
            + pre["output"]
            + post["input"]
            + post["output"]
            + recon["input"]
            + recon["output"]
        )

        total_input_tokens = pre["input"] + post["input"] + recon["input"]
        total_output_tokens = pre["output"] + post["output"] + recon["output"]

        return {
            "run_id": run_id,
            "run_number": run_number,
            "task_name": task_name,
            "run_name": run_name,
            "performance_score": self.calculate_performance_score(),
            "status": status,
            "timestamp": timestamp,
            "prompt_snippet": original_prompt[:50] + "...",
            "issues": issues,
            "metrics": {
                "time": {"value": round(run_time_seconds, 2), "unit": "s"},
                "tokens": {"value": total_tokens, "unit": None},
                "memory": {"value": round(memory_usage, 2), "unit": "MB"},
                "cost": {
                    "value": self.calculate_current_run_cost(
                        total_input_tokens, total_output_tokens
                    ),
                    "unit": "$",
                },
            },
            "run_detail": self.build_run_details(
                exec_time=run_time_seconds,
                reward=reward,
                timestamp=timestamp,
                task_name=task_name,
                original_prompt=original_prompt,
                suggested_prompt=suggested_prompt,
                memory_usage=memory_usage,
                api_calls=api_calls,
                error=error,
                execution_logs=execution_logs,
            )
            or None,
        }

    def add_run_summary(
        self,
        run_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        status: str,
        issues: list,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: list = None,
    ):
        """
        Build and add a run summary to the runs list.
        """
        summary = self.build_run_summary(
            run_name,
            reward,
            api_calls,
            suggested_prompt,
            status,
            issues,
            error,
            memory_usage,
            run_time_seconds,
            execution_logs,
        )

        self.runs.append(summary)

    def get_runs_report(self) -> list:
        """
        Get the report containing all runs.
        """
        return self.runs

    def create_error_info(
        self,
        exception,
        error_type="pipeline_execution_error",
        severity="Critical",
        include_stack_trace=True,
    ):
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
        if exception is None:
            return []
        else:
            return {
                "error_type": error_type,
                "severity": severity,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "description": str(exception),
                "stack_trace": traceback.format_exc() if include_stack_trace else None,
            }

    def build_run_details(
        self,
        exec_time: float,
        reward: float,
        timestamp: str = None,
        task_name: str = None,
        original_prompt: str = None,
        suggested_prompt: str = None,
        memory_usage: float = None,
        api_calls: int = None,
        error: str = None,
        execution_logs: list = None,
    ) -> dict:
        """
        Build a detailed prompt analysis JSON for a single run.

        Args:
            run_number (int): The run number.
            markdown_text (str): The markdown report as a string.
            model (str, optional): Model name. If None, uses config.
            timestamp (str, optional): Timestamp. If None, uses current UTC.
            tools_used (list, optional): List of tool usage dicts.
            performance_metrics (dict, optional): Performance metrics dict.
            errors (list, optional): List of error dicts.
            execution_logs (list, optional): List of execution log dicts.

        Returns:
            dict: The prompt analysis JSON.
        """
        log_file_path = (
            self.config.get("framework_adapter", {})
            .get("settings", {})
            .get("log_source", {})
            .get("path")
        )

        # Use provided or fallback values
        execution_logs = execution_logs or []

        orig_len = len(original_prompt) if original_prompt else 0
        sugg_len = len(suggested_prompt) if suggested_prompt else 0

        if orig_len > 0:
            detail_added = ((sugg_len - orig_len) / orig_len) * 100
        else:
            detail_added = 0.0

        # Reward improvement calculation
        if self._reward_sum == 0:
            reward_improvement = (
                reward - self._reward_sum
            ) * 100  # e.g., absolute change
        else:
            reward_improvement = (
                (reward - self._reward_sum) / self._reward_sum
            ) * 100  # e.g., percent change

        avg_errors = self.get_avg_errors()
        error_rate = (
            round((avg_errors / self._run_count) * 100, 1) if self._run_count else 0.0
        )

        # Trend logic
        def get_trend(label, value):
            if label == "Reward Improvement":
                if value > 0.5:
                    return "positive"
                elif value < -0.5:
                    return "negative"
                else:
                    return "neutral"
            if label == "Detail Added":
                if value > 0:
                    return "positive"
                else:
                    return "neutral"
            if label == "Execution Time":
                if value > 10:
                    return "negative"
                else:
                    return "positive"
            if label == "Error Rate":
                if value > 10:
                    return "negative"
                else:
                    return "positive"
            return "neutral"

        summary_metrics = [
            {
                "label": "Reward Improvement",
                "value": f"{reward_improvement:+.1f}%",
                "trend": get_trend("Reward Improvement", reward_improvement),
            },
            {
                "label": "Detail Added",
                "value": f"{detail_added:.0f}%",
                "trend": get_trend("Detail Added", detail_added),
            },
            {
                "label": "Execution Time",
                "value": f"{exec_time:.2f}s",
                "trend": get_trend("Execution Time", exec_time),
            },
            {
                "label": "Error Rate",
                "value": f"{error_rate:.1f}%",
                "trend": get_trend("Error Rate", error_rate),
            },
        ]

        # Estimate tokens and costs
        original_tokens, suggested_tokens = self.estimate_prompt_tokens(
            original_prompt, suggested_prompt
        )

        return {
            "title": f"Prompt Analysis - Run #{self._run_count}",
            "task_name": task_name,
            "model": self.config.get("llm_config", {}).get("model_name", "unknown"),
            "timestamp": timestamp or datetime.datetime.utcnow().isoformat(),
            "prompt_analysis": {
                "original_text": original_prompt,
                "estimated_tokens": original_tokens,
                "estimated_cost": (
                    (original_tokens / 1000) * self.input_price
                    if original_tokens
                    else 0.0
                ),
                "suggestion_text": suggested_prompt,
                "optimized_tokens": suggested_tokens,
                "optimized_cost": (
                    (suggested_tokens / 1000) * self.input_price
                    if suggested_tokens
                    else 0.0
                ),
            },
            "execution_analysis": {
                "summary_metrics": summary_metrics,
                "tools_used": self.parse_log_file(log_file_path, task_name),
            },
            "performance_metrics": {
                "total_time_value": round(exec_time, 3),
                "total_time_unit": "s",
                "memory_peak_value": round(memory_usage, 3),
                "memory_peak_unit": "MB",
                "api_calls": api_calls,
                "retries": 0,
            },
            "errors": self.create_error_info(exception=error),
            "execution_logs": execution_logs,
        }

    def send_run_results(self, data: dict) -> bool:
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
        try:
            response = requests.post(self.url_report, json=data, timeout=30)
            if response.status_code == 201:
                return True
            else:
                print(
                    f"Request failed with status {response.status_code}: {response.text}"
                )
                return False
        except requests.RequestException as e:
            print(f"An error occurred: {e}")
            return False

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
        try:
            # Get the absolute path of the directory where the script is executed
            script_dir = os.path.abspath(os.getcwd())

            # Create the reports_data folder path
            reports_folder = os.path.join(script_dir, "reports_data")

            # Create the reports_data directory if it doesn't exist
            os.makedirs(reports_folder, exist_ok=True)

            # Create the full file path
            file_path = os.path.join(reports_folder, filename)

            # Save the JSON data to the file
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

            print(f"JSON file saved successfully at: {file_path}")
            return file_path

        except (OSError, IOError) as e:
            print(f"Error creating directory or writing file: {e}")
            raise
        except (TypeError, ValueError) as e:
            print(f"Error serializing data to JSON: {e}")
            raise

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
        try:
            # Get the absolute path of the directory where the script is executed
            script_dir = os.path.abspath(os.getcwd())

            # Look for default_run.json in reports_data folder
            default_file_path = os.path.join(
                script_dir, "reports_data", "default_run.json"
            )

            # Check if file exists
            if not os.path.exists(default_file_path):
                raise FileNotFoundError(
                    f"default_run.json not found at: {default_file_path}"
                )

            # Load the existing default report
            with open(default_file_path, "r", encoding="utf-8") as file:
                default_report = json.load(file)

            # Check if project names match
            default_project = default_report.get("overview", {}).get("project_name", "")
            new_project = new_json_data.get("overview", {}).get("project_name", "")

            if default_project != new_project:
                raise ValueError(
                    f"Project names don't match: '{default_project}' vs '{new_project}'"
                )

            # Create a deep copy of the default report to avoid modifying the original
            merged_report = deepcopy(default_report)

            # Get runs from both reports
            default_runs = default_report.get("runs", [])
            new_runs = new_json_data.get("runs", [])

            # Calculate total runs for averaging
            total_runs = len(default_runs) + len(new_runs)

            if total_runs == 0:
                print("Warning: No runs found in either report")
                return merged_report

            # Update metadata
            merged_report["overview"]["metadata"]["total_runs_analyzed"] = total_runs

            # Merge and average summary metrics
            default_summary = default_report.get("overview", {}).get(
                "summary_metrics", []
            )
            new_summary = new_json_data.get("overview", {}).get("summary_metrics", [])

            # Create a mapping of metric IDs to their data for easier processing
            default_metrics = {metric["id"]: metric for metric in default_summary}
            new_metrics = {metric["id"]: metric for metric in new_summary}

            # Define the desired order of metrics based on your JSON example
            metric_order = [
                "total_runs",
                "avg_reward",
                "avg_tokens",
                "total_cost",
                "avg_time",
                "error_rate",
            ]

            # Merge and average the metrics in the specified order
            merged_summary_metrics = []

            # Process metrics in the desired order
            for metric_id in metric_order:
                default_metric = default_metrics.get(metric_id)
                new_metric = new_metrics.get(metric_id)

                if default_metric and new_metric:
                    # Special handling for total_runs - use actual count, not average
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        # Both reports have this metric - calculate average
                        default_value = default_metric.get("value", 0)
                        new_value = new_metric.get("value", 0)

                        # Calculate weighted average based on number of runs
                        default_weight = len(default_runs)
                        new_weight = len(new_runs)

                        if isinstance(default_value, (int, float)) and isinstance(
                            new_value, (int, float)
                        ):
                            averaged_value = (
                                default_value * default_weight + new_value * new_weight
                            ) / total_runs

                            # Round to 3 decimal places for readability
                            if isinstance(averaged_value, float):
                                averaged_value = round(averaged_value, 3)
                        else:
                            # If values are not numeric, keep the default value
                            averaged_value = default_value

                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = averaged_value
                        merged_summary_metrics.append(merged_metric)

                elif default_metric:
                    # Only default report has this metric
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        merged_summary_metrics.append(deepcopy(default_metric))
                elif new_metric:
                    # Only new report has this metric
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(new_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        merged_summary_metrics.append(deepcopy(new_metric))

            # Handle any additional metrics not in the predefined order
            all_metric_ids = set(default_metrics.keys()) | set(new_metrics.keys())
            remaining_metrics = all_metric_ids - set(metric_order)

            for metric_id in remaining_metrics:
                default_metric = default_metrics.get(metric_id)
                new_metric = new_metrics.get(metric_id)

                if default_metric and new_metric:
                    # Special handling for total_runs - use actual count, not average
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        # Both reports have this metric - calculate average
                        default_value = default_metric.get("value", 0)
                        new_value = new_metric.get("value", 0)

                        # Calculate weighted average based on number of runs
                        default_weight = len(default_runs)
                        new_weight = len(new_runs)

                        if isinstance(default_value, (int, float)) and isinstance(
                            new_value, (int, float)
                        ):
                            averaged_value = (
                                default_value * default_weight + new_value * new_weight
                            ) / total_runs

                            # Round to 3 decimal places for readability
                            if isinstance(averaged_value, float):
                                averaged_value = round(averaged_value, 3)
                        else:
                            # If values are not numeric, keep the default value
                            averaged_value = default_value

                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = averaged_value
                        merged_summary_metrics.append(merged_metric)

                elif default_metric:
                    # Only default report has this metric
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(default_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        merged_summary_metrics.append(deepcopy(default_metric))
                elif new_metric:
                    # Only new report has this metric
                    if metric_id == "total_runs":
                        merged_metric = deepcopy(new_metric)
                        merged_metric["value"] = total_runs
                        merged_summary_metrics.append(merged_metric)
                    else:
                        merged_summary_metrics.append(deepcopy(new_metric))

            # Update the merged report's summary metrics
            merged_report["overview"]["summary_metrics"] = merged_summary_metrics

            # Merge runs and update run_number sequentially
            merged_runs = []

            # Add default runs first (keeping their original run_number or updating if needed)
            for i, run in enumerate(default_runs, 1):
                updated_run = deepcopy(run)
                updated_run["run_number"] = i
                merged_runs.append(updated_run)

            # Add new runs with incremented run_number
            for i, run in enumerate(new_runs, len(default_runs) + 1):
                updated_run = deepcopy(run)
                updated_run["run_number"] = i
                merged_runs.append(updated_run)

            merged_report["runs"] = merged_runs

            return merged_report

        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"Error: {e}")
            raise

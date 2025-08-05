import datetime
import logging
import traceback
import uuid
from typing import Any, Dict, List, Optional


class ReportBuilder:
    """
    Handles all report generation, JSON structure building, and formatting
    for project summaries, run details, and performance analysis.
    """

    def __init__(self, config: Dict):
        """
        Initialize the report builder.

        Args:
            config (Dict): Configuration dictionary containing project and model info
        """
        self.logger = logging.getLogger("ADAPTIQ-Aggregator-ReportBuilder")
        self.config = config
        self.runs = []

    def build_project_result(self, total_runs: int, summary_metrics: List[Dict]) -> Dict:
        """
        Build a project overview JSON structure from the config and run count.
        
        Args:
            total_runs (int): Total number of runs analyzed
            summary_metrics (List[Dict]): List of summary metrics
            
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
                    "total_runs_analyzed": total_runs,
                    "model": self.config.get("llm_config", {}).get("model_name", ""),
                },
                "summary_metrics": summary_metrics,
            },
            "runs": self.runs,
        }

    def build_summary_metrics(
        self,
        total_runs: int,
        avg_reward: float,
        overall_avg_tokens: float,
        total_cost: float,
        avg_time: float,
        error_rate: float,
    ) -> List[Dict]:
        """
        Build the summaryMetrics JSON structure for reporting.

        Args:
            total_runs (int): Total number of runs
            avg_reward (float): Average reward across runs
            overall_avg_tokens (float): Average tokens per run
            total_cost (float): Total cost across runs
            avg_time (float): Average execution time
            error_rate (float): Error rate percentage

        Returns:
            list: List of summary metric dictionaries.
        """
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
                "value": round(avg_reward, 3),
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
                "value": round(total_cost, 3),
                "unit": "$",
            },
            {
                "id": "avg_time",
                "icon": "clock",
                "label": "Avg Time",
                "description": "Execution duration",
                "value": round(avg_time, 2),
                "unit": "s",
            },
            {
                "id": "error_rate",
                "icon": "error_triangle",
                "label": "Error Rate",
                "description": "Average failures",
                "value": round(error_rate, 1),
                "unit": "%",
            },
        ]

    def build_run_summary(
        self,
        run_number: int,
        run_name: str,
        task_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        original_prompt: str,
        status: str,
        issues: List,
        performance_score: float,
        total_tokens: int,
        total_input_tokens: int,
        total_output_tokens: int,
        current_run_cost: float,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: List = None,
    ) -> Dict:
        """
        Build a summary JSON for a single run.
        
        Args:
            run_number (int): Sequential run number
            run_name (str): Name of the run
            task_name (str): Name of the task
            reward (float): Reward/performance score for this run
            api_calls (int): Number of API calls made
            suggested_prompt (str): The suggested/optimized prompt
            original_prompt (str): The original prompt
            status (str): Run status (success, failed, etc.)
            issues (List): List of issues encountered
            performance_score (float): Calculated performance score
            total_tokens (int): Total tokens used in run
            total_input_tokens (int): Total input tokens
            total_output_tokens (int): Total output tokens
            current_run_cost (float): Cost for this specific run
            error (str, optional): Error message if any
            memory_usage (float, optional): Memory usage in MB
            run_time_seconds (float, optional): Execution time in seconds
            execution_logs (List, optional): List of execution logs
            
        Returns:
            dict: Run summary dictionary
        """
        run_id = str(uuid.uuid4())
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"

        return {
            "run_id": run_id,
            "run_number": run_number,
            "task_name": task_name,
            "run_name": run_name,
            "performance_score": performance_score,
            "status": status,
            "timestamp": timestamp,
            "prompt_snippet": original_prompt[:50] + "..." if original_prompt else "N/A",
            "issues": issues,
            "metrics": {
                "time": {"value": round(run_time_seconds or 0, 2), "unit": "s"},
                "tokens": {"value": total_tokens, "unit": None},
                "memory": {"value": round(memory_usage or 0, 2), "unit": "MB"},
                "cost": {"value": round(current_run_cost, 4), "unit": "$"},
            },
            "run_detail": self.build_run_details(
                run_number=run_number,
                exec_time=run_time_seconds or 0,
                reward=reward,
                timestamp=timestamp,
                task_name=task_name,
                original_prompt=original_prompt,
                suggested_prompt=suggested_prompt,
                memory_usage=memory_usage or 0,
                api_calls=api_calls,
                error=error,
                execution_logs=execution_logs or [],
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            ),
        }

    def build_run_details(
        self,
        run_number: int,
        exec_time: float,
        reward: float,
        timestamp: str,
        task_name: str,
        original_prompt: str,
        suggested_prompt: str,
        memory_usage: float,
        api_calls: int,
        total_input_tokens: int,
        total_output_tokens: int,
        error: str = None,
        execution_logs: List = None,
        tools_used: List = None,
        reward_sum: float = 0.0,
        run_count: int = 1,
        input_price: float = 0.0,
        default_run_mode: bool = True,
        log_file_path: str = None,
    ) -> Dict:
        """
        Build a detailed prompt analysis JSON for a single run.

        Args:
            run_number (int): The run number
            exec_time (float): Execution time in seconds
            reward (float): Reward for this run
            timestamp (str): Timestamp of the run
            task_name (str): Name of the task
            original_prompt (str): Original prompt text
            suggested_prompt (str): Suggested/optimized prompt text
            memory_usage (float): Memory usage in MB
            api_calls (int): Number of API calls
            total_input_tokens (int): Total input tokens
            total_output_tokens (int): Total output tokens
            error (str, optional): Error information
            execution_logs (List, optional): Execution logs
            tools_used (List, optional): List of tools used
            reward_sum (float): Cumulative reward sum for improvement calculation
            run_count (int): Total run count for calculations
            input_price (float): Price per 1000 input tokens
            default_run_mode (bool): Whether in default run mode
            log_file_path (str, optional): Path to log file for tool parsing

        Returns:
            dict: Detailed run analysis
        """
        # Calculate metrics for summary
        orig_len = len(original_prompt) if original_prompt else 0
        sugg_len = len(suggested_prompt) if suggested_prompt else 0

        if orig_len > 0:
            detail_added = ((sugg_len - orig_len) / orig_len) * 100
        else:
            detail_added = 0.0

        # Reward improvement calculation
        if reward_sum == 0:
            reward_improvement = reward * 100  # First run, use absolute value
        else:
            avg_previous_reward = reward_sum / run_count if run_count > 0 else 0
            if avg_previous_reward == 0:
                reward_improvement = reward * 100
            else:
                reward_improvement = ((reward - avg_previous_reward) / avg_previous_reward) * 100

        # Error rate calculation (simplified for single run)
        error_rate = 100.0 if error else 0.0

        # Trend logic
        def get_trend(label: str, value: float) -> str:
            if label == "Reward Improvement":
                if value > 0.5:
                    return "positive"
                elif value < -0.5:
                    return "negative"
                else:
                    return "neutral"
            elif label == "Detail Added":
                if value > 0:
                    return "positive"
                else:
                    return "neutral"
            elif label == "Execution Time":
                if value > 10:
                    return "negative"
                else:
                    return "positive"
            elif label == "Error Rate":
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

        # Estimate tokens and costs using tiktoken approximation
        original_tokens = len(original_prompt) // 4 if original_prompt else 0
        suggested_tokens = len(suggested_prompt) // 4 if suggested_prompt else 0

        return {
            "title": f"Prompt Analysis - Run #{run_number}",
            "task_name": task_name,
            "model": self.config.get("llm_config", {}).get("model_name", "unknown"),
            "timestamp": timestamp,
            "prompt_analysis": {
                "original_text": original_prompt,
                "estimated_tokens": original_tokens,
                "estimated_cost": round((original_tokens / 1000) * input_price, 4) if original_tokens else 0.0,
                "suggestion_text": suggested_prompt,
                "optimized_tokens": suggested_tokens,
                "optimized_cost": round((suggested_tokens / 1000) * input_price, 4) if suggested_tokens else 0.0,
            },
            "execution_analysis": {
                "summary_metrics": summary_metrics,
                "tools_used": tools_used or [],
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
            "execution_logs": execution_logs or [],
        }

    def add_run_summary(
        self,
        run_number: int,
        run_name: str,
        task_name: str,
        reward: float,
        api_calls: int,
        suggested_prompt: str,
        original_prompt: str,
        status: str,
        issues: List,
        performance_score: float,
        total_tokens: int,
        total_input_tokens: int,
        total_output_tokens: int,
        current_run_cost: float,
        error: str = None,
        memory_usage: float = None,
        run_time_seconds: float = None,
        execution_logs: List = None,
    ):
        """
        Build and add a run summary to the runs list.
        
        Args:
            Same as build_run_summary
        """
        summary = self.build_run_summary(
            run_number=run_number,
            run_name=run_name,
            task_name=task_name,
            reward=reward,
            api_calls=api_calls,
            suggested_prompt=suggested_prompt,
            original_prompt=original_prompt,
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

        self.runs.append(summary)

    def get_runs_report(self) -> List[Dict]:
        """
        Get the report containing all runs.
        
        Returns:
            List[Dict]: List of run summaries
        """
        return self.runs

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
        if exception is None:
            return {}
        else:
            return {
                "error_type": error_type,
                "severity": severity,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "description": str(exception),
                "stack_trace": traceback.format_exc() if include_stack_trace else None,
            }

    def clear_runs(self):
        """Clear all stored runs."""
        self.runs = []

    def get_run_count(self) -> int:
        """Get the number of runs stored."""
        return len(self.runs)

    def get_latest_run(self) -> Optional[Dict]:
        """
        Get the most recently added run.
        
        Returns:
            Optional[Dict]: Latest run summary or None if no runs exist
        """
        return self.runs[-1] if self.runs else None

    def update_run_status(self, run_id: str, new_status: str):
        """
        Update the status of a specific run.
        
        Args:
            run_id (str): The run ID to update
            new_status (str): The new status to set
        """
        for run in self.runs:
            if run.get("run_id") == run_id:
                run["status"] = new_status
                break
        else:
            self.logger.warning(f"Run with ID {run_id} not found for status update")

    def get_runs_by_status(self, status: str) -> List[Dict]:
        """
        Get all runs with a specific status.
        
        Args:
            status (str): The status to filter by
            
        Returns:
            List[Dict]: List of runs with the specified status
        """
        return [run for run in self.runs if run.get("status") == status]
import datetime
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, List

from adaptiq.cloud.adaptiq_client import AdaptiqCloud


class DataProcessor:
    """
    Handles all file I/O operations, parsing, data extraction, and external communication
    for the ADAPTIQ aggregation system.
    """

    def __init__(self):
        """Initialize the data processor."""
        self.logger = logging.getLogger("ADAPTIQ-Aggregator-DataProcessor")
        self.adaptiq_cloud = AdaptiqCloud()

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

        try:
            with open(log_file_path, "r", encoding="utf-8") as file:
                log_data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error reading JSON log file: {e}")

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

    def send_run_results(self, data: Dict) -> bool:
        """
        Send the run results using the Adaptiq client.

        Args:
            data (dict): The JSON payload containing run or project results.

        Returns:
            bool: True if the request was successful, False otherwise.
        """
        return self.adaptiq_cloud.send_run_results(data)

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
                print(
                    "No existing default_run.json found. Using new data as initial report."
                )
                return new_json_data

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

import json
import math
import os
from typing import Any, Dict, List, Union


class AdaptiqLogParser:
    """
    AdaptiqLogParser transforms raw CrewAI logs into a state-action-reward mapping
    suitable for training or evaluation purposes.

    This class processes different log entry types (AgentAction, AgentFinish, TaskLog) and calculates
    a normalized reward signal based on rule-based heuristics, such as the quality of thoughts,
    tool usage success, output length, and error detection.
    """

    # --- Constants for Reward Calculation ---

    # Thresholds for thought/output quality
    MIN_MEANINGFUL_THOUGHT_LEN = 250
    SHORT_OUTPUT_LEN_THRESHOLD = 500
    MEDIUM_OUTPUT_LEN_THRESHOLD = 1000

    # General
    BONUS_GOOD_THOUGHT = 0.15
    PENALTY_POOR_THOUGHT = -0.15  # For empty/placeholder/very short thoughts

    # AgentAction: Tool Usage
    REWARD_TOOL_SUCCESS = 1.0
    REWARD_TOOL_SUCCESS_EMPTY_RESULT = (
        0.25  # Tool worked, but result was empty (e.g., "", [], {})
    )
    PENALTY_TOOL_ERROR = -1.0
    PENALTY_TOOL_NO_RESULT_FIELD = -0.75  # Tool was called, but 'result' key is missing
    PENALTY_TOOL_NAME_EMPTY_STRING = (
        -1.0
    )  # If 'tool' field is an empty string in AgentAction

    # AgentAction: Thinking (No Tool)
    REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT = (
        0.3  # For AgentAction with no tool and good thought
    )
    PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT = (
        -0.3
    )  # For AgentAction with no tool and poor thought

    # AgentFinish: Final Output
    REWARD_FINAL_OUTPUT_LONG = 0.75
    REWARD_FINAL_OUTPUT_MEDIUM = 0.5
    REWARD_FINAL_OUTPUT_SHORT = 0.2
    PENALTY_FINAL_OUTPUT_EMPTY_OR_PLACEHOLDER = -0.5

    # TaskLog
    REWARD_TASKLOG_HAS_DESCRIPTION = 0.25
    PENALTY_TASKLOG_NO_DESCRIPTION = -0.25
    REWARD_TASKLOG_HAS_RAW = 0.5
    PENALTY_TASKLOG_NO_RAW = -0.5
    PENALTY_TASKLOG_RAW_CONTAINS_ERROR = (
        -0.75
    )  # If raw output in TaskLog indicates an error

    # Keywords
    ERROR_KEYWORDS = [
        "error:",
        "traceback:",
        "failed to execute",
        "exception:",
        "failure:",
    ]
    PLACEHOLDER_STRINGS_LOWER = [
        "none",
        "n/a",
        "missing thought",
        "empty thought",
        "task log content",
        "null",
    ]

    # Action representations for keys
    ACTION_AGENT_THOUGHT_PROCESS = "AgentThoughtProcess"
    ACTION_INVALID_TOOL_EMPTY_NAME = "InvalidTool(EmptyName)"
    ACTION_FINAL_ANSWER = "FinalAnswer"
    TASKLOG_NO_RAW_OUTPUT_REPR = "NoRawOutputInTaskLog"

    def __init__(self, logs_path: str, output_path: str):
        """
        Initialize the AdaptiqLogParser with the path to the logs file.

        Args:
            logs_path (str): Path to a CrewAI log file (JSON format).
            output_path (str): Path where the processed logs will be saved.
        """
        self.logs_path = logs_path
        self.output_path = output_path

    def load_json_data(self) -> Union[Dict, List[Dict[str, Any]]]:
        """
        Loads and parses a JSON file, ensuring it exists and is properly formatted.

        Returns:
            Union[Dict, List[Dict]]: Parsed JSON content — either a dictionary or a list of dictionaries.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            json.JSONDecodeError: If the file content is not valid JSON.
            ValueError: If the JSON is neither a dict nor a list of dicts.
        """
        if not os.path.isfile(self.logs_path):
            raise FileNotFoundError(f"File not found: {self.logs_path}")

        with open(self.logs_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON in file '{self.logs_path}': {e.msg}", e.doc, e.pos
                )

        if isinstance(data, dict):
            return data
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        else:
            raise ValueError(
                "JSON content must be either a dictionary or a list of dictionaries."
            )

    @staticmethod
    def normalize_reward(reward: float) -> float:
        """
        Normalize using "hyperbolic tangent function" to force floats between -1 and 1.

        Args:
            reward (float): The raw reward value.

        Returns:
            float: The normalized reward value between -1 and 1.
        """
        return round(math.tanh(reward), 4)

    @classmethod
    def is_string_effectively_empty_or_placeholder(cls, s: Any) -> bool:
        """
        Checks if a string is None, empty, whitespace only, or a known placeholder.

        Args:
            s (Any): The string to check.

        Returns:
            bool: True if the string is effectively empty or a placeholder, False otherwise.
        """
        if s is None:
            return True
        s_str = str(s).strip()
        if not s_str:  # Empty after stripping
            return True
        return s_str.lower() in cls.PLACEHOLDER_STRINGS_LOWER

    def parse_logs(self) -> Dict[str, List[Dict]]:
        """
        Transforms raw CrewAI logs into a state-action-reward mapping suitable for training or evaluation purposes.

        This method processes different log entry types (AgentAction, AgentFinish, TaskLog) and calculates
        a normalized reward signal based on rule-based heuristics, such as the quality of thoughts, tool usage success,
        output length, and error detection. It correctly maintains the chronological flow of actions.

        Returns:
            Dict[str, List[Dict]]: A processed log data structure with chronological steps and rewards.
        """
        logs: List[Dict[str, Any]] = self.load_json_data()

        if not logs:
            return {}

        processed_logs = []

        # Extract agent name if available
        agent_name = "Unknown Agent"
        for log_item in logs:
            if log_item.get("type") == "TaskLog":
                agent_name_raw = log_item.get("agent")
                if isinstance(agent_name_raw, str) and agent_name_raw.strip():
                    agent_name = agent_name_raw.strip()
                break

        # For the first step, initialize with default values
        previous_action = "None"
        previous_outcome = "None"
        previous_thought = "None"

        for i, log_entry in enumerate(logs):
            entry_type = log_entry.get("type")
            # Skip entries we don't care about
            if entry_type not in ["AgentAction", "AgentFinish"]:
                continue

            reward = 0.0
            current_action = "UnknownAction"
            current_thought = "N/A"
            current_outcome = None

            # Extract thought or description based on entry type
            if entry_type == "AgentAction" or entry_type == "AgentFinish":
                current_thought = log_entry.get("thought", "Missing thought")
                if (
                    not current_thought
                    or self.is_string_effectively_empty_or_placeholder(current_thought)
                ):
                    current_thought = "Empty thought"
                    reward += self.PENALTY_POOR_THOUGHT
                elif (
                    len(str(current_thought).strip()) < self.MIN_MEANINGFUL_THOUGHT_LEN
                ):
                    reward += self.PENALTY_POOR_THOUGHT
                else:
                    reward += self.BONUS_GOOD_THOUGHT
            elif entry_type == "TaskLog":
                current_thought = log_entry.get("description", "")
                if not current_thought:
                    current_thought = log_entry.get("summary", "Task Log Content")
                if self.is_string_effectively_empty_or_placeholder(current_thought):
                    reward += self.PENALTY_TASKLOG_NO_DESCRIPTION
                else:
                    reward += self.REWARD_TASKLOG_HAS_DESCRIPTION

            # --- AgentAction Processing ---
            if entry_type == "AgentAction":
                tool_name = log_entry.get("tool")

                if isinstance(tool_name, str) and tool_name.strip():
                    current_action = tool_name.strip()
                    tool_result = log_entry.get("result")

                    if tool_result is not None:
                        current_outcome = str(tool_result)
                        result_lower = current_outcome.lower().strip()

                        is_error = any(
                            err_keyword in result_lower
                            for err_keyword in self.ERROR_KEYWORDS
                        )

                        if is_error:
                            reward += self.PENALTY_TOOL_ERROR
                        elif (
                            not result_lower
                            or result_lower == "[]"
                            or result_lower == "{}"
                        ):
                            reward += self.REWARD_TOOL_SUCCESS_EMPTY_RESULT
                        else:
                            reward += self.REWARD_TOOL_SUCCESS
                    else:
                        current_outcome = "NoResultField"
                        reward += self.PENALTY_TOOL_NO_RESULT_FIELD

                elif tool_name == "":
                    current_action = self.ACTION_INVALID_TOOL_EMPTY_NAME
                    current_outcome = "InvalidToolName(EmptyString)"
                    reward += self.PENALTY_TOOL_NAME_EMPTY_STRING

                else:  # No tool specified (tool is None) - "Thinking" action
                    current_action = self.ACTION_AGENT_THOUGHT_PROCESS
                    current_outcome = current_thought
                    if reward > 0:  # Implies BONUS_GOOD_THOUGHT was added
                        reward += self.REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT
                    else:
                        reward += self.PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT

            # --- AgentFinish Processing ---
            elif entry_type == "AgentFinish":
                current_action = self.ACTION_FINAL_ANSWER

                final_output = log_entry.get("output")
                current_outcome = (
                    str(final_output).strip() if final_output is not None else ""
                )

                if self.is_string_effectively_empty_or_placeholder(current_outcome):
                    reward += self.PENALTY_FINAL_OUTPUT_EMPTY_OR_PLACEHOLDER
                    current_outcome = "EmptyFinalOutput"
                elif len(current_outcome) <= self.SHORT_OUTPUT_LEN_THRESHOLD:
                    reward += self.REWARD_FINAL_OUTPUT_SHORT
                elif len(current_outcome) <= self.MEDIUM_OUTPUT_LEN_THRESHOLD:
                    reward += self.REWARD_FINAL_OUTPUT_MEDIUM
                else:
                    reward += self.REWARD_FINAL_OUTPUT_LONG

            # --- TaskLog Processing ---
            elif entry_type == "TaskLog":
                raw_output = log_entry.get("raw", "")
                current_outcome = str(raw_output).strip()

                if not current_outcome:
                    current_action = self.TASKLOG_NO_RAW_OUTPUT_REPR
                    reward += self.PENALTY_TASKLOG_NO_RAW
                else:
                    current_action = "TaskLogRawOutput"
                    reward += self.REWARD_TASKLOG_HAS_RAW
                    if any(
                        err_keyword in current_outcome.lower()
                        for err_keyword in self.ERROR_KEYWORDS
                    ):
                        reward += self.PENALTY_TASKLOG_RAW_CONTAINS_ERROR

            # Create the log entry with correct chronological ordering
            log_item = {
                "key": {
                    "state": {
                        "current_sub_task_or_thought": str(current_thought).strip(),
                        "last_action_taken": previous_action,
                        "last_outcome": previous_outcome,
                        "agent_context": agent_name,
                    },
                    "agent_action": current_action,
                },
                "reward_exec": self.normalize_reward(reward),
            }

            processed_logs.append(log_item)

            # Update previous values for next iteration
            previous_action = current_action
            previous_outcome = (
                current_outcome if current_outcome is not None else "NoOutcome"
            )
            previous_thought = current_thought

        if self.output_path:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(processed_logs, f, indent=2, ensure_ascii=False)

        return {"processed_logs": processed_logs}

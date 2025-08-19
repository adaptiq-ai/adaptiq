from typing import Any, Dict, List
from adaptiq.core.abstract.integrations.base_log_parser import BaseLogParser


class CrewLogParser(BaseLogParser):
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
    REWARD_TOOL_SUCCESS_EMPTY_RESULT = 0.25  # Tool worked, but result was empty
    PENALTY_TOOL_ERROR = -1.0
    PENALTY_TOOL_NO_RESULT_FIELD = -0.75  # Tool was called, but 'result' key is missing
    PENALTY_TOOL_NAME_EMPTY_STRING = -1.0  # If 'tool' field is an empty string

    # AgentAction: Thinking (No Tool)
    REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT = 0.3
    PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT = -0.3

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
    PENALTY_TASKLOG_RAW_CONTAINS_ERROR = -0.75

    # Keywords and Placeholder strings
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

    def get_supported_entry_types(self) -> List[str]:
        """
        Get the list of log entry types supported by this parser.

        Returns:
            List[str]: List of supported entry types.
        """
        return ["AgentAction", "AgentFinish", "TaskLog"]

    def extract_agent_name(self, logs: List[Dict[str, Any]]) -> str:
        """
        Extract agent name from CrewAI logs.

        Args:
            logs (List[Dict[str, Any]]): List of log entries.

        Returns:
            str: The extracted agent name or default value.
        """
        for log_item in logs:
            if log_item.get("type") == "TaskLog":
                agent_name_raw = log_item.get("agent")
                if isinstance(agent_name_raw, str) and agent_name_raw.strip():
                    return agent_name_raw.strip()
        return "Unknown Agent"

    def extract_thought_or_description(self, log_entry: Dict[str, Any], entry_type: str) -> str:
        """
        Extract thought or description from a log entry.

        Args:
            log_entry (Dict[str, Any]): The log entry to process.
            entry_type (str): The type of the log entry.

        Returns:
            str: The extracted thought or description.
        """
        if entry_type in ["AgentAction", "AgentFinish"]:
            thought = log_entry.get("thought", "Missing thought")
            if not thought or self.is_string_effectively_empty_or_placeholder(thought):
                return "Empty thought"
            return str(thought).strip()
        
        elif entry_type == "TaskLog":
            description = log_entry.get("description", "")
            if not description:
                description = log_entry.get("summary", "Task Log Content")
            if self.is_string_effectively_empty_or_placeholder(description):
                return "Empty description"
            return str(description).strip()
        
        return "N/A"

    def extract_action_and_outcome(self, log_entry: Dict[str, Any], entry_type: str) -> tuple[str, Any]:
        """
        Extract the action and outcome from a log entry.

        Args:
            log_entry (Dict[str, Any]): The log entry to process.
            entry_type (str): The type of the log entry.

        Returns:
            tuple[str, Any]: A tuple containing (action, outcome).
        """
        if entry_type == "AgentAction":
            return self._process_agent_action(log_entry)
        
        elif entry_type == "AgentFinish":
            return self._process_agent_finish(log_entry)
        
        elif entry_type == "TaskLog":
            return self._process_task_log(log_entry)
        
        return "UnknownAction", None

    def _process_agent_action(self, log_entry: Dict[str, Any]) -> tuple[str, Any]:
        """Process AgentAction entry and return action and outcome."""
        tool_name = log_entry.get("tool")
        
        if isinstance(tool_name, str) and tool_name.strip():
            action = tool_name.strip()
            tool_result = log_entry.get("result")
            
            if tool_result is not None:
                outcome = str(tool_result)
            else:
                outcome = "NoResultField"
            
            return action, outcome
        
        elif tool_name == "":
            return self.ACTION_INVALID_TOOL_EMPTY_NAME, "InvalidToolName(EmptyString)"
        
        else:  # No tool specified (thinking action)
            thought = self.extract_thought_or_description(log_entry, "AgentAction")
            return self.ACTION_AGENT_THOUGHT_PROCESS, thought

    def _process_agent_finish(self, log_entry: Dict[str, Any]) -> tuple[str, Any]:
        """Process AgentFinish entry and return action and outcome."""
        final_output = log_entry.get("output")
        outcome = str(final_output).strip() if final_output is not None else ""
        
        if self.is_string_effectively_empty_or_placeholder(outcome):
            outcome = "EmptyFinalOutput"
        
        return self.ACTION_FINAL_ANSWER, outcome

    def _process_task_log(self, log_entry: Dict[str, Any]) -> tuple[str, Any]:
        """Process TaskLog entry and return action and outcome."""
        raw_output = log_entry.get("raw", "")
        outcome = str(raw_output).strip()
        
        if not outcome:
            return self.TASKLOG_NO_RAW_OUTPUT_REPR, ""
        else:
            return "TaskLogRawOutput", outcome

    def calculate_reward(self, log_entry: Dict[str, Any], entry_type: str) -> float:
        """
        Calculate the reward for a given log entry based on its type and content.

        Args:
            log_entry (Dict[str, Any]): The log entry to process.
            entry_type (str): The type of the log entry.

        Returns:
            float: The calculated reward value.
        """
        reward = 0.0
        
        # Base reward for thought quality
        reward += self._calculate_thought_reward(log_entry, entry_type)
        
        # Specific rewards based on entry type
        if entry_type == "AgentAction":
            reward += self._calculate_agent_action_reward(log_entry)
        
        elif entry_type == "AgentFinish":
            reward += self._calculate_agent_finish_reward(log_entry)
        
        elif entry_type == "TaskLog":
            reward += self._calculate_task_log_reward(log_entry)
        
        return reward

    def _calculate_thought_reward(self, log_entry: Dict[str, Any], entry_type: str) -> float:
        """Calculate reward based on thought/description quality."""
        thought = self.extract_thought_or_description(log_entry, entry_type)
        
        if self.is_string_effectively_empty_or_placeholder(thought) or thought == "Empty thought":
            return self.PENALTY_POOR_THOUGHT
        elif len(thought) < self.MIN_MEANINGFUL_THOUGHT_LEN:
            return self.PENALTY_POOR_THOUGHT
        else:
            return self.BONUS_GOOD_THOUGHT

    def _calculate_agent_action_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward specific to AgentAction entries."""
        reward = 0.0
        tool_name = log_entry.get("tool")
        
        if isinstance(tool_name, str) and tool_name.strip():
            # Tool usage reward
            tool_result = log_entry.get("result")
            
            if tool_result is not None:
                result_str = str(tool_result).lower().strip()
                
                # Check for errors
                is_error = any(err_keyword in result_str for err_keyword in self.ERROR_KEYWORDS)
                
                if is_error:
                    reward += self.PENALTY_TOOL_ERROR
                elif not result_str or result_str == "[]" or result_str == "{}":
                    reward += self.REWARD_TOOL_SUCCESS_EMPTY_RESULT
                else:
                    reward += self.REWARD_TOOL_SUCCESS
            else:
                reward += self.PENALTY_TOOL_NO_RESULT_FIELD
        
        elif tool_name == "":
            reward += self.PENALTY_TOOL_NAME_EMPTY_STRING
        
        else:  # Thinking action (no tool)
            thought_reward = self._calculate_thought_reward(log_entry, "AgentAction")
            if thought_reward > 0:  # Good thought
                reward += self.REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT
            else:
                reward += self.PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT
        
        return reward

    def _calculate_agent_finish_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward specific to AgentFinish entries."""
        final_output = log_entry.get("output")
        output_str = str(final_output).strip() if final_output is not None else ""
        
        if self.is_string_effectively_empty_or_placeholder(output_str):
            return self.PENALTY_FINAL_OUTPUT_EMPTY_OR_PLACEHOLDER
        elif len(output_str) <= self.SHORT_OUTPUT_LEN_THRESHOLD:
            return self.REWARD_FINAL_OUTPUT_SHORT
        elif len(output_str) <= self.MEDIUM_OUTPUT_LEN_THRESHOLD:
            return self.REWARD_FINAL_OUTPUT_MEDIUM
        else:
            return self.REWARD_FINAL_OUTPUT_LONG

    def _calculate_task_log_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward specific to TaskLog entries."""
        reward = 0.0
        
        # Description reward
        description = log_entry.get("description", "")
        if not description:
            description = log_entry.get("summary", "")
        
        if self.is_string_effectively_empty_or_placeholder(description):
            reward += self.PENALTY_TASKLOG_NO_DESCRIPTION
        else:
            reward += self.REWARD_TASKLOG_HAS_DESCRIPTION
        
        # Raw output reward
        raw_output = log_entry.get("raw", "")
        raw_str = str(raw_output).strip()
        
        if not raw_str:
            reward += self.PENALTY_TASKLOG_NO_RAW
        else:
            reward += self.REWARD_TASKLOG_HAS_RAW
            # Check for errors in raw output
            if any(err_keyword in raw_str.lower() for err_keyword in self.ERROR_KEYWORDS):
                reward += self.PENALTY_TASKLOG_RAW_CONTAINS_ERROR
        
        return reward
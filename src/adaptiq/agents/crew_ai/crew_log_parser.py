from typing import Any, Dict, List

from adaptiq.core.abstract.integrations.base_log_parser import BaseLogParser
from adaptiq.core.entities import CrewRewards
from adaptiq.core.entities import LogItem, ValidationResults


class CrewLogParser(BaseLogParser):
    """
    AdaptiqLogParser transforms raw CrewAI logs into a state-action-reward mapping
    suitable for training or evaluation purposes.

    This class processes different log entry types (AgentAction, AgentFinish, TaskLog) and calculates
    a normalized reward signal based on rule-based heuristics, such as the quality of thoughts,
    tool usage success, output length, and error detection.
    """

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

    def extract_thought_or_description(
        self, log_entry: Dict[str, Any], entry_type: str
    ) -> str:
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

    def extract_action_and_outcome(
        self, log_entry: Dict[str, Any], entry_type: str
    ) -> tuple[str, Any]:
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
            return CrewRewards.ACTION_INVALID_TOOL_EMPTY_NAME.value, "InvalidToolName(EmptyString)"

        else:  # No tool specified (thinking action)
            thought = self.extract_thought_or_description(log_entry, "AgentAction")
            return CrewRewards.ACTION_AGENT_THOUGHT_PROCESS.value, thought

    def _process_agent_finish(self, log_entry: Dict[str, Any]) -> tuple[str, Any]:
        """Process AgentFinish entry and return action and outcome."""
        final_output = log_entry.get("output")
        outcome = str(final_output).strip() if final_output is not None else ""

        if self.is_string_effectively_empty_or_placeholder(outcome):
            outcome = "EmptyFinalOutput"

        return CrewRewards.ACTION_FINAL_ANSWER.value, outcome

    def _process_task_log(self, log_entry: Dict[str, Any]) -> tuple[str, Any]:
        """Process TaskLog entry and return action and outcome."""
        raw_output = log_entry.get("raw", "")
        outcome = str(raw_output).strip()

        if not outcome:
            return CrewRewards.TASKLOG_NO_RAW_OUTPUT_REPR.value, ""
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

    def _calculate_thought_reward(
        self, log_entry: Dict[str, Any], entry_type: str
    ) -> float:
        """Calculate reward based on thought/description quality."""
        thought = self.extract_thought_or_description(log_entry, entry_type)

        if (
            self.is_string_effectively_empty_or_placeholder(thought)
            or thought == "Empty thought"
        ):
            return CrewRewards.PENALTY_POOR_THOUGHT.value
        elif len(thought) < CrewRewards.MIN_MEANINGFUL_THOUGHT_LEN.value:
            return CrewRewards.PENALTY_POOR_THOUGHT.value
        else:
            return CrewRewards.BONUS_GOOD_THOUGHT.value

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
                is_error = any(
                    err_keyword in result_str for err_keyword in CrewRewards.ERROR_KEYWORDS.value
                )

                if is_error:
                    reward += CrewRewards.PENALTY_TOOL_ERROR.value
                elif not result_str or result_str == "[]" or result_str == "{}":
                    reward += CrewRewards.REWARD_TOOL_SUCCESS_EMPTY_RESULT.value
                else:
                    reward += CrewRewards.REWARD_TOOL_SUCCESS.value
            else:
                reward += CrewRewards.PENALTY_TOOL_NO_RESULT_FIELD.value

        elif tool_name == "":
            reward += CrewRewards.PENALTY_TOOL_NAME_EMPTY_STRING.value

        else:  # Thinking action (no tool)
            thought_reward = self._calculate_thought_reward(log_entry, "AgentAction")
            if thought_reward > 0:  # Good thought
                reward += CrewRewards.REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT.value
            else:
                reward += CrewRewards.PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT.value

        return reward

    def _calculate_agent_finish_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward specific to AgentFinish entries."""
        final_output = log_entry.get("output")
        output_str = str(final_output).strip() if final_output is not None else ""

        if self.is_string_effectively_empty_or_placeholder(output_str):
            return CrewRewards.PENALTY_FINAL_OUTPUT_EMPTY_OR_PLACEHOLDER.value
        elif len(output_str) <= CrewRewards.SHORT_OUTPUT_LEN_THRESHOLD.value:
            return CrewRewards.REWARD_FINAL_OUTPUT_SHORT.value
        elif len(output_str) <= CrewRewards.MEDIUM_OUTPUT_LEN_THRESHOLD.value:
            return CrewRewards.REWARD_FINAL_OUTPUT_MEDIUM.value
        else:
            return CrewRewards.REWARD_FINAL_OUTPUT_LONG.value

    def _calculate_task_log_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward specific to TaskLog entries."""
        reward = 0.0

        # Description reward
        description = log_entry.get("description", "")
        if not description:
            description = log_entry.get("summary", "")

        if self.is_string_effectively_empty_or_placeholder(description):
            reward += CrewRewards.PENALTY_TASKLOG_NO_DESCRIPTION.value
        else:
            reward += CrewRewards.REWARD_TASKLOG_HAS_DESCRIPTION.value

        # Raw output reward
        raw_output = log_entry.get("raw", "")
        raw_str = str(raw_output).strip()

        if not raw_str:
            reward += CrewRewards.PENALTY_TASKLOG_NO_RAW.value
        else:
            reward += CrewRewards.REWARD_TASKLOG_HAS_RAW.value
            # Check for errors in raw output
            if any(
                err_keyword in raw_str.lower() for err_keyword in CrewRewards.ERROR_KEYWORDS.value
            ):
                reward += CrewRewards.PENALTY_TASKLOG_RAW_CONTAINS_ERROR.value

        return reward

    def validate_parsing(self, raw_logs: Dict[str, Any], parsed_logs: List[LogItem]) -> ValidationResults:
        """
            Validate the parsing of logs by comparing raw and parsed logs.
        """
        pass
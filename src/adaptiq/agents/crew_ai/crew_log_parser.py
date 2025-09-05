from typing import Any, Dict, List

from adaptiq.core.abstract.integrations.base_log_parser import BaseLogParser
from adaptiq.core.entities import CrewRewards
from adaptiq.core.entities import LogItem, ValidationResults, RewardAssessment, ValidationSummary, ValidatedEntry


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
        return ["AgentAction", "AgentFinish"]

    def is_string_effectively_empty_or_placeholder(self, s: Any) -> bool:
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
        return s_str.lower() in CrewRewards.PLACEHOLDER_STRINGS_LOWER.value

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
        
    def calculate_step_time(self, current_entry: Dict[str, Any], previous_entry: Dict[str, Any] = None) -> float:
        """
        Calculate the time taken for a CrewAI step based on timestamps.
        
        Args:
            current_entry (Dict[str, Any]): The current log entry.
            previous_entry (Dict[str, Any], optional): The previous log entry for time comparison.

        Returns:
            float: Time taken in seconds. Returns 0.0 for first step or if calculation fails.
        """
        if not previous_entry:
            return 0.0
        
        try:
            from datetime import datetime
            
            current_timestamp = current_entry.get("timestamp")
            previous_timestamp = previous_entry.get("timestamp")
            
            if not current_timestamp or not previous_timestamp:
                return 0.0
            
            # Parse CrewAI timestamp format: "2025-07-19 13:13:15"
            current_time = datetime.strptime(current_timestamp, "%Y-%m-%d %H:%M:%S")
            previous_time = datetime.strptime(previous_timestamp, "%Y-%m-%d %H:%M:%S")
            
            time_diff = (current_time - previous_time).total_seconds()
            return max(0.0, time_diff)  # Ensure non-negative
            
        except Exception:
            return 0.0

    def extract_step_content(self, log_entry: Dict[str, Any]) -> str:
        """
        Extract all text content from a log entry for token counting.
        
        Args:
            log_entry (Dict[str, Any]): The log entry to extract content from.
            
        Returns:
            str: Combined text content from the entry.
        """
        content_parts = []
        
        # Add all text fields that contribute to the step's content
        fields_to_include = ["thought", "text", "tool_input", "result", "output"]
        
        for field in fields_to_include:
            value = log_entry.get(field)
            if value and not self.is_string_effectively_empty_or_placeholder(value):
                content_parts.append(str(value))
        
        return " ".join(content_parts)

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

        # Time-based rewards
        reward += self._calculate_time_reward(log_entry)
        
        # Token-based rewards
        reward += self._calculate_token_reward(log_entry)

        return reward

    def _calculate_time_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward based on step execution time."""
        step_time = self.calculate_step_time(log_entry, self._previous_entry)
        
        if step_time == 0.0:  # First step or calculation failed
            return 0.0
        elif step_time <= CrewRewards.FAST_STEP_TIME_THRESHOLD.value:
            return CrewRewards.REWARD_FAST_EXECUTION.value
        elif step_time <= CrewRewards.SLOW_STEP_TIME_THRESHOLD.value:
            return 0.0  # Neutral - reasonable time
        elif step_time <= CrewRewards.MAX_REASONABLE_STEP_TIME.value:
            return CrewRewards.PENALTY_SLOW_EXECUTION.value
        else:
            return CrewRewards.PENALTY_EXCESSIVE_TIME.value

    def _calculate_token_reward(self, log_entry: Dict[str, Any]) -> float:
        """Calculate reward based on token efficiency."""
        content = self.extract_step_content(log_entry)
        token_count = self.calculate_token_count(content)
        
        if token_count <= CrewRewards.EFFICIENT_TOKEN_THRESHOLD.value:
            return CrewRewards.REWARD_EFFICIENT_TOKENS.value
        elif token_count <= CrewRewards.VERBOSE_TOKEN_THRESHOLD.value:
            return 0.0  # Neutral - reasonable length
        elif token_count <= CrewRewards.EXCESSIVE_TOKEN_THRESHOLD.value:
            return CrewRewards.PENALTY_VERBOSE_TOKENS.value
        else:
            return CrewRewards.PENALTY_EXCESSIVE_TOKENS.value
    
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

    def validate_parsing(self, raw_logs: List[Dict[str, Any]], parsed_logs: List[LogItem]) -> ValidationResults:
        """
        Validate the parsing of logs by comparing raw and parsed logs using semantic similarity.
        Only validates entries with reward_exec in range (-0.25, 0.25).
        """
        validated_entries = []
        min_length = min(len(raw_logs), len(parsed_logs))
        
        for i in range(min_length):
            raw_entry = raw_logs[i]
            parsed_entry = parsed_logs[i]
            
            # Only validate if reward is in the target range
            if -0.25 < parsed_entry.reward_exec < 0.25:
                # Extract content from raw entry
                raw_content_parts = []
                if raw_entry.get("type"):
                    raw_content_parts.append(raw_entry["type"])
                if raw_entry.get("thought"):
                    raw_content_parts.append(raw_entry["thought"])
                if raw_entry.get("tool"):
                    raw_content_parts.append(f"Tool: {raw_entry['tool']}")
                if raw_entry.get("tool_input"):
                    raw_content_parts.append(f"Input: {raw_entry['tool_input']}")
                if raw_entry.get("result"):
                    raw_content_parts.append(f"Result: {raw_entry['result']}")
                
                raw_content = " ".join(raw_content_parts)
                
                # Extract content from parsed entry (focus on action and thought, not outcomes)
                parsed_content_parts = [
                    parsed_entry.key.state.current_sub_task_or_thought,
                    parsed_entry.key.agent_action
                ]
                parsed_content = " ".join(str(part) for part in parsed_content_parts if part)
                
                # Generate embeddings
                raw_embedding = self.embeddings.embed_query(raw_content)
                parsed_embedding = self.embeddings.embed_query(parsed_content)
                
                # Calculate cosine similarity
                import numpy as np
                similarity = np.dot(raw_embedding, parsed_embedding) / (
                    np.linalg.norm(raw_embedding) * np.linalg.norm(parsed_embedding)
                )
                
                # Determine adjustment based on similarity - BOOST rewards for good parsing
                original_reward = parsed_entry.reward_exec
                is_appropriate = True
                adjusted_reward = original_reward
                reason = "High semantic similarity - reward boosted significantly"
                
                if similarity > 0.8:
                    # High confidence - BOOST the reward significantly since parsing was accurate
                    adjusted_reward = 0.7  # Boost to high positive reward
                    is_appropriate = False  # Mark as adjusted
                    reason = f"High semantic similarity ({similarity:.3f}) - reward boosted to 0.7"
                elif similarity > 0.6:
                    # Medium confidence - moderate boost
                    adjusted_reward = 0.4  # Moderate positive reward
                    is_appropriate = False
                    reason = f"Medium semantic similarity ({similarity:.3f}) - reward boosted to 0.4"
                else:
                    # Low confidence - keep original low reward or slight penalty
                    adjusted_reward = original_reward * 0.8  # Small penalty for poor parsing
                    is_appropriate = False
                    reason = f"Low semantic similarity ({similarity:.3f}) - small penalty applied"
                
                # Ensure adjusted reward stays within reasonable bounds (but allow higher rewards)
                adjusted_reward = max(-0.25, min(1.0, adjusted_reward))
                
                # Update the parsed entry with adjusted reward
                corrected_entry = parsed_entry.model_copy()
                corrected_entry.reward_exec = adjusted_reward
                
            else:
                # Skip validation for rewards outside range
                is_appropriate = True
                adjusted_reward = parsed_entry.reward_exec
                reason = "Reward outside validation range (-0.25, 0.25) - skipped"
                corrected_entry = parsed_entry
            
            # Create validated entry
            validated_entry = ValidatedEntry(
                reward_assessment=RewardAssessment(
                    original=parsed_entry.reward_exec,
                    is_appropriate=is_appropriate,
                    adjusted=adjusted_reward,
                    reason=reason
                ),
                corrected_entry=corrected_entry
            )
            validated_entries.append(validated_entry)
        
        # Calculate summary statistics
        total_entries = len(validated_entries)
        appropriate_count = sum(1 for v in validated_entries if v.reward_assessment.is_appropriate)
        adjustment_count = total_entries - appropriate_count
        
        # Calculate average adjustment magnitude
        adjustments = [
            abs(v.reward_assessment.adjusted - v.reward_assessment.original)
            for v in validated_entries 
            if not v.reward_assessment.is_appropriate
        ]
        avg_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0.0
        
        summary = ValidationSummary(
            total_entries=total_entries,
            entries_with_appropriate_rewards=appropriate_count,
            entries_with_reward_adjustments=adjustment_count,
            appropriate_reward_rate=appropriate_count / total_entries if total_entries > 0 else 0.0,
            reward_adjustment_rate=adjustment_count / total_entries if total_entries > 0 else 0.0,
            average_adjustment_magnitude=avg_adjustment
        )
        
        return ValidationResults(
            summary=summary,
            validated_entries=validated_entries
        )
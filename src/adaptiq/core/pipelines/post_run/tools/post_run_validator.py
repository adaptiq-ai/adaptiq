import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from adaptiq.core.entities import ProcessedLogs
from adaptiq.core.entities.adaptiq_parsers import LogItem, RewardAssessment, ValidatedEntry, ValidationResults, ValidationSummary
logger = logging.getLogger(__name__)


class PostRunValidator:
    """
    A class to validate and refine parsed agent logs using an LLM.

    This validator checks parsed logs for accuracy, focuses on validating reward values
    and adjusts them if necessary based on the context of agent actions and outcomes.
    """

    def __init__(
        self,
        raw_logs: List[Dict[str, Any]],
        parsed_logs: ProcessedLogs,
        llm:BaseChatModel
    ):
        """
        Initialize the validator with raw and parsed logs.

        Args:
            raw_logs: List of raw log entries as dictionaries
            parsed_logs: ProcessedLogs containing parsed log entries
            llm: BaseChatModel instance for LLM interactions
        """
        self.raw_logs = raw_logs
        self.parsed_logs = parsed_logs
        self.llm = llm

    def _create_validation_prompt(
        self, raw_log_entry: Dict[str, Any], parsed_log_entry: LogItem
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM to validate a specific log entry, focusing on reward values.

        Args:
            raw_log_entry: Original log entry
            parsed_log_entry: Parsed log entry with state and reward

        Returns:
            A list of messages for the LLM to validate the parsing
        """
        system_message = """You are an expert AI agent log validator and Q-learning specialist. Your primary task is to verify that the 
            reward value assigned in the parsed log entry is appropriate given the raw log entry.
            
            Rewards should be between -1.0 and 1.0, with:
            - Positive rewards (0.0 to 1.0) for successful actions, helpful thoughts, and good outcomes
            - Negative rewards (-1.0 to 0.0) for failed actions, irrelevant thoughts, or poor outcomes
            
            IMPORTANT REWARD GUIDELINES FOR Q-LEARNING:
            - For successful actions that directly advance the task, assign rewards in the 0.5 to 0.9 range
            - For partially successful actions or minor progress, assign rewards in the 0.3 to 0.5 range
            - For neutral or minimal progress actions, assign rewards in the 0.1 to 0.3 range
            - For minor errors or slightly unhelpful actions, assign rewards in the -0.3 to 0 range
            - For significant errors or counterproductive actions, assign rewards in the -0.7 to -0.3 range
            - For critical failures that severely impact task completion, assign rewards in the -1.0 to -0.7 range
            
            Additional considerations:
            - Successful information retrieval that directly contributes to the task should get at least 0.5 reward
            - Small positive rewards (< 0.3) are too weak for effective Q-learning when an action is successful
            - Actions that result in errors or failures should have appropriately negative rewards
            - The final action that completes the primary task should receive a high reward (0.8-1.0)
            
            Please analyze the raw log entry and its corresponding parsed version, then provide your assessment 
            in the following JSON format:
            
            ```json
            {
                "reward_assessment": {
                    "original": 0.0,
                    "is_appropriate": true/false,
                    "adjusted": 0.0,
                    "reason": "reason for adjustment or confirmation"
                },
                "corrected_entry": {} // The corrected parsed log entry with adjusted reward if needed
            }
            ```
            
            If the reward is appropriate, set "is_appropriate" to true and keep the original reward value as the "adjusted" value.
            But be strict about following the reward guidelines - small positive rewards for successful actions are usually inappropriate in Q-learning.
            """

        human_message = f"""
            Raw Log Entry:
            ```json
            {json.dumps(raw_log_entry, indent=2)}
            ```
            
            Parsed Log Entry:
            ```json
            {json.dumps(parsed_log_entry.model_dump(), indent=2)}
            ```
            
            Please validate the reward value in this parsing and provide your assessment:
            """

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message},
        ]

    def validate_single_entry(
        self, raw_log_idx: int, parsed_log_idx: int
    ) -> ValidatedEntry:
        """
        Validate a single log entry pair, focusing on reward validation.

        Args:
            raw_log_idx: Index of the raw log entry
            parsed_log_idx: Index of the parsed log entry

        Returns:
            ValidatedEntry: Validation results with any reward corrections
        """
        if raw_log_idx >= len(self.raw_logs) or parsed_log_idx >= len(self.parsed_logs.processed_logs):
            raise IndexError("Log index out of range")

        raw_entry = self.raw_logs[raw_log_idx]
        parsed_entry: LogItem = self.parsed_logs.processed_logs[parsed_log_idx]

        # Ask LLM for validation
        messages = self._create_validation_prompt(raw_entry, parsed_entry)
        response = self.llm.invoke(messages)

        parsed_response = None

        try:
            parsed_response = json.loads(response.content)
        except (json.JSONDecodeError, AttributeError):
            import re
            json_match = re.search(
                r"```json\s*(.*?)\s*```",
                response.content if hasattr(response, "content") else str(response),
                re.DOTALL,
            )
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Fallback if parsing fails
        if not parsed_response:
            parsed_response = {
                "reward_assessment": {
                    "original": parsed_entry.reward_exec,
                    "is_appropriate": True,
                    "adjusted": parsed_entry.reward_exec,
                    "reason": "Failed to parse LLM response, keeping original reward",
                },
                "corrected_entry": parsed_entry.model_dump(),
            }

        # Ensure we return a ValidatedEntry model
        try:
            validated_entry = ValidatedEntry(
                reward_assessment=RewardAssessment(**parsed_response["reward_assessment"]),
                corrected_entry=LogItem.model_validate(
                    parsed_response["corrected_entry"]
                ),
            )
        except Exception:
            # As a last resort, construct from the parsed_entry directly
            validated_entry = ValidatedEntry(
                reward_assessment=RewardAssessment(
                    original=parsed_entry.reward_exec,
                    is_appropriate=True,
                    adjusted=parsed_entry.reward_exec,
                    reason="Validation fallback due to schema mismatch",
                ),
                corrected_entry=parsed_entry,
            )

        return validated_entry

    def validate_all_entries(self) -> List[ValidatedEntry]:
        """
        Validate all log entries and return validation results.
        Only entries with reward_exec in range (-0.25, 0.25) are validated.

        Returns:
            List of validation results for all entries
        """
        results: List[ValidatedEntry] = []
        min_length = min(len(self.raw_logs), len(self.parsed_logs.processed_logs))

        for i in range(min_length):
            parsed_entry = self.parsed_logs.processed_logs[i]
            reward = parsed_entry.reward_exec 
            # fallback to "reward" if "reward_exec" missing
            if -0.25 < reward < 0.25:
                validated_entry = self.validate_single_entry(i, i)
            else:
                # Skip validation, return as appropriate without changes
                validated_entry = ValidatedEntry(
                    reward_assessment=RewardAssessment(
                        original=reward,
                        is_appropriate=True,
                        adjusted=reward,
                        reason="Skipped validation due to reward outside (-0.25, 0.25) range",
                    ),
                    corrected_entry=self.parsed_logs.processed_logs[i]
                )
            results.append(validated_entry)

        return results

    def get_corrected_logs(self) -> List[LogItem]:
        """
        Get the corrected version of all parsed logs after validation.

        Returns:
            List of corrected log entries
        """
        validations = self.validate_all_entries()
        corrected_logs = []

        for validation in validations:
            # Make sure to update the reward value in the corrected entry
            corrected_entry = validation.corrected_entry
            if validation.reward_assessment.is_appropriate is False:
                corrected_entry.reward_exec = validation.reward_assessment.adjusted

            corrected_logs.append(corrected_entry)

        return corrected_logs

    def summarize_validations(
        self, validations: List[ValidatedEntry]
    ) -> ValidationSummary:
        """
        Generate a summary of reward validation results.

        Args:
            validations: List of validation results

        Returns:
            ValidationSummary: Summary statistics about the reward validations
        """
        total = len(validations) if validations else 0
        appropriate_rewards = sum(
            1 for v in (validations or []) if v.reward_assessment.is_appropriate
        )
        reward_adjustments = total - appropriate_rewards

        # Calculate average magnitude of adjustments
        adjustments: List[float] = []
        for v in (validations or []):
            if not v.reward_assessment.is_appropriate:
                original = v.reward_assessment.original
                adjusted = v.reward_assessment.adjusted
                adjustments.append(abs(adjusted - original))

        avg_adjustment = sum(adjustments) / len(adjustments) if adjustments else 0

        return ValidationSummary(
            total_entries=total,
            entries_with_appropriate_rewards=appropriate_rewards,
            entries_with_reward_adjustments=reward_adjustments,
            appropriate_reward_rate=appropriate_rewards / total if total > 0 else 0,
            reward_adjustment_rate=reward_adjustments / total if total > 0 else 0,
            average_adjustment_magnitude=avg_adjustment,
        )

    def run_validation_pipeline(self) -> Tuple[List[LogItem], ValidationResults]:
        """
        Run the complete validation pipeline and return both the corrected logs and validation results.

        Returns:
            Tuple containing:
            - List of corrected logs with validated rewards
            - Dictionary with validation results and summary
        """
        validations = self.validate_all_entries()
        corrected_logs = self.get_corrected_logs()
        summary = self.summarize_validations(validations)

        # Create the validation results output
        validation_results = ValidationResults(validated_entries=validations, summary=summary)

        return corrected_logs, validation_results

    def run(self) -> List[Dict[str, Any]]:
        """
        Run the validation process and return the corrected logs.
        This is a simplified method for use when only the corrected logs are needed.

        Returns:
            List of corrected logs with validated rewards
        """
        corrected_logs = self.get_corrected_logs()
        return corrected_logs

    def extract_corrected_entries(
        self, validation_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract only the corrected entries from validation results.

        This method takes validation results (either provided or by running the validation pipeline)
        and extracts just the corrected log entries, removing all the assessment metadata.

        Args:
            validation_results: Dictionary containing validations and summary (if None, will run validation pipeline)

        Returns:
            List of corrected log entries without validation metadata
        """
        if validation_results is None:
            # Run the validation pipeline to get results
            _, validation_results = self.run_validation_pipeline()

        # Extract just the corrected entries from the validations
        corrected_entries = []

        if "validations" in validation_results:
            for validation in validation_results["validations"]:
                if "corrected_entry" in validation:
                    corrected_entries.append(validation["corrected_entry"])

        return corrected_entries
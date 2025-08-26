import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
from langchain_core.embeddings import Embeddings
from adaptiq.core.entities import LogItem, LogKey, LogState, ProcessedLogs
from adaptiq.core.entities.adaptiq_parsers import ValidationResults


class BaseLogParser(ABC):
    """
    Abstract base class for log parsers that transform raw logs into state-action-reward mappings
    suitable for training or evaluation purposes.

    This class defines the interface and common functionality that all log parsers should implement.
    Concrete implementations should process different log entry types and calculate normalized
    reward signals based on rule-based heuristics.
    """

    # --- Abstract Constants (to be defined by subclasses) ---

    @property
    @abstractmethod
    def MIN_MEANINGFUL_THOUGHT_LEN(self) -> int:
        """Minimum length for a meaningful thought/description."""
        pass

    @property
    @abstractmethod
    def ERROR_KEYWORDS(self) -> List[str]:
        """Keywords that indicate errors in log outputs."""
        pass

    @property
    @abstractmethod
    def PLACEHOLDER_STRINGS_LOWER(self) -> List[str]:
        """Lowercase strings considered as placeholders or empty content."""
        pass

    def __init__(self, logs_path: str, output_path: str = None):
        """
        Initialize the log parser with input and output paths.

        Args:
            embeddings (Embeddings): The embeddings model to use for text representation.
            logs_path (str): Path to the log file to be processed.
            output_path (str, optional): Path where processed logs will be saved.
        """
        self.logs_path = logs_path
        self.output_path = output_path
        self.parsed_file_name = "parsed_logs.json"
        self.embeddings = None

    def set_embeddings(self, embeddings: Embeddings):
        """
        Set the embeddings model for the log parser.

        Args:
            embeddings (Embeddings): The embeddings model to use.
        """
        self.embeddings = embeddings

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
        Normalize using hyperbolic tangent function to force floats between -1 and 1.

        Args:
            reward (float): The raw reward value.

        Returns:
            float: The normalized reward value between -1 and 1.
        """
        return round(math.tanh(reward), 4)

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
        return s_str.lower() in self.PLACEHOLDER_STRINGS_LOWER

    @abstractmethod
    def calculate_reward(self, log_entry: Dict[str, Any], entry_type: str) -> float:
        """
        Calculate the reward for a given log entry based on its type and content.

        Args:
            log_entry (Dict[str, Any]): The log entry to process.
            entry_type (str): The type of the log entry (e.g., 'AgentAction', 'AgentFinish').

        Returns:
            float: The calculated reward value.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_supported_entry_types(self) -> List[str]:
        """
        Get the list of log entry types supported by this parser.

        Returns:
            List[str]: List of supported entry types.
        """
        pass

    @abstractmethod
    def validate_parsing(self, raw_logs:Dict[str, Any], parsed_logs: List[LogItem])-> ValidationResults:
        """
            Validate the parsing of logs by comparing raw and parsed logs.
        """
        pass

    def extract_agent_name(self, logs: List[Dict[str, Any]]) -> str:
        """
        Extract agent name from logs. Can be overridden by subclasses for specific extraction logic.

        Args:
            logs (List[Dict[str, Any]]): List of log entries.

        Returns:
            str: The extracted agent name or a default value.
        """
        return "Unknown Agent"

    def create_log_item(
        self,
        current_thought: str,
        previous_action: str,
        previous_outcome: Any,
        agent_name: str,
        current_action: str,
        reward: float,
    ) -> LogItem:
        return LogItem(
            key=LogKey(
                state=LogState(
                    current_sub_task_or_thought=str(current_thought).strip(),
                    last_action_taken=previous_action,
                    last_outcome=previous_outcome,
                    agent_context=agent_name,
                ),
                agent_action=current_action,
            ),
            reward_exec=self.normalize_reward(reward),
        )

    def save_processed_logs(self, processed_logs: List[LogItem]) -> None:
        if self.output_path and self.parsed_file_name:
            try:
                full_path = os.path.join(self.output_path, self.parsed_file_name)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # Serialize using Pydantic `.model_dump()`
                with open(full_path, "w", encoding="utf-8") as f:
                    json.dump(
                        [log.model_dump() for log in processed_logs],
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

            except Exception as e:
                raise RuntimeError(f"Failed to save logs to {full_path}: {e}") from e

    def parse_logs(self) -> Tuple[ProcessedLogs, ValidationResults]:
        """
        Parse the logs from the specified log file.

        """
        logs: List[Dict[str, Any]] = self.load_json_data()

        if not logs:
            raise ValueError("No logs found to parse.")

        processed_logs: List[LogItem] = []
        agent_name = self.extract_agent_name(logs)
        supported_types = self.get_supported_entry_types()

        previous_action = "None"
        previous_outcome = "None"

        for log_entry in logs:
            entry_type = log_entry.get("type")
            if entry_type not in supported_types:
                continue

            current_thought = self.extract_thought_or_description(log_entry, entry_type)
            current_action, current_outcome = self.extract_action_and_outcome(
                log_entry, entry_type
            )
            reward = self.calculate_reward(log_entry, entry_type)

            log_item = self.create_log_item(
                current_thought=current_thought,
                previous_action=previous_action,
                previous_outcome=previous_outcome,
                agent_name=agent_name,
                current_action=current_action,
                reward=reward,
            )
            processed_logs.append(log_item)

            previous_action = current_action
            previous_outcome = (
                current_outcome if current_outcome is not None else "NoOutcome"
            )

        self.save_processed_logs(processed_logs)

        validation_results = self.validate_parsing(raw_logs=logs, parsed_logs=processed_logs)
        return ProcessedLogs(processed_logs=processed_logs), validation_results

import datetime
import json
import os
import re
from abc import ABC, abstractmethod


class BaseAgentLogger(ABC):
    """
    Abstract base class for CrewAI agent logging implementations.

    This base class defines the interface that all logger implementations must follow.
    It provides common initialization and utility methods while requiring subclasses
    to implement the core logging methods for agent thoughts and task summaries.

    """

    def __init__(self, log_file="log.txt", json_file="log.json"):
        """
        Initialize the base logger with file paths.

        Args:
            log_file (str, optional): Path to the human‑readable plaintext log
                file. Written in append mode. Defaults to "log.txt".
            json_file (str, optional): Path to the structured JSON log file that
                stores a list of log entry objects. Defaults to "log.json".
        """
        self.log_file = log_file
        self.json_file = json_file
        self._initialize_files()

    def _initialize_files(self):
        """
        Initialize log files if they don't exist.

        This method can be overridden by subclasses to customize file initialization.
        """
        # Initialize JSON log file with array if not existing
        if not os.path.exists(self.json_file):
            with open(self.json_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _get_timestamp(self):
        """
        Get the current timestamp formatted for logging.

        Returns:
            str: Formatted timestamp string
        """
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _append_to_json(self, log_data):
        """
        Append a structured log entry to the JSON log file.

        Reads the current JSON array from disk (recovering gracefully from
        JSON decode errors by starting a new list), appends `log_data`,
        and rewrites the file. Entries that contain only a timestamp (i.e.,
        no additional diagnostic fields) are ignored to avoid noise.

        Args:
            log_data (dict): Structured log payload to append. Must contain at
                least one key in addition to "timestamp" to be written.
        """
        # Skip writing logs with only timestamp (invalid)
        if len(log_data.keys()) <= 1:
            return

        with open(self.json_file, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

            data.append(log_data)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

    @abstractmethod
    def log_thoughts(self, formatted_answer):
        """
        Log an agent step, final result, or parsing error from a CrewAI run.

        This method must be implemented by subclasses to handle logging of:
        * AgentAction  – logs thought, action text, tool name, tool input, result.
        * AgentFinish  – logs final thought and output text.
        * OutputParserException – logs the parser error message.

        Args:
            formatted_answer (AgentAction | AgentFinish | OutputParserException | Any):
                Object produced during agent execution.
        """
        pass

    @abstractmethod
    def log_task(self, output):
        """
        Log a high-level task record summarizing an agent's work.

        This method must be implemented by subclasses to handle logging of
        high-level task checkpoints (e.g., after a CrewAI Task run).
        Should capture agent name, task description, raw details, and summary.

        Args:
            output: An object with the attributes `agent`, `description`, `raw`,
                and `summary` (such as a CrewAI Task output object).
        """
        pass

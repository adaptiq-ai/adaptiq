import ast
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from adaptiq.core.entities import (
    ClassificationEntry,
    ClassificationResponse,
    ProcessedLogs,
    LogItem
)




class BaseStateMapper(ABC):
    """
    Abstract base class for matching execution trace states with Q-table states.
    """

    def __init__(self, warmed_qtable_data: Dict[str, Any], llm: BaseChatModel):
        """
        Initialize the StateMapper.

        Args:
            warmed_qtable_data: Q-table data containing Q_table and seen_states
            llm: BaseChatModel
        """
        self.llm = llm
        # Store the Q-table data
        self.qtable = warmed_qtable_data.get("Q_table", {})

        # Combine states from Q-table and seen_states, ensuring uniqueness
        self.known_states = set(self.qtable.keys())
        for state in warmed_qtable_data.get("seen_states", []):
            self.known_states.add(state)

        # Convert to a list for easier processing
        self.known_states = list(self.known_states)

        # Parse states for better matching
        self.parsed_states = self._parse_known_states()

        # Create classification prompt template
        self.classification_prompt_template = self._create_classification_prompt()

    def _parse_known_states(self) -> List[Tuple[str, List]]:
        """
        Parse known states into a more comparable format.

        Returns:
            List of tuples containing (original_state_string, parsed_components)
        """
        parsed_states = []

        for state_str in self.known_states:
            try:
                # Handle tuple-like strings
                if state_str.startswith("(") and state_str.endswith(")"):
                    components = ast.literal_eval(state_str)
                    parsed_states.append((state_str, list(components)))
                # Handle list-like strings
                elif state_str.startswith("[") and state_str.endswith("]"):
                    components = ast.literal_eval(state_str)
                    parsed_states.append((state_str, components))
                else:
                    # For any other format, store as is
                    parsed_states.append((state_str, [state_str]))
            except (SyntaxError, ValueError):
                # If parsing fails, store original string
                parsed_states.append((state_str, [state_str]))

        return parsed_states

    @abstractmethod
    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """
        Create the prompt template for state classification.

        Returns:
            ChatPromptTemplate instance
        """
        pass

    @abstractmethod
    def _invoke_llm_for_classification(
        self, input_state: LogItem
    ) -> ClassificationResponse:
        """
        Invoke the LLM to classify a state.

        Args:
            input_state: State to classify

        Returns:
            ClassificationResponse: The LLM's classification output
        """
        pass

    def classify_states(
        self, processed_logs: ProcessedLogs
    ) -> List[ClassificationEntry]:
        """
        Classify input states against the known states.

        Args:
            input_states: List of states to classify

        Returns:
            List of classification results
        """
        classification_results: List[ClassificationEntry] = []

        for index, processed_log in enumerate(processed_logs):
            # Invoke the LLM for classification
            classification_output = self._invoke_llm_for_classification(processed_log)

            # Create the classification entry
            classification_entry = ClassificationEntry(
                index=index,
                input_state=classification_output.input_state,
                classification=classification_output.classification,
            )

            classification_results.append(classification_entry)

        return classification_results

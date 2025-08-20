from abc import ABC, abstractmethod
import ast
import json
from typing import Any, Dict, List, Tuple, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

class BaseStateMapper(ABC):
    """
    Abstract base class for matching execution trace states with Q-table states.
    """

    def __init__(
        self,
        warmed_qtable_data: Dict[str, Any],
        llm: BaseChatModel
    ):
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

    @abstractmethod
    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """
        Create the prompt template for state classification.
        
        Returns:
            ChatPromptTemplate instance
        """
        pass

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

    def _extract_state_from_input(self, input_data: Union[str, List, Dict]) -> Union[List, str]:
        """
        Extract the state from input data.

        Args:
            input_data: Input data (string, list, or dictionary)

        Returns:
            Extracted state (list or string)
        """
        if isinstance(input_data, dict) and "state" in input_data:
            return input_data["state"]
        return input_data

    @abstractmethod
    def _invoke_llm_for_classification(self, input_state: Union[str, List, Dict]) -> Dict:
        """
        Invoke the LLM to classify a state.

        Args:
            input_state: State to classify (can be string, list, or dict)

        Returns:
            Dict containing the LLM's classification output
        """
        pass

    def _validate_classification(self, classification_output: Dict) -> Dict:
        """
        Validate the classification output from the LLM.

        Args:
            classification_output: The LLM's classification output

        Returns:
            Validated classification output
        """
        classification = classification_output.get("classification", {})

        # If LLM says it's a known state, verify the matched state is actually in our known states
        if classification.get("is_known_state", False):
            matched_state = classification.get("matched_state")

            if matched_state not in self.known_states:
                # If matched state not in known states, invalidate the classification
                classification["is_known_state"] = False
                classification["matched_state"] = None
                classification["reasoning"] = (
                    "State validation: matched state not found in known states"
                )

        classification_output["classification"] = classification
        return classification_output

    def classify_states(self, input_states: List[Union[str, List, Dict]]) -> List[Dict]:
        """
        Classify input states against the known states.

        Args:
            input_states: List of states to classify

        Returns:
            List of classification results
        """
        classification_results = []

        for index, input_state in enumerate(input_states):
            # Invoke the LLM for classification
            classification_output = self._invoke_llm_for_classification(input_state)

            # Validate the classification output
            validated_output = self._validate_classification(classification_output)

            # Create the classification entry
            classification_entry = {
                "index": index,
                "input_state": input_state,
                "classification": validated_output.get("classification", {}),
            }

            classification_results.append(classification_entry)

        return classification_results

    def classify_single_state(self, input_state: Union[str, List, Dict]) -> Dict:
        """
        Classify a single input state against known states.

        Args:
            input_state: State to classify

        Returns:
            Classification result for the input state
        """
        results = self.classify_states([input_state])
        if results:
            return results[0]
        return {
            "index": 0,
            "input_state": input_state,
            "classification": {
                "is_known_state": False,
                "matched_state": None,
                "reasoning": "Classification failed",
            },
        }

    def save_classification_json(self, classification_results: List[Dict], output_path: str) -> None:
        """
        Save the classification results to a JSON file.

        Args:
            classification_results: Classification results to save
            output_path: Path to save the JSON file
        """
        with open(output_path, "w") as f:
            json.dump(classification_results, f, indent=2)

    @classmethod
    @abstractmethod
    def from_qtable_file(
        cls, qtable_file_path: str, llm_model_name: str, llm_api_key: str, provider: str
    ) -> "BaseStateMapper":
        """
        Create a StateMapper instance from a Q-table file.

        Args:
            qtable_file_path: Path to the Q-table JSON file
            llm_model_name: Model name to use for reconciliation
            llm_api_key: API key for the provider
            provider: Provider for the LLM

        Returns:
            StateMapper instance
        """
        pass
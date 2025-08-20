from abc import ABC, abstractmethod
from typing import Dict, Tuple
from langchain_core.prompts import  PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

class BaseStateActionExtractor(ABC):
    """
    Abstract base class for extracting state and action information from execution data
    and transforming it into a standardized format.
    """

    def __init__(self, llm: BaseChatModel):
        """
        Initialize the extractor.

        Args:
            provider (str): The LLM provider (e.g., "openai", "anthropic", etc.)
            model (str): The model name to use
            api_key (str, optional): API key for the provider
        """
        self.llm = llm
        self.prompt_template = self._create_prompt_template()


    @abstractmethod
    def _create_prompt_template(self) -> PromptTemplate:
        """
        Create the prompt template for state-action extraction.
        
        Returns:
            PromptTemplate instance
        """
        pass

    @abstractmethod
    def _extract_raw_state_and_action(self, input_data)-> Tuple[Dict, str]:
        """
        Extract raw state and action from the input data.

        Args:
            input_data: The input data containing state and action information

        Returns:
            tuple: (state_dict, action_str)
        """
        pass

    @abstractmethod
    def _transform_with_llm(self, state_dict, action_str):
        """
        Use LLM to transform the extracted state and action.

        Args:
            state_dict: The extracted state dictionary
            action_str: The extracted action string

        Returns:
            dict: Transformed state and action
        """
        pass

    def extract(self, input_data):
        """
        Extract and transform state and action from the input data.

        Args:
            input_data: The input data containing state and action information

        Returns:
            dict: Transformed state and action
        """
        state_dict, action_str = self._extract_raw_state_and_action(input_data)
        transformed_data = self._transform_with_llm(state_dict, action_str)
        return transformed_data

    def process_batch(self, input_data_list):
        """
        Process a batch of input data.

        Args:
            input_data_list (list): List of input data dictionaries or strings

        Returns:
            list: List of transformed state and action dictionaries
        """
        results = []
        for input_data in input_data_list:
            try:
                result = self.extract(input_data)
                results.append(result)
            except Exception as e:
                print(f"Error processing input: {str(e)}")
                results.append({"error": str(e)})
        return results
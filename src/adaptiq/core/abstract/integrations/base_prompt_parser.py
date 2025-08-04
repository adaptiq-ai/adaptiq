from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BasePromptParser(ABC):
    """
    Abstract base class for prompt parsers that analyze agent task descriptions
    and infer idealized sequences of steps.
    
    This class defines the interface that all prompt parser implementations
    must follow, ensuring consistency across different parsing strategies
    and LLM providers.
    """

    def __init__(self, config_path: str):
        """
        Initialize the prompt parser with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.adaptiq_config = self._load_config(config_path)
        self._validate_config()
        self._initialize_components()

    @abstractmethod
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and parse the configuration file.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dictionary containing the parsed configuration

        Raises:
            ValueError: If configuration cannot be loaded or is invalid
        """
        pass

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate that the loaded configuration contains all required fields.

        Raises:
            ValueError: If required configuration fields are missing or invalid
        """
        pass

    @abstractmethod
    def _initialize_components(self) -> None:
        """
        Initialize LLM client, tools, and other components based on configuration.
        
        This method should set up:
        - LLM client/model
        - Agent tools list
        - Prompt templates
        - Any other parser-specific components

        Raises:
            ValueError: If components cannot be initialized
        """
        pass

    @abstractmethod
    def _load_agent_task_description(self) -> str:
        """
        Load the agent's task description from the configured source.

        Returns:
            String containing the agent's task description

        Raises:
            ValueError: If task description cannot be loaded
        """
        pass

    @abstractmethod
    def _construct_parsing_prompt(self, task_description: str) -> Any:
        """
        Construct the prompt/input for the parsing model.

        Args:
            task_description: The agent's task description text

        Returns:
            Formatted prompt ready for the parsing model
        """
        pass

    @abstractmethod
    def _invoke_parsing_model(self, prompt: Any) -> str:
        """
        Invoke the parsing model with the constructed prompt.

        Args:
            prompt: The formatted prompt for the model

        Returns:
            Raw response from the parsing model

        Raises:
            Exception: If model invocation fails
        """
        pass

    @abstractmethod
    def _parse_model_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse and validate the model's response into structured steps.

        Args:
            response: Raw response from the parsing model

        Returns:
            List of dictionaries representing parsed steps

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        pass

    def parse_prompt(self) -> List[Dict[str, Any]]:
        """
        Main method to parse the agent's prompt and infer idealized steps.
        
        This template method orchestrates the parsing process by calling
        the abstract methods in the correct sequence.

        Returns:
            List of dictionaries, each containing step information with keys:
            - Intended_SubTask: Description of the subtask
            - Intended_Action: Primary action to be taken
            - Preconditions_Mentioned_In_Prompt: Required preconditions
            - Expected_Ideal_Outcome_Mentioned_In_Prompt: Expected outcome

        Raises:
            ValueError: If parsing fails at any step
        """
        # Load the task description
        task_description = self._load_agent_task_description()
        
        # Construct the parsing prompt
        prompt = self._construct_parsing_prompt(task_description)
        
        # Invoke the parsing model
        response = self._invoke_parsing_model(prompt)
        
        # Parse and validate the response
        parsed_steps = self._parse_model_response(response)
        
        return parsed_steps

    def _validate_parsed_steps(self, parsed_steps: List[Dict[str, Any]]) -> None:
        """
        Validate the structure of parsed steps.
        
        This method can be used by concrete implementations to ensure
        the parsed steps follow the expected format.

        Args:
            parsed_steps: List of step dictionaries to validate

        Raises:
            ValueError: If step structure is invalid
        """
        if not isinstance(parsed_steps, list):
            raise ValueError("Parsed steps should be a list")

        required_fields = [
            "Intended_SubTask",
            "Intended_Action", 
            "Preconditions_Mentioned_In_Prompt",
            "Expected_Ideal_Outcome_Mentioned_In_Prompt",
        ]

        for i, step in enumerate(parsed_steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} should be a dictionary")

            for field in required_fields:
                if field not in step:
                    raise ValueError(f"Step {i} is missing required field '{field}'")
                
                if not isinstance(step[field], str):
                    raise ValueError(f"Step {i} field '{field}' should be a string")

    @property
    @abstractmethod
    def supported_providers(self) -> List[str]:
        """
        Return list of supported LLM providers for this parser implementation.

        Returns:
            List of supported provider names
        """
        pass

    @property
    @abstractmethod
    def parser_name(self) -> str:
        """
        Return the name/identifier of this parser implementation.

        Returns:
            String identifier for this parser
        """
        pass

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the current configuration.

        Returns:
            Dictionary containing key configuration information
        """
        return {
            "parser_name": self.parser_name,
            "supported_providers": self.supported_providers,
            "config_path": self.config_path,
            "llm_model": getattr(self, 'llm_model_name', 'Not configured'),
            "provider": getattr(self, 'provider', 'Not configured'),
            "agent_tools_count": len(getattr(self, 'agent_tools', [])),
        }
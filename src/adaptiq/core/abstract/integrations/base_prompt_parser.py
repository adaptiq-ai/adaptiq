from abc import ABC, abstractmethod
import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from adaptiq.core.entities import AdaptiQConfig, AgentTool, TaskIntent


class BasePromptParser(ABC):
    """
    Abstract base class for prompt parsers that analyze agent task descriptions
    and infer idealized sequences of steps.
    
    This class defines the interface that all prompt parser implementations
    must follow, ensuring consistency across different parsing strategies
    and LLM providers.
    """

    def __init__(self, config_data: AdaptiQConfig, task:str, tools: List[AgentTool] = []):
        """
        Initialize the prompt parser with configuration.

        Args:
            config_path: Path to the configuration file
        """
        self.config_data = config_data
        self.llm_model_name = self.config_data.llm_config.model_name
        self.provider = self.config_data.llm_config.provider
        self.api_key = self.config_data.llm_config.api_key
        self.task = task
        self.agent_tools = tools
        self.required_fields = [
            "intended_subTask",
            "intended_action", 
            "preconditions_mentioned_in_prompt",
            "expected_ideal_outcome_mentioned_in_prompt",
        ]
        # Initialize prompt template
        self.prompt_parser_template = self._create_prompt_template()

        # TODO this step will be removed from the parser and will have a unified llm client init 
        self._initialize_components()

    def _initialize_components(self) -> None:
        """
        Initialize LLM client and prompt template based on configuration.

        Raises:
            ValueError: If components cannot be initialized
        """

        # Initialize LLM client
        if self.provider.value == "openai":
            self.prompt_parser_llm = ChatOpenAI(
                temperature=0.0, 
                model=self.llm_model_name, 
                api_key=self.api_key
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )   

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the LLM to parse the task description.

        Returns:
            ChatPromptTemplate for the parsing task
        """
        prompt_template = """You are an AI Task Decomposer. Your goal is to analyze an agent's task description and its available tools, then break down the task into an *intended sequence of logical steps*.
        Available Tools for the Agent: {agent_tools}
        Agent's Task Description:
        ---
        {task_description_text}
        ---
        For each step you identify in the agent's plan, provide:
        1.  'Intended_SubTask': A concise description of the agent's immediate goal for this step as described in the task.
        2.  'Intended_Action': The primary strategic action (from Available Tools or conceptual actions like 'Write_Email_Body', 'Formulate_Final_Answer') planned to achieve this sub-task.
        3.  'Preconditions_Mentioned_In_Prompt': Any conditions mentioned in the task description that must be met before this action.
        4.  'Expected_Ideal_Outcome_Mentioned_In_Prompt': What the task description suggests is the successful result of this action.
        Output ONLY a valid JSON array of these step objects. Example:
        [
        {{
            "intended_subTask": "example subtask",
            "intended_action": "example action or tool",
            "preconditions_mentioned_in_prompt": "example precondition",
            "expected_ideal_outcome_mentioned_in_prompt": "example expected outcome"
        }},
        {{
            "intended_subTask": "example subtask 2",
            "intended_action": "example action 2 or tool",
            "preconditions_mentioned_in_prompt": "example precondition 2",
            "expected_ideal_outcome_mentioned_in_prompt": "example expected outcome 2"
        }}
        ]

        Ensure your output is a perfectly formatted JSON array. Do not include any explanations or additional text outside the JSON array."""

        return ChatPromptTemplate.from_template(prompt_template)

    def _construct_parsing_prompt(self) -> Dict[str, str]:
        """
        Construct the complete prompt for the LLM to parse the task description.

        Args:
            task_description: The text of the agent's task description

        Returns:
            Dictionary with the parameters for the prompt template
            + agent_tools: str
            + task_description_text: str
        """

        tool_strings = [
            f"{tool.name}: {tool.description}"
            for tool in self.agent_tools
        ]
        return {
            "agent_tools": ", ".join(tool_strings),
            "task_description_text": self.task,
        }

    def _invoke_parsing_model(self, prompt_params: Dict[str, Any]) -> str:
        """
        Invoke the parsing model with the constructed prompt.

        Args:
            prompt_params: Dictionary containing prompt parameters

        Returns:
            Raw response content from the parsing model

        Raises:
            Exception: If model invocation fails
        """
        # Format the prompt template with the parameters
        formatted_prompt = self.prompt_parser_template.format_messages(**prompt_params)

        # Invoke the LLM
        llm_response = self.prompt_parser_llm.invoke(formatted_prompt)

        # Extract and return the content from the response
        return llm_response.content

    def _validate_parsed_steps(self, parsed_steps: List[Dict[str, str]]) -> None:
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

        for i, step in enumerate(parsed_steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} should be a dictionary")

            for field in self.required_fields:
                if field not in step:
                    raise ValueError(f"Step {i} is missing required field '{field}'")
                
                if not isinstance(step[field], str):
                    raise ValueError(f"Step {i} field '{field}' should be a string")
                
    def _parse_model_response(self, response: str) -> List[TaskIntent]:
        """
        Parse and validate the model's response into structured steps.

        Args:
            response: Raw response from the parsing model

        Returns:
            List of dictionaries representing parsed steps

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            # Parse the JSON response
            parsed_steps: List[Dict[str, str]] = json.loads(response)

            # Validate the structure of the parsed steps
            self._validate_parsed_steps(parsed_steps)

            return [TaskIntent(**step) for step in parsed_steps]
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

    def run_parse_prompt(self) -> List[TaskIntent]:
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
        
        # Construct the parsing prompt
        prompt = self._construct_parsing_prompt()
        
        # Invoke the parsing model
        response = self._invoke_parsing_model(prompt)
        
        # Parse and validate the response
        parsed_steps = self._parse_model_response(response)
        
        return parsed_steps

    @property
    def supported_providers(self) -> List[str]:
        """
        Return list of supported LLM providers for this parser implementation.

        Returns:
            List of supported provider names
        """
        return ["openai"]

    @property
    def parser_name(self) -> str:
        """
        Return the name/identifier of this parser implementation.

        Returns:
            String identifier for this parser
        """
        return "AdaptiqPromptParser"

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Return a summary of the current configuration.

        Returns:
            Dictionary containing key configuration information
        """
        return {
            "parser_name": self.parser_name,
            "supported_providers": self.supported_providers,
            "config_path": self.config_data,
            "llm_model": getattr(self, 'llm_model_name', 'Not configured'),
            "provider": getattr(self, 'provider', 'Not configured'),
            "agent_tools_count": len(getattr(self, 'agent_tools', [])),
        }
import json
import os
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adaptiq.core.reporting.adaptiq_metrics import (
    capture_llm_response,
    instrumental_track_tokens,
)
from adaptiq.core.abstract.integrations.base_prompt_parser import BasePromptParser

instance = CrewPromptParser("path/to/adaptiq_config.yml")


pipeline ()

class CrewPromptParser(BasePromptParser):
    """
    AdaptiqPromptParser analyzes an agent's task description and its declared tools,
    and using an LLM, infers an idealized sequence of steps. Each step is represented
    with the intended subtask, action, preconditions, and expected outcome.
    
    This implementation uses OpenAI's LLM models and supports YAML configuration files.
    """

    def __init__(self, config_path: str):
        """
        Initialize the AdaptiqPromptParser with the path to the configuration file.

        Args:
            config_path: Path to the adaptiq_config.yml file
        """
        super().__init__(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load the configuration file from the given path.

        Args:
            config_path: Path to the configuration file

        Returns:
            Dict containing the configuration
        """
        try:
            with open(config_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {e}")

    def _validate_config(self) -> None:
        """
        Validate that the loaded configuration contains all required fields.

        Raises:
            ValueError: If required configuration fields are missing or invalid
        """
        # Extract and validate LLM configuration
        llm_config = self.adaptiq_config.get("llm_config", {})
        self.llm_model_name = llm_config.get("model_name")
        self.provider = llm_config.get("provider")
        
        if not self.llm_model_name:
            raise ValueError("Model name not found in configuration file")
        
        if not self.provider:
            raise ValueError("Provider not found in configuration file")
        
        if self.provider not in self.supported_providers:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: {', '.join(self.supported_providers)}"
            )

        # Extract and validate agent configuration
        agent_config = self.adaptiq_config.get("agent_modifiable_config", {})
        self.agent_task_description_path = agent_config.get("prompt_configuration_file_path")
        
        if not self.agent_task_description_path:
            raise ValueError(
                "Agent task description path not found in configuration file"
            )

        # Extract tools configuration
        tools_config = agent_config.get("agent_tools", [])
        self.agent_tools = (
            [{tool["name"]: tool["description"]} for tool in tools_config]
            if tools_config
            else []
        )

    def _initialize_components(self) -> None:
        """
        Initialize LLM client and prompt template based on configuration.

        Raises:
            ValueError: If components cannot be initialized
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from config or environment
        api_key = self.adaptiq_config.get("llm_config", {}).get("api_key") or os.getenv(
            "OPENAI_API_KEY"
        )
        if not api_key:
            raise ValueError("OpenAI API key not found in configuration or environment")

        # Initialize LLM client
        if self.provider == "openai":
            self.prompt_parser_llm = ChatOpenAI(
                temperature=0.0, 
                model=self.llm_model_name, 
                api_key=api_key
            )
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        # Initialize prompt template
        self.prompt_parser_template = self._create_prompt_template()

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
            "Intended_SubTask": "Retrieve company information",
            "Intended_Action": "Use_Tool_ReadFileContent",
            "Preconditions_Mentioned_In_Prompt": "None explicitly mentioned for this first step.",
            "Expected_Ideal_Outcome_Mentioned_In_Prompt": "Company background information is available."
        }},
        {{
            "Intended_SubTask": "Retrieve lead's personal information",
            "Intended_Action": "Use_Tool_LeadNameTool",
            "Preconditions_Mentioned_In_Prompt": "After company info is retrieved.",
            "Expected_Ideal_Outcome_Mentioned_In_Prompt": "Lead's name and email are available."
        }}
        ]

        Ensure your output is a perfectly formatted JSON array. Do not include any explanations or additional text outside the JSON array."""

        return ChatPromptTemplate.from_template(prompt_template)

    def _load_agent_task_description(self) -> str:
        """
        Load and extract the agent's task description from the specified file.

        Returns:
            String containing the agent's task description
        """
        try:
            with open(self.agent_task_description_path, "r", encoding="utf-8") as file:
                tasks_data = yaml.safe_load(file)

            # Extract the task description - this might need to be adjusted
            # based on the actual structure of the tasks.yaml file
            agent_name = self.adaptiq_config.get("agent_modifiable_config", {}).get(
                "agent_name"
            )

            # Try to find the task description
            task_description = None

            # First approach: Look for task with matching agent name
            if isinstance(tasks_data, list):
                for task in tasks_data:
                    if isinstance(task, dict) and task.get("agent") == agent_name:
                        task_description = task.get("description")
                        break

            # Second approach: Look for a direct description field
            if task_description is None and isinstance(tasks_data, dict):
                task_description = tasks_data.get("description")

            # If still not found, try to get any description field from nested dictionaries
            if task_description is None and isinstance(tasks_data, dict):
                for key, value in tasks_data.items():
                    if isinstance(value, dict) and "description" in value:
                        task_description = value["description"]
                        break

            # If all else fails, just return the entire yaml content as a string
            if task_description is None:
                return str(tasks_data)

            return task_description
        except Exception as e:
            raise ValueError(f"Failed to load agent task description: {e}")

    def _construct_parsing_prompt(self, task_description: str) -> Dict[str, Any]:
        """
        Construct the complete prompt for the LLM to parse the task description.

        Args:
            task_description: The text of the agent's task description

        Returns:
            Dictionary with the parameters for the prompt template
        """
        tool_strings = [
            f"{name}: {desc}"
            for tool_dict in self.agent_tools
            for name, desc in tool_dict.items()
        ]
        return {
            "agent_tools": ", ".join(tool_strings),
            "task_description_text": task_description,
        }

    @instrumental_track_tokens(mode="pre_run", provider="openai")
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
        capture_llm_response(llm_response)

        # Extract and return the content from the response
        return llm_response.content

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
        try:
            # Parse the JSON response
            parsed_steps = json.loads(response)

            # Validate the structure of the parsed steps
            self._validate_parsed_steps(parsed_steps)

            return parsed_steps
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")

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

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def parse_prompt(self) -> List[Dict[str, Any]]:
        """
        Parse the agent's prompt to infer an idealized sequence of steps.
        
        This method overrides the parent's template method to maintain
        compatibility with the instrumental tracking decorator.

        Returns:
            List of dictionaries, each containing step information with the new keys
        """
        return super().parse_prompt()
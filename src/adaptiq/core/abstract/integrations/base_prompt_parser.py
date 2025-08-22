from abc import ABC, abstractmethod
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from adaptiq.core.entities import AdaptiQConfig, AgentTool, TaskIntent


class BasePromptParser(ABC):
    """
    Abstract base class for prompt parsers that analyze agent task descriptions
    and infer idealized sequences of steps using XML output for reliable parsing.
    
    This class defines the interface that all prompt parser implementations
    must follow, ensuring consistency across different parsing strategies
    and LLM providers.
    """

    def __init__(self, config_data: AdaptiQConfig, task: str, tools: List[AgentTool] = []):
        """
        Initialize the prompt parser with configuration.

        Args:
            config_data: Configuration data object
            task: Task description string
            tools: List of available agent tools
        """
        self.config_data = config_data
        self.llm_model_name = self.config_data.llm_config.model_name
        self.provider = self.config_data.llm_config.provider
        self.api_key = self.config_data.llm_config.api_key
        self.task = task
        self.agent_tools = tools
        self.required_fields = [
            "intended_subtask",  # Fixed: lowercase 's' to match XML output
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
        Create the prompt template for the LLM to parse the task description using XML output.

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
        1. 'Intended_SubTask': A very brief 1-2 word description of the agent's immediate goal for this step (e.g., "Get weather", "Send email", "Analyze data").
        2. 'Intended_Action': The primary strategic action (from Available Tools or conceptual actions like 'Write_Email_Body', 'Formulate_Final_Answer') planned to achieve this sub-task.
        3. 'Preconditions_Mentioned_In_Prompt': Any conditions mentioned in the task description that must be met before this action.
        4. 'Expected_Ideal_Outcome_Mentioned_In_Prompt': What the task description suggests is the successful result of this action.

        OUTPUT FORMAT:
        Return your response as XML with this exact structure:

        <task_steps>
            <step>
                <intended_subtask>1-2 WORDS ONLY</intended_subtask>
                <intended_action>ACTION_OR_TOOL_HERE</intended_action>
                <preconditions_mentioned_in_prompt>PRECONDITIONS_HERE</preconditions_mentioned_in_prompt>
                <expected_ideal_outcome_mentioned_in_prompt>EXPECTED_OUTCOME_HERE</expected_ideal_outcome_mentioned_in_prompt>
            </step>
            <step>
                <intended_subtask>1-2 WORDS ONLY</intended_subtask>
                <intended_action>ACTION_OR_TOOL_2_HERE</intended_action>
                <preconditions_mentioned_in_prompt>PRECONDITIONS_2_HERE</preconditions_mentioned_in_prompt>
                <expected_ideal_outcome_mentioned_in_prompt>EXPECTED_OUTCOME_2_HERE</expected_ideal_outcome_mentioned_in_prompt>
            </step>
            <!-- Add more <step> elements as needed -->
        </task_steps>

        IMPORTANT: 
        - Keep intended_subtask to 1-2 words maximum (e.g., "Describe image", "Create prompt", "Format output")
        - Use clear, descriptive text in other fields
        - Avoid special characters that might break XML
        - If no preconditions exist, use "None" or "No specific preconditions mentioned"
        - Each step should be logically sequential
        - Do not include any text outside the XML structure"""

        return ChatPromptTemplate.from_template(prompt_template)

    def _construct_parsing_prompt(self) -> Dict[str, str]:
        """
        Construct the complete prompt for the LLM to parse the task description.

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

    def _extract_xml_content(self, content: str) -> str:
        """
        Extract XML content from LLM response, handling potential markdown wrapping.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Clean XML content
        """
        # Remove markdown code blocks if present
        if "```xml" in content:
            xml_match = re.search(r"```xml\s*(.*?)\s*```", content, re.DOTALL)
            if xml_match:
                content = xml_match.group(1)
        elif "```" in content:
            # Handle generic code blocks
            xml_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if xml_match:
                content = xml_match.group(1)

        # Look for XML content between <task_steps> tags
        xml_pattern = r"<task_steps>.*?</task_steps>"
        xml_match = re.search(xml_pattern, content, re.DOTALL)
        
        if xml_match:
            return xml_match.group(0)
        else:
            # If no wrapper found, assume the entire content is XML
            return content.strip()

    def _parse_xml_response(self, xml_content: str) -> List[Dict[str, str]]:
        """
        Parse XML response and extract task steps.
        
        Args:
            xml_content: XML string containing task steps
            
        Returns:
            List of dictionaries with parsed step data
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            steps = []
            
            for step_elem in root.findall('step'):
                step_data = {
                    'intended_subtask': self._get_xml_text(step_elem, 'intended_subtask'),  # Fixed: lowercase 's'
                    'intended_action': self._get_xml_text(step_elem, 'intended_action'),
                    'preconditions_mentioned_in_prompt': self._get_xml_text(step_elem, 'preconditions_mentioned_in_prompt'),
                    'expected_ideal_outcome_mentioned_in_prompt': self._get_xml_text(step_elem, 'expected_ideal_outcome_mentioned_in_prompt')
                }
                steps.append(step_data)
            
            return steps
            
        except ET.ParseError as e:
            # If XML parsing fails, try regex fallback
            print(f"XML parsing error: {e}. Attempting regex fallback.")
            return self._parse_xml_with_regex(xml_content)

    def _get_xml_text(self, element: ET.Element, tag_name: str) -> str:
        """
        Safely extract text from XML element.
        
        Args:
            element: XML element to search in
            tag_name: Tag name to find
            
        Returns:
            Text content or empty string if not found
        """
        child = element.find(tag_name)
        return child.text.strip() if child is not None and child.text else ""

    def _parse_xml_with_regex(self, xml_content: str) -> List[Dict[str, str]]:
        """
        Fallback regex-based XML parsing for malformed XML.
        
        Args:
            xml_content: XML string to parse
            
        Returns:
            List of dictionaries with extracted data
        """
        steps = []
        
        # Pattern to match each step block
        step_pattern = r'<step>(.*?)</step>'
        step_matches = re.findall(step_pattern, xml_content, re.DOTALL)
        
        for step_content in step_matches:
            step_data = {
                'intended_subtask': self._extract_tag_content(step_content, 'intended_subtask'),  # Fixed: lowercase 's'
                'intended_action': self._extract_tag_content(step_content, 'intended_action'),
                'preconditions_mentioned_in_prompt': self._extract_tag_content(step_content, 'preconditions_mentioned_in_prompt'),
                'expected_ideal_outcome_mentioned_in_prompt': self._extract_tag_content(step_content, 'expected_ideal_outcome_mentioned_in_prompt')
            }
            steps.append(step_data)
        
        return steps

    def _extract_tag_content(self, xml_string: str, tag_name: str) -> str:
        """
        Extract content from a specific XML tag using regex.
        
        Args:
            xml_string: XML string to search in
            tag_name: Name of the tag to extract
            
        Returns:
            Content of the tag or empty string if not found
        """
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        match = re.search(pattern, xml_string, re.DOTALL)
        return match.group(1).strip() if match else ""

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
        Parse and validate the model's XML response into structured steps.

        Args:
            response: Raw response from the parsing model

        Returns:
            List of TaskIntent objects representing parsed steps

        Raises:
            ValueError: If response cannot be parsed or is invalid
        """
        try:
            # Extract XML content from response
            xml_content = self._extract_xml_content(response)
            
            # Parse XML and extract steps
            parsed_steps: List[Dict[str, str]] = self._parse_xml_response(xml_content)

            # Validate the structure of the parsed steps
            self._validate_parsed_steps(parsed_steps)

            return [TaskIntent(**step) for step in parsed_steps]
            
        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Failed to parse LLM XML response: {e}\n\n"
            error_msg += f"Raw response:\n{response[:500]}..."  # Show first 500 chars
            raise ValueError(error_msg) from e

    def run_parse_prompt(self) -> List[TaskIntent]:
        """
        Main method to parse the agent's prompt and infer idealized steps.
        
        This template method orchestrates the parsing process by calling
        the methods in the correct sequence.

        Returns:
            List of TaskIntent objects, each containing step information with:
            - intended_subtask: Description of the subtask
            - intended_action: Primary action to be taken
            - preconditions_mentioned_in_prompt: Required preconditions
            - expected_ideal_outcome_mentioned_in_prompt: Expected outcome

        Raises:
            ValueError: If parsing fails at any step
        """
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

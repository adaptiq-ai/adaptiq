import json
import re
import ast
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class HypotheticalStateGenerator:
    """
    Generator that transforms a parsed plan into hypothetical state-action pairs
    using a single LLM invocation. The LLM returns structured state-action mappings.
    """

    def __init__(
        self,
        prompt_parsed_plan: List[Dict],
        model_name: str,
        api_key: str,
        provider: str,
    ):
        """
        Initialize with a parsed plan.

        Args:
            prompt_parsed_plan: List of dictionaries containing intended steps.
        """
        self.prompt_parsed_plan = prompt_parsed_plan
        self.api_key = api_key
        self.model = model_name
        self.provider = provider

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        # Updated prompt to prevent JSON formatting issues with escaped quotes
        self.prompt_template = ChatPromptTemplate.from_template(
            """
        You are an RL state-action pair generator for agent training.

        TASK:
        For EACH step in the provided plan, create a corresponding state-action pair.

        STATE FORMAT (as tuple string):
        ('Current_SubTask_Category', 'Last_Action_Taken', 'Last_Outcome_Category', 'Key_Context')

        Where:
        - Current_SubTask_Category: Identify the general category/type of subtask based on what you observe in the current step
        - Last_Action_Taken: The Intended_Action from previous step (use "None" for first step)
        - Last_Outcome_Category: Categorize from: [Success_DataFound, Success_ActionCompleted, Success_NoDataFound, Failure_PreconditionNotMet, Outcome_Unknown, None]
        - Key_Context: 1-3 keywords (max 3 words) summarizing information up to this point

        ACTION:
        The Intended_Action from the current step.

        OUTPUT:
        A JSON list where each item has:
        1. "hypothetical_state_representation": The 4-element tuple string
        2. "hypothetical_action": The current step's Intended_Action
        3. "source_prompt_step_details": Copy of the original step object

        IMPORTANT: Use single quotes inside JSON strings to avoid escaping issues.
        For example, write "Expected_Ideal_Outcome_Mentioned_In_Prompt": "Lead's name is retrieved."
        instead of using escaped double quotes.

        Parse this plan: {parsed_plan}
        """
        )

    def configure_llm(self, llm_instance: ChatOpenAI, prompt_template: str = None):
        """
        Configure the LLM and prompt template.

        Args:
            llm_instance: LLM to use for state representation
            prompt_template: Optional custom prompt template
        """
        self.llm = llm_instance
        if prompt_template:
            self.prompt_template = ChatPromptTemplate.from_template(prompt_template)

    def generate_hypothetical_state_action_pairs(self) -> List[Dict]:
        """
        Generate all hypothetical state-action pairs in a single LLM call.

        Returns:
            List of state-action pairs with detailed step context.
        """
        # Prepare full context for LLM
        context = {"parsed_plan": self.prompt_parsed_plan}

        prompt = self.prompt_template.format_messages(**context)
        response = self.llm.invoke(prompt)

        # Get the content from the response
        content = response.content

        # Check if response is wrapped in markdown code blocks (```json ... ```)
        if "```" in content:
            # Extract content between code blocks
            code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
            matches = re.findall(code_block_pattern, content)
            if matches:
                # Use the first matched code block
                content = matches[0].strip()

        # Try parsing as JSON first, then Python literal
        try:
            # Apply comprehensive JSON fixing
            fixed_content = self._comprehensive_json_fix(content)
            result = json.loads(fixed_content)
        except Exception as e:
            # If JSON parsing fails, try Python literal parsing
            try:
                result = self._parse_python_literal(content)
            except Exception as literal_e:
                # If that fails too, try manual parsing
                try:
                    result = self._parse_json_manually(content)
                except Exception as inner_e:
                    # If all parsing attempts fail, provide detailed error info
                    error_msg = f"Failed to parse LLM response. Tried JSON: {e}, Python literal: {literal_e}, Manual: {inner_e}\n\n"
                    error_msg += f"Processed content:\n{content}\n\n"
                    error_msg += f"Original response:\n{response.content}"
                    raise ValueError(error_msg) from inner_e

        return result
    
    def _parse_python_literal(self, content: str) -> List[Dict]:
        """
        Parse Python literal format (single quotes, tuples) returned by some models.
        """
        try:
            # Safely evaluate the Python literal
            result = ast.literal_eval(content.strip())
            
            # Convert tuples to strings for consistency
            for item in result:
                if isinstance(item.get('hypothetical_state_representation'), tuple):
                    item['hypothetical_state_representation'] = str(item['hypothetical_state_representation'])
            
            return result
        except Exception as e:
            raise ValueError(f"Failed to parse Python literal: {e}")

    def _comprehensive_json_fix(self, json_str: str) -> str:
        """
        Comprehensive JSON fixing that handles various issues.

        Args:
            json_str: JSON string with potential issues

        Returns:
            Fixed JSON string
        """
        # Step 1: Normalize line endings
        fixed_str = json_str.replace("\r\n", "\n").replace("\r", "\n")

        # Step 2: Handle escaped quotes properly
        # Replace \" with ' when it appears within strings (between quotes)
        fixed_str = re.sub(r"(\"[^\"]*?)\\\"([^\"]*?\")", r"\1\'\2", fixed_str)

        # Step 3: Fix control characters
        control_chars = ["\b", "\f", "\n", "\r", "\t"]
        for char in control_chars:
            # Replace unescaped control characters within strings
            fixed_str = re.sub(f'(?<="[^"]*){char}(?=[^"]*")', " ", fixed_str)

        # Step 4: Fix quotes around JSON keys
        # This finds JSON keys that are not properly quoted
        fixed_str = re.sub(
            r"([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', fixed_str
        )

        # Step 5: Fix issues with apostrophes
        # Convert apostrophes in words to avoid JSON parsing issues
        fixed_str = re.sub(r'(\w)"(\w)', r"\1'\2", fixed_str)

        # Step 6: Fix missing quotes around string values
        # This is more complex and might need refinement for specific cases

        return fixed_str

    def _parse_json_manually(self, content: str) -> List[Dict]:
        """
        Manual JSON parsing for cases where automatic parsing fails.
        Implements a simplified parser for the specific structure we expect.

        Args:
            content: The JSON-like string to parse

        Returns:
            List of dictionaries representing the parsed JSON
        """
        result = []

        # Pattern to match each JSON object in the array
        object_pattern = r'{\s*"hypothetical_state_representation":\s*"([^"]*)",\s*"hypothetical_action":\s*"([^"]*)",\s*"source_prompt_step_details":\s*{([^}]*)}\s*}'

        # Find all objects in the content
        objects = re.findall(object_pattern, content, re.DOTALL)

        for obj in objects:
            state_repr = obj[0]
            action = obj[1]
            details_str = obj[2]

            # Parse the details dictionary
            details = {}
            detail_pattern = r'"([^"]*)"\s*:\s*"([^"]*)"'
            detail_matches = re.findall(detail_pattern, details_str)

            for key, value in detail_matches:
                details[key] = value

            # Create the result dictionary
            result.append(
                {
                    "hypothetical_state_representation": state_repr,
                    "hypothetical_action": action,
                    "source_prompt_step_details": details,
                }
            )

        return result

    def clean_representation(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Cleaner for LLM output. Handles escaped quotes and properly formats JSON.

        Args:
            raw_data: Raw state-action pairs

        Returns:
            Cleaned list of dictionaries
        """
        cleaned = []

        for item in raw_data:
            # Process any potentially problematic fields
            details = item.get("source_prompt_step_details", {})
            if isinstance(details, dict):
                # Deep clone to avoid modifying the original
                cleaned_details = {}
                for key, value in details.items():
                    # Handle escaped quotes in string values
                    if isinstance(value, str):
                        # Normalize the string value to avoid escaped quote issues
                        cleaned_details[key] = value.replace('\\"', '"').replace(
                            '\\"', '"'
                        )
                    else:
                        cleaned_details[key] = value
            else:
                cleaned_details = details

            cleaned.append(
                {
                    "state": item["hypothetical_state_representation"],
                    "action": item["hypothetical_action"],
                    "details": cleaned_details,
                }
            )

        return cleaned

import ast
import logging
import re
from typing import Any, Dict, List
import xml.etree.ElementTree as ET

from langchain_core.prompts import ChatPromptTemplate

from adaptiq.core.abstract.q_table.base_state_mapper import BaseStateMapper
from adaptiq.core.entities import (
    Classification,
    ClassificationResponse,
    LogItem
)

logger = logging.getLogger(__name__)


class StateMapper(BaseStateMapper):
    """
    AdaptiqStateMapper - Matches execution trace states with Q-table states using XML format.

    Takes the "Warmed Q-table" (from previous runs) and matches input states to see
    if they correspond to any known state from the Q-table, ignoring actions completely.
    """

    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """Create the XML-based prompt template for state classification."""
        classification_template = """You are an AI Agent State Classifier specializing in semantic matching.

                # Input State to Classify:
                ```
                {input_state}
                ```

                # KNOWN STATES (Find the closest semantic match):
                ```
                {known_states}
                ```

                # Your Task:
                1. Analyze the provided input state components WITHOUT considering any action values.
                2. Find the semantically closest match from the known states list.
                3. Focus ONLY on matching the core state concepts (ignore syntax differences between arrays/tuples).
                4. Pay attention to the semantic meaning of the state components:
                - First component: Task/phase name (e.g., "RetrieveCompanyInfo" vs "InformationRetrieval_Company")
                - Second component: Previous tool used
                - Third component: Status/outcome
                - Fourth component: Context/data description

                # Examples of semantic matches:
                - "RetrieveCompanyInfo" could match with "InformationRetrieval_Company" (both about company info retrieval)
                - "CompanyData" could match with "company background" (both about company information)
                - "None" should match with "None" (both indicate no previous state)

                # OUTPUT FORMAT:

                State is represented as a 4-element list: ["...", "...", "...", "..."]

                Elements:
                - Extract the **core meaning** of "current_sub_task_or_thought" → first element
                - Use "last_action_taken" → second element (or None if not present)
                - Use "last_outcome" → third element (or None if not present)  
                - Extract the **main role or situation** from "agent_context" → fourth element

                Return your response as XML with this exact structure:

                <classification_result>
                    <is_known_state>true/false</is_known_state>
                    <matched_state>The exact matching state from known states if found, format ["...", "...", "...", "..."]</matched_state>
                    <new_state>If no match found, create new state representation, format ["...", "...", "...", "..."]</new_state>
                    <action>The action associated with the state (tool name only, e.g. "FileReadTool"), or null</action>
                    <reasoning>Explanation of why this state matches or doesn't match a known state</reasoning>
                </classification_result>

                # IMPORTANT:
                - Use clear, concise text in each field
                - Avoid special characters that might break XML
                - Use "null" as text when no value is found
                - Use "true" or "false" as text for is_known_state
                - IGNORE any "action" field in input - ONLY match on state components
                - Find CLOSEST semantic match, not just exact string matches
                - Return EXACT matching state string from known states without modification
                - Only return is_known_state: true if there's a clear semantic match

                Return ONLY the XML structure, no additional text."""

        return ChatPromptTemplate.from_template(classification_template)

    def _invoke_llm_for_classification(
        self, input_state: LogItem
    ) -> ClassificationResponse:
        """
        Invoke the LLM to classify a state using XML output format.

        Args:
            input_state: LogItem to classify

        Returns:
            ClassificationResponse containing the LLM's classification output
        """
        try:
            # Create formatted known states for better comparison
            formatted_known_states = []
            for original, parsed in self.parsed_states:
                formatted_known_states.append({"original": original, "components": parsed})
            
            log_state = input_state.key.state

            # Create inputs for the LLM
            inputs = {
                "input_state": {
                    "current_sub_task_or_thought": log_state.current_sub_task_or_thought,
                    "last_action_taken": log_state.last_action_taken, 
                    "last_outcome": log_state.last_outcome,
                    "agent_context": log_state.agent_context,
                    "action": input_state.key.agent_action
                },
                "known_states": self._format_known_states_for_display(formatted_known_states),
            }

            # Create and invoke the prompt
            prompt = self.classification_prompt_template.format_messages(**inputs)
            response = self.llm.invoke(prompt)

            # Extract XML content from response
            xml_content = self._extract_xml_content(response.content)

            # Parse XML and extract classification data
            classification_data = self._parse_xml_response(xml_content)

            return ClassificationResponse(
                input_state=classification_data["input_state"],
                classification=Classification(**classification_data["classification"])
            )

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Return fallback response
            return ClassificationResponse(
                input_state={
                    "state": [None, None, None, None],
                    "action": None
                },
                classification=Classification(
                    is_known_state=False,
                    # FIX: Convert the list to a string to match the Pydantic model
                    state=str([None, None, None, None]),
                    reasoning=f"Classification error: {str(e)}"
                )
            )

    def _format_known_states_for_display(self, formatted_known_states: List[Dict]) -> str:
        """
        Format known states for display in the prompt.

        Args:
            formatted_known_states: List of state dictionaries

        Returns:
            Formatted string representation of known states
        """
        if not formatted_known_states:
            return "No known states available."

        result = []
        for i, state_info in enumerate(formatted_known_states):
            result.append(f"State {i+1}:")
            result.append(f"  Original: {state_info['original']}")
            result.append(f"  Components: {state_info['components']}")
            result.append("")
        
        return "\n".join(result)

    def _extract_xml_content(self, content: str) -> str:
        """
        Extract XML content from LLM response, handling potential markdown wrapping.

        Args:
            content: Raw LLM response content

        Returns:
            Clean XML content
        """
        if not content:
            return ""

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

        # Look for XML content between <classification_result> tags
        xml_pattern = r"<classification_result>.*?</classification_result>"
        xml_match = re.search(xml_pattern, content, re.DOTALL)

        if xml_match:
            return xml_match.group(0)
        else:
            # If no wrapper found, assume the entire content is XML
            return content.strip()
        
    def _parse_state_list(self, state_text: str) -> List[Any]:
        """
        Safely parse a state string into a 4-element list.

        Args:
            state_text: String representation of state

        Returns:
            List of 4 elements representing the state
        """
        if not state_text or state_text.lower() == "null":
            return [None, None, None, None]

        # Ensure we have a string
        if not isinstance(state_text, str):
            return [str(state_text), None, None, None]

        state_text = state_text.strip()

        # Handle empty or whitespace-only strings
        if not state_text:
            return [None, None, None, None]

        try:
            # Try to parse as literal (list or tuple)
            if (state_text.startswith("[") and state_text.endswith("]")) or \
               (state_text.startswith("(") and state_text.endswith(")")):
                parsed = ast.literal_eval(state_text)
                if isinstance(parsed, (list, tuple)):
                    # Ensure we have exactly 4 elements
                    result = list(parsed)
                    while len(result) < 4:
                        result.append(None)
                    return result[:4]  # Truncate if more than 4
                else:
                    return [str(parsed), None, None, None]
            else:
                # Treat as single string element
                return [state_text, None, None, None]

        except (SyntaxError, ValueError) as e:
            logger.warning(f"Failed to parse state '{state_text}': {e}")
            return [state_text, None, None, None]

    def _parse_xml_response(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse XML response and extract classification data.

        Args:
            xml_content: XML string containing classification result

        Returns:
            Dictionary with parsed classification data compatible with entities
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)

            # Extract classification components
            is_known_state_str = self._get_xml_text(root, "is_known_state")
            matched_state_text = self._get_xml_text(root, "matched_state")
            new_state_text = self._get_xml_text(root, "new_state")
            action = self._get_xml_text(root, "action")
            reasoning = self._get_xml_text(root, "reasoning")

            # Convert string boolean to actual boolean
            is_known_state = is_known_state_str and is_known_state_str.lower() == "true"

            # Normalize action
            if not action or action.lower() == "null":
                action = None

            # Parse states into lists
            matched_state = self._parse_state_list(matched_state_text)
            new_state = self._parse_state_list(new_state_text)

            # Determine final state based on classification result
            if is_known_state and matched_state:
                final_state = matched_state
            elif new_state:
                final_state = new_state
            else:
                # Fallback to default state
                final_state = [None, None, None, None]

            return {
                "classification": {
                    "is_known_state": is_known_state,
                    "state": f"[{', '.join(map(str, final_state))}]",
                    "reasoning": reasoning or "No reasoning provided",
                },
                "input_state": {
                    "state": final_state,
                    "action": action,
                },
            }

        except ET.ParseError as e:
            # If XML parsing fails, try regex fallback
            logger.warning(f"XML parsing error: {e}. Attempting regex fallback.")
            return self._parse_xml_with_regex(xml_content)

    def _get_xml_text(self, element: ET.Element, tag_name: str) -> str:
        """
        Safely extract text from XML element.

        Args:
            element: XML element to search in
            tag_name: Tag name to find

        Returns:
            Text content or None if not found
        """
        child = element.find(tag_name)
        if child is not None and child.text:
            return child.text.strip()
        return None

    def _parse_xml_with_regex(self, xml_content: str) -> Dict[str, Any]:
        """
        Fallback regex-based XML parsing for malformed XML.

        Args:
            xml_content: XML string to parse

        Returns:
            Dictionary with extracted classification data
        """
        # Extract individual components using regex
        is_known_state_str = self._extract_tag_content(xml_content, "is_known_state")
        matched_state_text = self._extract_tag_content(xml_content, "matched_state")
        new_state_text = self._extract_tag_content(xml_content, "new_state")
        action = self._extract_tag_content(xml_content, "action")
        reasoning = self._extract_tag_content(xml_content, "reasoning")

        # Convert string boolean to actual boolean
        is_known_state = is_known_state_str and is_known_state_str.lower() == "true"

        # Normalize action
        if not action or action.lower() == "null":
            action = None

        # Parse states
        matched_state = self._parse_state_list(matched_state_text)
        new_state = self._parse_state_list(new_state_text)

        # Determine final state
        if is_known_state and matched_state:
            final_state = matched_state
        elif new_state:
            final_state = new_state
        else:
            final_state = [None, None, None, None]

        return {
            "classification": {
                "is_known_state": is_known_state,
                # FIX: Convert the final_state list to a string, just like in _parse_xml_response
                "state": f"[{', '.join(map(str, final_state))}]",
                "reasoning": reasoning or "Error parsing LLM output",
            },
            "input_state": {
                "state": final_state,
                "action": action,
            },
        }

    def _extract_tag_content(self, xml_string: str, tag_name: str) -> str:
        """
        Extract content from a specific XML tag using regex.

        Args:
            xml_string: XML string to search in
            tag_name: Name of the tag to extract

        Returns:
            Content of the tag or empty string if not found
        """
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, xml_string, re.DOTALL)
        return match.group(1).strip() if match else ""

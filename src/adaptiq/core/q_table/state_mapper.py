import re
import xml.etree.ElementTree as ET
import logging
from langchain_core.prompts import ChatPromptTemplate
from adaptiq.core.abstract.q_table.base_state_mapper import BaseStateMapper
from adaptiq.core.entities import StateActionMapping, ClassificationResponse, Classification

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

        OUTPUT FORMAT:
        Return your response as XML with this exact structure:

        <classification_result>
            <is_known_state>true/false</is_known_state>
            <matched_state>The exact matching state from the known states if found, or null if not found</matched_state>
            <reasoning>Explanation of why this state matches or doesn't match a known state, with component-by-component comparison</reasoning>
        </classification_result>

        IMPORTANT:
        - Use clear, concise text in each field
        - Avoid special characters that might break XML
        - Use "null" as text for matched_state when no match is found
        - Use "true" or "false" as text for is_known_state

        CRITICAL REQUIREMENTS:
        - IGNORE any "action" field in the input state - ONLY match on the state components
        - If the input is a dictionary with a "state" key, extract and use ONLY that state field
        - Find the CLOSEST semantic match, not just exact string matches
        - You must return the EXACT matching state string from the known states without modification
        - Only return is_known_state: true if there's a clear semantic match
        - Be thorough in your reasoning, explaining similarities and differences by component

        Return ONLY the XML structure, no additional text."""

        return ChatPromptTemplate.from_template(classification_template)

    def _invoke_llm_for_classification(self, input_state: StateActionMapping) -> ClassificationResponse:
        """
        Invoke the LLM to classify a state using XML output format.

        Args:
            input_state: State to classify (can be string, list, or dict)

        Returns:
            ClassificationResponse containing the LLM's classification output
        """
        # Create formatted known states for better comparison
        formatted_known_states = []
        for original, parsed in self.parsed_states:
            formatted_known_states.append({"original": original, "components": parsed})

        # Create inputs for the LLM
        inputs = {
            "input_state": input_state.state,
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
            classification=Classification(**classification_data)
        )

    def _format_known_states_for_display(self, formatted_known_states: list) -> str:
        """
        Format known states for display in the prompt.
        
        Args:
            formatted_known_states: List of state dictionaries
            
        Returns:
            Formatted string representation of known states
        """
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

    def _parse_xml_response(self, xml_content: str) -> dict:
        """
        Parse XML response and extract classification data.
        
        Args:
            xml_content: XML string containing classification result
            
        Returns:
            Dictionary with parsed classification data
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract classification components
            is_known_state_str = self._get_xml_text(root, 'is_known_state')
            matched_state = self._get_xml_text(root, 'matched_state')
            reasoning = self._get_xml_text(root, 'reasoning')
            
            # Convert string boolean to actual boolean
            is_known_state = is_known_state_str.lower() == 'true'
            
            # Handle null matched_state
            if matched_state.lower() == 'null' or matched_state == '':
                matched_state = None
            
            return {
                'is_known_state': is_known_state,
                'matched_state': matched_state,
                'reasoning': reasoning
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
            Text content or empty string if not found
        """
        child = element.find(tag_name)
        return child.text.strip() if child is not None and child.text else ""

    def _parse_xml_with_regex(self, xml_content: str) -> dict:
        """
        Fallback regex-based XML parsing for malformed XML.
        
        Args:
            xml_content: XML string to parse
            
        Returns:
            Dictionary with extracted classification data
        """
        # Extract individual components using regex
        is_known_state_str = self._extract_tag_content(xml_content, 'is_known_state')
        matched_state = self._extract_tag_content(xml_content, 'matched_state')
        reasoning = self._extract_tag_content(xml_content, 'reasoning')
        
        # Convert string boolean to actual boolean
        is_known_state = is_known_state_str.lower() == 'true' if is_known_state_str else False
        
        # Handle null matched_state
        if matched_state.lower() == 'null' or matched_state == '':
            matched_state = None
        
        return {
            'is_known_state': is_known_state,
            'matched_state': matched_state,
            'reasoning': reasoning or "Error parsing LLM output"
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
        pattern = f'<{tag_name}>(.*?)</{tag_name}>'
        match = re.search(pattern, xml_string, re.DOTALL)
        return match.group(1).strip() if match else ""
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from adaptiq.core.entities import TaskIntent, HypotheticalStateRepresentation

class HypotheticalStateGenerator:
    """
    Generator that transforms a parsed plan into hypothetical state-action pairs
    using XML output format for more reliable parsing.
    """

    def __init__(
        self,
        prompt_parsed_plan: List[TaskIntent],
        llm: BaseChatModel
    ):
        """
        Initialize with a parsed plan.

        Args:
            prompt_parsed_plan: List of dictionaries containing intended steps.
        """
        self.prompt_parsed_plan = prompt_parsed_plan
        self.llm = llm

        # XML-based prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """
        You are an RL state-action pair generator for agent training.

        TASK:
        For EACH step in the provided plan, create a corresponding state-action pair.

        STATE COMPONENTS:
        - Current_SubTask_Category: Identify the general category/type of subtask based on what you observe in the current step
        - Last_Action_Taken: The Intended_Action from previous step (use "None" for first step)
        - Last_Outcome_Category: Categorize from: [Success_DataFound, Success_ActionCompleted, Success_NoDataFound, Failure_PreconditionNotMet, Outcome_Unknown, None]
        - Key_Context: 1-3 keywords (max 3 words) summarizing information up to this point

        OUTPUT FORMAT:
        Return your response as XML with this exact structure:

        <state_action_pairs>
            <pair>
                <current_subtask_category>CATEGORY_HERE</current_subtask_category>
                <last_action_taken>ACTION_HERE</last_action_taken>
                <last_outcome_category>OUTCOME_HERE</last_outcome_category>
                <key_context>CONTEXT_HERE</key_context>
                <hypothetical_action>INTENDED_ACTION_HERE</hypothetical_action>
                <source_step>
                    <step_number>NUMBER</step_number>
                    <intended_action>ACTION</intended_action>
                    <expected_outcome>OUTCOME</expected_outcome>
                    <reasoning>REASONING</reasoning>
                </source_step>
            </pair>
            <!-- Repeat <pair> for each step -->
        </state_action_pairs>

        IMPORTANT: 
        - Use clear, concise text in each field
        - Avoid special characters that might break XML
        - Keep Key_Context to maximum 3 words
        - Use the exact outcome categories listed above

        Parse this plan: {parsed_plan}
        """
        )

    def generate_hypothetical_state_action_pairs(self) -> List[HypotheticalStateRepresentation]:
        """
        Generate all hypothetical state-action pairs using XML output.

        Returns:
            List of HypotheticalStateRepresentation objects.
        """
        # Prepare context for LLM
        context = {"parsed_plan": [plan.model_dump() for plan in self.prompt_parsed_plan]}

        prompt = self.prompt_template.format_messages(**context)
        response = self.llm.invoke(prompt)

        # Extract XML content from response
        xml_content = self._extract_xml_content(response.content)
        
        # Parse XML and extract state-action pairs
        pairs = self._parse_xml_response(xml_content)
        
        return [HypotheticalStateRepresentation(**pair) for pair in pairs]

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

        # Look for XML content between <state_action_pairs> tags
        xml_pattern = r"<state_action_pairs>.*?</state_action_pairs>"
        xml_match = re.search(xml_pattern, content, re.DOTALL)
        
        if xml_match:
            return xml_match.group(0)
        else:
            # If no wrapper found, assume the entire content is XML
            return content.strip()

    def _parse_xml_response(self, xml_content: str) -> List[Dict]:
        """
        Parse XML response and extract state-action pair data.
        
        Args:
            xml_content: XML string containing state-action pairs
            
        Returns:
            List of dictionaries with parsed data
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            pairs = []
            
            for pair_elem in root.findall('pair'):
                # Extract state components
                current_subtask = self._get_xml_text(pair_elem, 'current_subtask_category')
                last_action = self._get_xml_text(pair_elem, 'last_action_taken')
                last_outcome = self._get_xml_text(pair_elem, 'last_outcome_category')
                key_context = self._get_xml_text(pair_elem, 'key_context')
                
                # Create state tuple string
                state_tuple = f"('{current_subtask}', '{last_action}', '{last_outcome}', '{key_context}')"
                
                # Extract action
                action = self._get_xml_text(pair_elem, 'hypothetical_action')
                
                # Extract source step details
                source_step_elem = pair_elem.find('source_step')
                source_details = {}
                
                if source_step_elem is not None:
                    source_details = {
                        'step_number': self._get_xml_text(source_step_elem, 'step_number'),
                        'intended_action': self._get_xml_text(source_step_elem, 'intended_action'),
                        'expected_outcome': self._get_xml_text(source_step_elem, 'expected_outcome'),
                        'reasoning': self._get_xml_text(source_step_elem, 'reasoning')
                    }
                
                pairs.append({
                    'state': state_tuple,
                    'action': action,
                    'details': source_details
                })
            
            return pairs
            
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

    def _parse_xml_with_regex(self, xml_content: str) -> List[Dict]:
        """
        Fallback regex-based XML parsing for malformed XML.
        
        Args:
            xml_content: XML string to parse
            
        Returns:
            List of dictionaries with extracted data
        """
        pairs = []
        
        # Pattern to match each pair block
        pair_pattern = r'<pair>(.*?)</pair>'
        pair_matches = re.findall(pair_pattern, xml_content, re.DOTALL)
        
        for pair_content in pair_matches:
            # Extract individual components using regex
            current_subtask = self._extract_tag_content(pair_content, 'current_subtask_category')
            last_action = self._extract_tag_content(pair_content, 'last_action_taken')
            last_outcome = self._extract_tag_content(pair_content, 'last_outcome_category')
            key_context = self._extract_tag_content(pair_content, 'key_context')
            action = self._extract_tag_content(pair_content, 'hypothetical_action')
            
            # Create state tuple
            state_tuple = f"('{current_subtask}', '{last_action}', '{last_outcome}', '{key_context}')"
            
            # Extract source step details
            source_details = {}
            source_step_match = re.search(r'<source_step>(.*?)</source_step>', pair_content, re.DOTALL)
            if source_step_match:
                source_content = source_step_match.group(1)
                source_details = {
                    'step_number': self._extract_tag_content(source_content, 'step_number'),
                    'intended_action': self._extract_tag_content(source_content, 'intended_action'),
                    'expected_outcome': self._extract_tag_content(source_content, 'expected_outcome'),
                    'reasoning': self._extract_tag_content(source_content, 'reasoning')
                }
            
            pairs.append({
                'state': state_tuple,
                'action': action,
                'details': source_details
            })
        
        return pairs

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
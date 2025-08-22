import xml.etree.ElementTree as ET
from typing import Dict, List

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from adaptiq.core.entities import FormattedAnalysis


class PromptConsulting:
    """
    Consultant that analyzes a given prompt and provides structured feedback
    using a single LLM invocation. The LLM returns structured analysis and recommendations.
    """

    def __init__(self, agent_prompt: str, llm: BaseChatModel):
        """
        Initialize with an agent prompt to analyze.

        Args:
            agent_prompt: The prompt text to be analyzed.
            llm: BaseChatModel instance for LLM interactions.
        """
        self.agent_prompt = agent_prompt
        self.llm = llm

        self.analysis_template = ChatPromptTemplate.from_template(
            """
        You are an expert Prompt Consultant for AI Agents.
        You will be given a prompt intended to guide the behavior of an AI agent.
        Your tasks:
        1. Summarize what the prompt is trying to accomplish.
        2. Identify the strengths and positive aspects of the prompt.
        3. Identify any potential weaknesses, ambiguities, or inconsistencies.
        4. Suggest specific improvements or edits to make it more effective.
        5. Provide 3 best practices for structuring prompts like this.
        6. Highlight any missing components (e.g., role definition, output format, input expectations).

        Here is the prompt to analyze:
        {agent_prompt}

        Respond in the following structured XML format:
        <analysis>
            <summary>Brief summary of what the prompt accomplishes</summary>
            <strengths>
                <strength>First strength identified</strength>
                <strength>Second strength identified</strength>
            </strengths>
            <weaknesses>
                <weakness>First weakness identified</weakness>
                <weakness>Second weakness identified</weakness>
            </weaknesses>
            <suggested_modifications>
                <modification>First suggested improvement</modification>
                <modification>Second suggested improvement</modification>
            </suggested_modifications>
            <best_practices>
                <practice>First best practice</practice>
                <practice>Second best practice</practice>
                <practice>Third best practice</practice>
            </best_practices>
            <missing_components>
                <component>First missing component</component>
                <component>Second missing component</component>
            </missing_components>
        </analysis>
        """
        )

    def analyze_prompt(self) -> FormattedAnalysis:
        """
        Analyze the prompt and generate structured feedback in a single LLM call.

        Returns:
            FormattedAnalysis containing analysis and recommendations.
        """
        # Prepare context for LLM
        context = {"agent_prompt": self.agent_prompt}

        prompt = self.analysis_template.format_messages(**context)
        response = self.llm.invoke(prompt)

        # Try parsing XML from response
        try:
            result = self._parse_xml_response(response.content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response as XML: {e}\n\nResponse:\n{response.content}"
            ) from e

        return self.get_formatted_analysis(raw_analysis=result)

    def _parse_xml_response(self, xml_content: str) -> Dict:
        """
        Parse XML response from LLM into a dictionary.

        Args:
            xml_content: XML string from LLM response

        Returns:
            Dictionary with parsed analysis components
        """
        # Clean up the XML content - remove any extra text before/after XML
        xml_content = xml_content.strip()

        # Find the XML content between <analysis> tags if it exists
        start_tag = xml_content.find("<analysis>")
        end_tag = xml_content.find("</analysis>") + len("</analysis>")

        if start_tag != -1 and end_tag != -1:
            xml_content = xml_content[start_tag:end_tag]

        # Parse the XML
        root = ET.fromstring(xml_content)

        # Extract data from XML structure
        result = {
            "summary": self._get_text_content(root, "summary"),
            "strengths": self._get_list_content(root, "strengths", "strength"),
            "weaknesses": self._get_list_content(root, "weaknesses", "weakness"),
            "suggested_modifications": self._get_list_content(
                root, "suggested_modifications", "modification"
            ),
            "best_practices": self._get_list_content(
                root, "best_practices", "practice"
            ),
            "missing_components": self._get_list_content(
                root, "missing_components", "component"
            ),
        }

        return result

    def _get_text_content(self, root: ET.Element, tag_name: str) -> str:
        """
        Extract text content from a single XML element.

        Args:
            root: Root XML element
            tag_name: Name of the tag to extract text from

        Returns:
            Text content of the element, or empty string if not found
        """
        element = root.find(tag_name)
        return element.text.strip() if element is not None and element.text else ""

    def _get_list_content(
        self, root: ET.Element, parent_tag: str, child_tag: str
    ) -> List[str]:
        """
        Extract list content from XML structure.

        Args:
            root: Root XML element
            parent_tag: Name of the parent container tag
            child_tag: Name of the child item tags

        Returns:
            List of text content from child elements
        """
        parent_element = root.find(parent_tag)
        if parent_element is None:
            return []

        items = []
        for child in parent_element.findall(child_tag):
            if child.text:
                items.append(child.text.strip())

        return items

    def get_formatted_analysis(self, raw_analysis: Dict) -> FormattedAnalysis:
        """
        Format the analysis results or get a new analysis if none provided.

        Args:
            raw_analysis: Optional pre-generated analysis results

        Returns:
            Formatted analysis dictionary
        """

        # Ensure all expected keys exist
        formatted_analysis = FormattedAnalysis(
            summary=raw_analysis.get("summary", ""),
            strengths=raw_analysis.get("strengths", []),
            weaknesses=raw_analysis.get("weaknesses", []),
            suggested_modifications=raw_analysis.get("suggested_modifications", []),
            best_practices=raw_analysis.get("best_practices", []),
            missing_components=raw_analysis.get("missing_components", []),
        )

        return formatted_analysis

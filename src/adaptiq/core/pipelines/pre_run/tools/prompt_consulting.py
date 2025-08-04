
import json
from typing import Dict

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from adaptiq.core.reporting.adaptiq_metrics import (
    capture_llm_response,
    instrumental_track_tokens,
)


class PromptConsulting:
    """
    Consultant that analyzes a given prompt and provides structured feedback
    using a single LLM invocation. The LLM returns structured analysis and recommendations.
    """

    def __init__(self, agent_prompt: str, model_name: str, api_key: str, provider: str):
        """
        Initialize with an agent prompt to analyze.

        Args:
            agent_prompt: The prompt text to be analyzed.
        """
        self.agent_prompt = agent_prompt
        self.api_key = api_key
        self.model = model_name
        self.provider = provider

        if self.provider == "openai":
            self.llm = ChatOpenAI(model=self.model, api_key=self.api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

        self.analysis_template = ChatPromptTemplate.from_template(
            """
        You are an expert Prompt Consultant for AI Agents.
        You will be given a prompt intended to guide the behavior of an AI agent.
        Your tasks:
        1. Summarize what the prompt is trying to accomplish.
        2. Identify any potential weaknesses, ambiguities, or inconsistencies.
        3. Suggest specific improvements or edits to make it more effective.
        4. Provide 3 best practices for structuring prompts like this.
        5. Highlight any missing components (e.g., role definition, output format, input expectations).

        Here is the prompt to analyze:
        {agent_prompt}

        Respond in the following structured format:
        {{
        "summary": "...",
        "weaknesses": ["...", "..."],
        "suggested_modifications": ["...", "..."],
        "best_practices": ["...", "...", "..."],
        "missing_components": ["...", "..."]
        }}
        """
        )

    def configure_llm(self, llm_instance: ChatOpenAI, prompt_template: str = None):
        """
        Configure the LLM and prompt template.

        Args:
            llm_instance: LLM to use for prompt analysis
            prompt_template: Optional custom prompt template
        """
        self.llm = llm_instance
        if prompt_template:
            self.analysis_template = ChatPromptTemplate.from_template(prompt_template)

    @instrumental_track_tokens(mode="pre_run", provider="openai")
    def analyze_prompt(self) -> Dict:
        """
        Analyze the prompt and generate structured feedback in a single LLM call.

        Returns:
            Dictionary containing analysis and recommendations.
        """
        # Prepare context for LLM
        context = {"agent_prompt": self.agent_prompt}

        prompt = self.analysis_template.format_messages(**context)
        response = self.llm.invoke(prompt)
        capture_llm_response(response)

        # Try parsing JSON from response
        try:
            result = json.loads(response.content)
        except Exception as e:
            raise ValueError(
                f"Failed to parse LLM response as JSON: {e}\n\nResponse:\n{response.content}"
            )

        return result

    def get_formatted_analysis(self, raw_analysis: Dict = None) -> Dict:
        """
        Format the analysis results or get a new analysis if none provided.

        Args:
            raw_analysis: Optional pre-generated analysis results

        Returns:
            Formatted analysis dictionary
        """
        if raw_analysis is None:
            raw_analysis = self.analyze_prompt()

        # Ensure all expected keys exist
        formatted_analysis = {
            "summary": raw_analysis.get("summary", ""),
            "weaknesses": raw_analysis.get("weaknesses", []),
            "suggested_modifications": raw_analysis.get("suggested_modifications", []),
            "best_practices": raw_analysis.get("best_practices", []),
            "missing_components": raw_analysis.get("missing_components", []),
        }

        return formatted_analysis
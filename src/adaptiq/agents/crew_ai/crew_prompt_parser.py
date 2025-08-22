from typing import Any, Dict, List

from adaptiq.core.abstract.integrations.base_prompt_parser import BasePromptParser
from adaptiq.core.entities import AdaptiQConfig, AgentTool


class CrewPromptParser(BasePromptParser):
    """
    AdaptiqPromptParser analyzes an agent's task description and its declared tools,
    and using an LLM, infers an idealized sequence of steps. Each step is represented
    with the intended subtask, action, preconditions, and expected outcome.

    This implementation uses OpenAI's LLM models and supports YAML configuration files.
    """

    def __init__(
        self, config_data: AdaptiQConfig, task: str, tools: List[AgentTool] = []
    ):
        """
        Initialize the AdaptiqPromptParser with the path to the configuration file.

        Args:
            config_data: The configuration data for the parser.
            task: The task description to be parsed.
            tools: A list of tools available to the agent.
        """
        super().__init__(config_data=config_data, task=task, tools=tools)

    def run_parse_prompt(self) -> List[Dict[str, Any]]:
        """
        Parse the agent's prompt to infer an idealized sequence of steps.

        This method overrides the parent's template method to maintain
        compatibility with the instrumental tracking decorator.

        Returns:
            List of dictionaries, each containing step information with the new keys
        """
        return super().run_parse_prompt()

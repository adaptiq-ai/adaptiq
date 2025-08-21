import logging
import os
from datetime import datetime
from typing import Tuple
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ADAPTIQ-Reconciliation")



class PromptEngineer:
    """
    AI Agent Prompt Engineer that analyzes agent performance and optimizes prompts.

    Uses Q-table insights from reinforcement learning to identify performance patterns
    and leverages LLM analysis to generate improved prompts for better agent behavior.
    Supports configuration-driven setup and automated report generation.
    """

    def __init__(self, 
        llm: BaseChatModel,
        report_path: str,
        old_prompt: str,
        agent_name: str = None,
        feedback: str = None
        ):
        """
        Initialize the PromptEngineerLLM.

        Args:
            main_config_path: Path to the main YAML configuration file.
            feedback: Human feedback for agent's evaluation.
        """
        self.llm = llm
        self.task_name = None
        self.new_prompt = None
        self.feedback = feedback
        self.old_prompt = old_prompt
        self.agent_name = agent_name if agent_name else "agent"
        self.report_path = report_path

        logger.info("PromptEngineerLLM initialized")

    def _invoke_llm_for_analysis(
        self, old_prompt: str, q_table_insights: str
    ) -> Tuple[str, str]:
        """
        Invokes the LLM to get analysis and a new prompt suggestion.

        Returns:
            Tuple (suggested_new_prompt, review_and_diagnostic)
        """
        if not self.llm:
            logger.error("LLM not initialized. Cannot perform analysis.")
            return "Error: LLM not available.", "Error: LLM not available for review."

        system_prompt_content = f"""
        You are an expert AI Agent Prompt Engineer and Performance Diagnostician.
        Your goal is to analyze an agent's current prompt and its recent performance data to provide a diagnostic review and suggest an enhanced prompt.

        You have access to two key sources of performance data:
        1. Q-table insights: Quantitative behavioral patterns showing state-action values and decision-making patterns
        2. Human feedback: Qualitative evaluation of the agent's actual task performance and results

        The new prompt should:
        - Address any observed weaknesses or suboptimal behaviors indicated by both Q-table insights and human feedback
        - Incorporate lessons learned from human evaluations of the agent's actual task outcomes
        - Guide the agent more effectively towards its objective for the task '{self.task_name}'
        - Maintain the original format and core intent of the prompt where appropriate
        - Be clearer, more specific, and provide better guidance based on both quantitative and qualitative performance data
        - If the original prompt has numbered steps or specific output format requirements, the new prompt should try to adhere to similar conventions
        - Prioritize addressing issues highlighted in human feedback, as these represent real-world performance gaps

        When human feedback is available, use it as the primary guide for improvements, with Q-table insights providing supporting behavioral context. When no human feedback is provided, rely primarily on Q-table analysis.
        """

        user_prompt_content = f"""
        Here is the information for your analysis:

        1. Task Key: {self.task_name}

        2. Current Agent Prompt:
        ---
        {old_prompt}
        ---

        3. Q-Table Insights (Observed State-Action Pairs and their Q-values):
        This data shows which actions were taken or learned in various states.
        High Q-values suggest actions believed to be good from those states.
        Low or zero Q-values for available actions might indicate less optimal choices or unexplored paths.
        Frequent transitions to certain states or repeated actions can also be inferred.
        ---
        {q_table_insights}
        ---

        4. Human Feedback on Agent Performance:
        {self.feedback if self.feedback and self.feedback.strip() else "No human feedback provided for this optimization cycle."}
        ---

        Based on all the above information, please provide the following in Markdown format:

        ## Agent Review and Diagnostic
        (Provide your analysis of the agent's likely behavior, strengths, weaknesses, and potential areas for improvement. Consider both the Q-table behavioral patterns and any human feedback about actual task performance. What patterns do you observe? Are there disconnects between what the Q-table suggests the agent learned and what humans observed in the results? How well does the current prompt seem to guide the agent based on both the quantitative behavioral data and qualitative human evaluation?)

        ## Key Issues Identified
        (Summarize the main problems or improvement opportunities identified from:
        - Human feedback (if available): What specific issues did humans identify with the agent's performance?
        - Q-table patterns: What behavioral patterns suggest suboptimal decision-making?
        - Prompt-performance gaps: Where does the current prompt appear insufficient based on the evidence?)

        ## Suggested Enhanced Prompt for Task '{self.task_name}'
        (Provide the full text of the new, improved prompt for the agent. The prompt should directly address the issues identified in human feedback and Q-table analysis. 
        Enclose the prompt itself within a code block for easy copying. 
        The prompt should be directly usable by an agent and incorporate specific improvements based on the performance data.)

        ```
        [Enhanced prompt text here]
        ```

        ## Rationale for Changes
        (Explain the key changes made to the prompt and how they address the identified issues from human feedback and Q-table insights. 
        Connect specific prompt modifications to specific problems observed in the performance data.)
        """
        messages = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_prompt_content),
        ]

        logger.info("Invoking LLM for prompt analysis and suggestion...")
        try:
            response = self.llm.invoke(messages)
            full_response_text = response.content
        except ValueError as e:
            logger.error("ValueError invoking LLM: %s", e)
            return (
                f"Error during LLM invocation: {e}",
                f"Error during LLM invocation for review: {e}",
            )
        except RuntimeError as e:
            logger.error("RuntimeError invoking LLM: %s", e)
            return (
                f"Error during LLM invocation: {e}",
                f"Error during LLM invocation for review: {e}",
            )

        # This is a simple split; more robust parsing might be needed if LLM format varies
        suggested_prompt_header = (
            f"## Suggested Enhanced Prompt for Task '{self.task_name}'"
        )

        parts = full_response_text.split(suggested_prompt_header, 1)

        review_and_diagnostic = (
            parts[0].replace("## Agent Review and Diagnostic", "").strip()
        )

        if len(parts) > 1:
            suggested_new_prompt = parts[1].strip()
            # Often LLMs will put the prompt in a code block, try to extract from it if present
            if "```" in suggested_new_prompt:
                # Attempt to find content within the first markdown code block
                code_block_content = suggested_new_prompt.split("```", 2)
                if len(code_block_content) > 1:  # Found at least one ```
                    # If it's like ```yaml\nPROMPT\n```, take what's between them.
                    # If it's just ```\nPROMPT\n```
                    potential_prompt = code_block_content[1]
                    # Remove potential language specifier like 'yaml' or 'text' from the start of the prompt
                    lines = potential_prompt.split("\n", 1)
                    if (
                        len(lines) > 1
                        and not lines[0].strip().isalnum()
                        and len(lines[0].strip()) > 0
                    ):  # e.g. not 'yaml' but some junk
                        suggested_new_prompt = potential_prompt.strip()
                    elif (
                        len(lines) > 1 and lines[0].strip().isalnum()
                    ):  # e.g. 'yaml' or 'text'
                        suggested_new_prompt = lines[1].strip()
                    else:  # Only one line or first line is empty
                        suggested_new_prompt = potential_prompt.strip()

        else:
            suggested_new_prompt = "LLM did not provide a clearly separated new prompt."
            logger.warning(
                "Could not clearly separate suggested prompt from LLM response."
            )

        self.new_prompt = suggested_new_prompt.strip()
        logger.info("LLM analysis and suggestion received.")
        return suggested_new_prompt, review_and_diagnostic

    def _save_report(self, report_content: str, agent_name_for_report: str):
        """
        Saves the generated report to a file.
        """
        output_path_template = str(self.report_path) if self.report_path else "reports/agent_report.md"

        # Substitute agent name if placeholder is present
        if "your_agent_name" in output_path_template and agent_name_for_report:
            output_path = output_path_template.replace(
                "your_agent_name", agent_name_for_report.replace(" ", "_")
            )
        elif agent_name_for_report:  # If no placeholder, but agent name given, append it to avoid overwrites
            base, ext = os.path.splitext(output_path_template)
            output_path = f"{base}_{agent_name_for_report.replace(' ', '_')}{ext}"
        else:
            output_path = output_path_template

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info("Created report directory: %s", output_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("Report saved to: %s", output_path)

    def generate_and_save_report(self, q_insights: str) -> str:
        """
        Orchestrates the process of loading data, invoking LLM, and saving the report.

        Args:
            q_insights: The insights extracted from the Q-table.
        Returns:
            The generated report content as a string.
        """
        agent_name = self.agent_name
        if agent_name == "your_agent_name":  # Use task_key if agent_name is placeholder
            agent_name = self.task_name

        try:
            old_prompt = self.old_prompt
        except AssertionError as e:
            logger.error("Failed to load old prompt for task %s: %s", self.task_name, e)
            report_content = f"# Prompt Engineering Report for Task: {self.task_name} (Agent: {agent_name})\n\n"
            report_content += f"Date: {datetime.now().isoformat()}\n\n"
            report_content += f"## Error\nFailed to load the original prompt: {e}\n"
            self._save_report(report_content, agent_name)
            return report_content

        suggested_new_prompt, review_diagnostic = self._invoke_llm_for_analysis(
            old_prompt, q_insights
        )

        report_content = f"# Prompt Engineering Report for Task: {self.task_name} (Agent: {agent_name})\n\n"
        report_content += f"Date: {datetime.now().isoformat()}\n\n"
        report_content += "## Agent Review and Diagnostic\n\n"
        report_content += f"{review_diagnostic}\n\n"
        report_content += f"## Original Prompt for Task '{self.task_name}'\n\n"
        report_content += "```text\n"  # Assuming the prompt is plain text
        report_content += f"{old_prompt}\n"
        report_content += "```\n\n"
        report_content += (
            f"## Suggested Enhanced Prompt for Task '{self.task_name}'\n\n"
        )
        report_content += "```text\n"  # Assuming the new prompt is also plain text
        report_content += f"{suggested_new_prompt}\n"
        report_content += "```\n\n"

        self._save_report(report_content, agent_name)
        return report_content
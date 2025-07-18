import ast
import os
import yaml
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from datetime import datetime

from adaptiq.q_learning.q_learning import AdaptiqOfflineLearner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ADAPTIQ-Reconciliation")


class AdaptiqQtablePostrunUpdate:
    """
    Class to update Q-tables based on state classifications and reward executions.
    Uses semantic matching to identify appropriate actions to update.
    """

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        alpha=0.8,
        gamma=0.8,
        similarity_threshold=0.7,
    ):
        """
        Initialize the AdaptiqQtableUpdate class.

        Args:
            api_key: OpenAI API key.
            model: OpenAI embedding model name.
            alpha: Learning rate for Q-learning updates (default 0.8)
            gamma: Discount factor for Q-learning updates (default 0.8)
            similarity_threshold: Threshold for action similarity matching (default 0.7)
        """
        self.learner = AdaptiqOfflineLearner(alpha=alpha, gamma=gamma)
        self.similarity_threshold = similarity_threshold
        self.provider = provider

        if self.provider == "openai":
            self.embeddings = OpenAIEmbeddings(model=model, api_key=api_key)
        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
            )

    def load_q_table(self, q_table_data: Dict) -> None:
        """
        Load the Q-table from a dictionary.

        Args:
            q_table_data: Dictionary containing Q-table data
        """
        self.learner.Q_table = {}
        self.learner.seen_states = set()

        serialized_q = q_table_data.get("Q_table", {})

        for state_str, actions in serialized_q.items():
            try:
                state_key = self._parse_state_key(state_str)
                if state_key is None:  # Should not happen with proper parsing
                    logger.warning(
                        f"Parsed state_key is None for input {state_str}, skipping."
                    )
                    continue

                self.learner.seen_states.add(state_key)

                for action, value in actions.items():
                    self.learner.Q_table[(state_key, action)] = float(value)

            except Exception as e:
                logger.warning(
                    f"Error processing Q-table entry for state string {state_str}: {str(e)}"
                )

        seen_states_list = q_table_data.get("seen_states", [])
        for state_str in seen_states_list:
            try:
                state_key = self._parse_state_key(state_str)
                if state_key is None:
                    logger.warning(
                        f"Parsed state_key for seen_state is None for input {state_str}, skipping."
                    )
                    continue
                self.learner.seen_states.add(state_key)
            except Exception as e:
                logger.warning(
                    f"Error adding seen state from string {state_str}: {str(e)}"
                )

        logger.info(
            f"Loaded Q-table with {len(self.learner.Q_table)} state-action pairs"
        )
        logger.info(f"Loaded {len(self.learner.seen_states)} unique seen states")

    def _parse_state_key(self, state_input: Any) -> Tuple:
        """
        Parse a state representation (string, list, or tuple) into a fully hashable key.
        Converts lists to tuples and dictionaries within structures to sorted tuples of items.

        Args:
            state_input: The state representation.

        Returns:
            A hashable representation of the state (typically a tuple).
        """
        try:
            # Inner function to recursively make elements hashable
            def _make_hashable_recursive(item: Any) -> Any:
                if isinstance(item, list):
                    return tuple(
                        _make_hashable_recursive(sub_item) for sub_item in item
                    )
                elif isinstance(item, tuple):
                    return tuple(
                        _make_hashable_recursive(sub_item) for sub_item in item
                    )
                elif isinstance(item, dict):
                    # Sort by key to ensure consistent hashing for equivalent dicts
                    # Recursively make values hashable
                    return tuple(
                        sorted(
                            (k, _make_hashable_recursive(v)) for k, v in item.items()
                        )
                    )
                # Basic types (int, str, float, bool, NoneType) are already hashable
                return item

            processed_state = state_input
            if isinstance(state_input, str):
                stripped_input = state_input.strip("'\"")
                if (
                    stripped_input.startswith("[") and stripped_input.endswith("]")
                ) or (stripped_input.startswith("(") and stripped_input.endswith(")")):
                    try:
                        processed_state = ast.literal_eval(stripped_input)
                    except (ValueError, SyntaxError) as eval_err:
                        logger.debug(
                            f"ast.literal_eval failed for {stripped_input}: {eval_err}. Treating as plain string."
                        )
                        # If ast.literal_eval fails, it's likely just a string,
                        # or a malformed string. processed_state remains state_input (the string).
                        pass

            # Ensure the (potentially parsed) state is fully hashable
            hashable_state = _make_hashable_recursive(processed_state)

            return hashable_state

        except Exception as e:
            logger.error(
                f"CRITICAL Error parsing state key {state_input!r}: {str(e)}. Falling back to string representation of input."
            )
            # Fallback to string representation of the original input if deep hashing fails
            return str(state_input)

    def _get_actions_for_state(self, state: Any) -> List[str]:
        """
        Get all actions available for a given state in the Q-table.

        Args:
            state: The state to look up (should be in its hashable form).

        Returns:
            List[str]: List of actions associated with the state
        """
        actions = []
        for s, a in self.learner.Q_table.keys():
            # Ensure comparison is between consistently parsed states
            # The `state` arg is already parsed. `s` from Q_table.keys() is also parsed.
            if s == state:  # Direct comparison of hashable types
                actions.append(a)
        return actions

    def _state_equals(self, state1: Any, state2: Any) -> bool:
        """
        Compare two states for equality. Assumes states are already parsed to their
        hashable forms by _parse_state_key.

        Args:
            state1: First state (hashable form)
            state2: Second state (hashable form)

        Returns:
            bool: True if states are equivalent
        """
        # After _parse_state_key, equivalent states should have identical hashable forms.
        # e.g. {'a':1, 'b':2} and {'b':2, 'a':1} both become (('a',1),('b',2))
        return state1 == state2

    def _calculate_action_similarity(
        self, input_action: str, q_table_actions: List[str]
    ) -> Dict[str, float]:
        """
        Calculate similarity between input action and actions in Q-table.

        Args:
            input_action: Action from the input
            q_table_actions: List of actions from the Q-table

        Returns:
            Dict: Dictionary mapping actions to similarity scores
        """
        try:
            if not q_table_actions:  # Handle case with no actions for similarity check
                return {}

            if input_action in q_table_actions:
                return {
                    action: (1.0 if action == input_action else 0.0)
                    for action in q_table_actions
                }

            input_embedding = self.embeddings.embed_query(input_action)
            if input_embedding is None:  # Embedding failed
                logger.error(
                    f"Failed to get embedding for input_action: {input_action}"
                )
                return {action: 0.0 for action in q_table_actions}

            action_embeddings = {}
            for action in q_table_actions:
                emb = self.embeddings.embed_query(action)
                if emb is not None:
                    action_embeddings[action] = emb
                else:
                    logger.warning(
                        f"Failed to get embedding for Q-table action: {action}"
                    )

            if not action_embeddings:  # All Q-table action embeddings failed
                return {action: 0.0 for action in q_table_actions}

            similarities = {}
            input_norm = np.linalg.norm(input_embedding)
            if input_norm == 0:  # Avoid division by zero
                logger.warning(
                    f"Input action embedding norm is zero for: {input_action}"
                )
                return {action: 0.0 for action in q_table_actions}

            for action, embedding in action_embeddings.items():
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm == 0:  # Avoid division by zero
                    similarities[action] = 0.0
                    logger.warning(
                        f"Q-table action embedding norm is zero for: {action}"
                    )
                    continue

                similarity = np.dot(input_embedding, embedding) / (
                    input_norm * embedding_norm
                )
                similarities[action] = float(similarity)

            return similarities
        except Exception as e:
            logger.error(
                f"Error calculating action similarity for input '{input_action}' against {q_table_actions}: {str(e)}"
            )
            return {
                action: 1.0 if action == input_action else 0.0
                for action in q_table_actions
            }

    def update_q_table(
        self, state_classifications: List[Dict], reward_execs: List[Dict]
    ) -> Dict:
        """
        Update the Q-table based on state classifications and reward executions.

        Args:
            state_classifications: List of state classification dictionaries
            reward_execs: List of reward execution dictionaries

        Returns:
            Dict: Updated Q-table data
        """
        for i, classification in enumerate(state_classifications):
            if i >= len(reward_execs):
                logger.warning(
                    f"No reward execution data for classification at index {i}"
                )
                continue

            reward_exec = reward_execs[i]

            if classification["classification"]["is_known_state"]:
                matched_state_repr = classification["classification"]["matched_state"]
                # Ensure matched_state is in the canonical hashable form
                matched_state = self._parse_state_key(matched_state_repr)

                input_action = classification["input_state"]["action"]
                reward = reward_exec["reward_exec"]

                available_actions = self._get_actions_for_state(matched_state)

                if not available_actions:
                    # This case implies the state might be in seen_states but has no actions yet,
                    # or it's a truly new state not even in seen_states.
                    # _parse_state_key makes it hashable, so it can be added.
                    if matched_state not in self.learner.seen_states:
                        logger.info(f"Adding new state {matched_state} to seen_states.")
                        self.learner.seen_states.add(matched_state)

                    logger.info(
                        f"State {matched_state} has no actions in Q-table. Adding new action {input_action}."
                    )
                    self.learner.Q_table[(matched_state, input_action)] = (
                        reward  # Initial Q-value set to reward
                    )
                else:
                    if input_action in available_actions:
                        logger.info(
                            f"Exact action match. Updating Q-value for state {matched_state}, action {input_action}"
                        )
                        next_state_parsed = None
                        next_actions_for_next_state = []
                        if (
                            i + 1 < len(state_classifications)
                            and state_classifications[i + 1]["classification"][
                                "is_known_state"
                            ]
                        ):
                            next_state_repr = state_classifications[i + 1][
                                "classification"
                            ]["matched_state"]
                            next_state_parsed = self._parse_state_key(next_state_repr)
                            next_actions_for_next_state = self._get_actions_for_state(
                                next_state_parsed
                            )

                        self.learner.update_policy(
                            matched_state,
                            input_action,
                            reward,
                            next_state_parsed,
                            next_actions_for_next_state,
                        )
                    else:  # Action not found, use similarity
                        similarities = self._calculate_action_similarity(
                            input_action, available_actions
                        )
                        if (
                            not similarities
                        ):  # No similar actions could be computed or no actions available
                            logger.info(
                                f"No similar actions found or could be computed for {input_action} in state {matched_state}. Adding as new action."
                            )
                            self.learner.Q_table[(matched_state, input_action)] = reward
                            continue  # Skip to next classification

                        most_similar_action_tuple = max(
                            similarities.items(), key=lambda x: x[1]
                        )

                        if most_similar_action_tuple[1] >= self.similarity_threshold:
                            action_to_update = most_similar_action_tuple[0]
                            logger.info(
                                f"Similar action found: '{action_to_update}' (sim: {most_similar_action_tuple[1]:.2f}) for input '{input_action}'. Updating Q-value for state {matched_state}."
                            )

                            next_state_parsed = None
                            next_actions_for_next_state = []
                            if (
                                i + 1 < len(state_classifications)
                                and state_classifications[i + 1]["classification"][
                                    "is_known_state"
                                ]
                            ):
                                next_state_repr = state_classifications[i + 1][
                                    "classification"
                                ]["matched_state"]
                                next_state_parsed = self._parse_state_key(
                                    next_state_repr
                                )
                                next_actions_for_next_state = (
                                    self._get_actions_for_state(next_state_parsed)
                                )

                            self.learner.update_policy(
                                matched_state,
                                action_to_update,  # Update the similar action
                                reward,
                                next_state_parsed,
                                next_actions_for_next_state,
                            )
                        else:
                            logger.info(
                                f"No action similar enough (max sim: {most_similar_action_tuple[1]:.2f} < {self.similarity_threshold}) for '{input_action}'. Adding as new action to state {matched_state}."
                            )
                            self.learner.Q_table[(matched_state, input_action)] = reward
            else:
                logger.info(
                    f"Skipping unknown state at index {i}: {classification['input_state']}"
                )

        output_q_table = {}
        for (state, action), value in self.learner.Q_table.items():
            state_key_str = str(
                state
            )  # Convert hashable state back to string for JSON output
            if state_key_str not in output_q_table:
                output_q_table[state_key_str] = {}
            output_q_table[state_key_str][action] = value

        seen_states_output = [str(s) for s in self.learner.seen_states]

        result = {
            "Q_table": output_q_table,
            "seen_states": seen_states_output,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0_updated",  # Indicate change
        }
        return result

    def process_data(
        self,
        state_classifications_data: List[Dict],
        reward_execs_data: List[Dict],
        q_table_data: Dict,
    ) -> Dict:
        """
        Process input data and update the Q-table.

        Args:
            state_classifications_data: List of state classification dictionaries
            reward_execs_data: List of reward execution dictionaries
            q_table_data: Dictionary containing Q-table data

        Returns:
            Dict: Updated Q-table data
        """
        self.load_q_table(q_table_data)
        updated_q_table = self.update_q_table(
            state_classifications_data, reward_execs_data
        )
        self.learner.save_q_table(
            file_path="adaptiq_q_table.json", prefix_version="post_run"
        )
        return updated_q_table


class AdaptiqPromptEngineer:
    """
    AI Agent Prompt Engineer that analyzes agent performance and optimizes prompts.

    Uses Q-table insights from reinforcement learning to identify performance patterns
    and leverages LLM analysis to generate improved prompts for better agent behavior.
    Supports configuration-driven setup and automated report generation.
    """

    def __init__(self, main_config_path: str, feedback: str = None):
        """
        Initialize the PromptEngineerLLM.

        Args:
            main_config_path: Path to the main YAML configuration file.
            feedback: Human feedback for agent's evaluation.
        """
        self.task_name = None
        self.new_prompt = None
        self.feedback = feedback

        if not os.path.exists(main_config_path):
            logger.error(f"Main configuration file not found: {main_config_path}")
            raise FileNotFoundError(
                f"Main configuration file not found: {main_config_path}"
            )

        with open(main_config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        llm_conf = self.config.get("llm_config", {})
        self.model_name = llm_conf.get(
            "model_name", "gpt-4.1-mini"
        )  # Default if not specified
        self.api_key = llm_conf.get("api_key")
        self.provider = llm_conf.get("provider")

        if not self.api_key:
            logger.warning("API key for LLM is not set in the configuration.")
            # Allow initialization for testing without LLM, but warn.
            # Or raise ValueError("API key for LLM must be provided.")

        self.agent_modifiable_config = self.config.get("agent_modifiable_config", {})
        self.report_config = self.config.get("report_config", {})

        # Initialize LLM, handle potential missing API key for non-LLM parts
        try:
            if self.provider == "openai":
                self.llm = ChatOpenAI(
                    model=self.model_name, api_key=self.api_key, temperature=0.3
                )
            else:
                raise ValueError(
                    f"Unsupported provider: {self.provider}. Only 'openai' is currently supported."
                )

        except Exception as e:
            logger.error(
                f"Failed to initialize ChatOpenAI: {e}. Ensure API key is valid."
            )
            self.llm = None  # Allow class to function for non-LLM tasks if needed

        logger.info(f"PromptEngineerLLM initialized with model: {self.model_name}")

    def _load_old_prompt(self) -> str:
        """
        Loads the old agent prompt by identifying the task key in the YAML file.
        Sets self.task_name with the detected task key.
        """
        prompt_file_path_str = self.agent_modifiable_config.get(
            "prompt_configuration_file_path"
        )
        if not prompt_file_path_str:
            logger.error("Path to prompt_configuration_file_path not found in config.")
            raise ValueError("prompt_configuration_file_path is not configured.")

        if not os.path.isabs(prompt_file_path_str):
            # Optional: Resolve relative paths against a known base directory if needed
            pass

        if not os.path.exists(prompt_file_path_str):
            logger.error(
                f"Agent prompt configuration file not found: {prompt_file_path_str}"
            )
            raise FileNotFoundError(
                f"Agent prompt configuration file not found: {prompt_file_path_str}"
            )

        with open(prompt_file_path_str, "r", encoding="utf-8") as f:
            prompts_config = yaml.safe_load(f)

        # Find the task with a "description" field
        for key, config in prompts_config.items():
            if isinstance(config, dict) and "description" in config:
                self.task_name = key
                old_prompt = config["description"]
                logger.info(
                    f"Successfully loaded old prompt for task: {self.task_name}"
                )
                return old_prompt.strip()

        logger.error(
            "No task with a 'description' field found in prompt configuration file."
        )
        raise ValueError(
            "No task with a 'description' field found in prompt configuration file."
        )

    def _extract_q_table_insights(self, q_table_output: Dict) -> str:
        """
        Extracts insights from the Q-table output.
        Focuses on states and actions with non-zero Q-values.
        """
        q_table = q_table_output.get("Q_table", {})
        if not q_table:
            return "No Q-table data available to analyze."

        insights = ["Q-Table Insights (States and Actions with Non-Zero Q-values):\n"]
        for state_str, actions_dict in q_table.items():
            if not actions_dict:
                continue

            # Filter actions with non-zero Q-values
            non_zero_actions = {a: q for a, q in actions_dict.items() if q != 0.0}
            if not non_zero_actions:
                continue

            insights.append(f"  State: {state_str}")
            for action, q_value in non_zero_actions.items():
                insights.append(f"    - Action: {action}, Q-Value: {q_value:.4f}")
            insights.append("")  # Newline for readability

        if len(insights) == 1:  # Only the header was added
            return "Q-table exists but contains no states with non-zero Q-values."

        logger.info("Extracted insights from Q-table.")
        return "\n".join(insights)

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
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
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
        output_path_template = self.report_config.get(
            "output_path", "reports/agent_report.md"
        )

        # Substitute agent name if placeholder is present
        if "your_agent_name" in output_path_template and agent_name_for_report:
            output_path = output_path_template.replace(
                "your_agent_name", agent_name_for_report.replace(" ", "_")
            )
        elif (
            agent_name_for_report
        ):  # If no placeholder, but agent name given, append it to avoid overwrites
            base, ext = os.path.splitext(output_path_template)
            output_path = f"{base}_{agent_name_for_report.replace(' ', '_')}{ext}"
        else:
            output_path = output_path_template

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created report directory: {output_dir}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info(f"Report saved to: {output_path}")

    def generate_and_save_report(self, q_table_output: Dict) -> str:
        """
        Orchestrates the process of loading data, invoking LLM, and saving the report.

        Args:
            q_table_output: The output dictionary from AdaptiqQtablePostrunUpdate.
        Returns:
            The generated report content as a string.
        """
        agent_name = self.agent_modifiable_config.get("agent_name", "unknown_agent")
        if agent_name == "your_agent_name":  # Use task_key if agent_name is placeholder
            agent_name = self.task_name

        try:
            old_prompt = self._load_old_prompt()
        except Exception as e:
            logger.error(f"Failed to load old prompt for task {self.task_name}: {e}")
            report_content = f"# Prompt Engineering Report for Task: {self.task_name} (Agent: {agent_name})\n\n"
            report_content += f"Date: {datetime.now().isoformat()}\n\n"
            report_content += f"## Error\nFailed to load the original prompt: {e}\n"
            self._save_report(report_content, agent_name)
            return report_content

        q_insights = self._extract_q_table_insights(q_table_output)

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

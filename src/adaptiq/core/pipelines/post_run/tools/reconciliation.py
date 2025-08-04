import ast
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
from langchain_openai import OpenAIEmbeddings

from adaptiq.core.q_table.q_table_manager import QTableManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ADAPTIQ-Reconciliation")


class Reconciliation:
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

        self.learner = QTableManager(alpha=alpha, gamma=gamma, file_path="adaptiq_q_table.json")

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

        self.learner.save_q_table(prefix_version="post_run")
        return updated_q_table

import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings

from adaptiq.core.entities.adaptiq_parsers import ClassificationEntry, LogItem
from adaptiq.core.entities.q_table import QTableAction, QTableQValue, QTableState
from adaptiq.core.q_table.q_table_manager import QTableManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ADAPTIQ-PostRunUpdater")


class PostRunUpdater:
    """
    Class to update Q-tables based on state classifications and reward executions.
    Uses semantic matching to identify appropriate actions to update.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        output_path: Path ,
        alpha: float = 0.8,
        gamma: float = 0.8,
        similarity_threshold: float = 0.7,
        
    ):
        """
        Initialize the AdaptiqQtableUpdate class.

        Args:
            embeddings: Embeddings instance for text embeddings
            alpha: Learning rate for Q-table updates
            gamma: Discount factor for future rewards
            similarity_threshold: Threshold for action similarity matching
        """
        self.embeddings = embeddings
        self.learner = QTableManager(
            alpha=alpha, gamma=gamma, file_path=output_path
        )
        self.similarity_threshold = similarity_threshold

    def load_q_table(self, q_table_data: Dict):
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
                # Parse state string back to QTableState object
                state_obj = self._parse_state_to_qtable_state(state_str)
                if state_obj is None:
                    logger.warning("Failed to parse state_str %s, skipping.", state_str)
                    continue

                self.learner.seen_states.add(state_obj)

                # Initialize state in Q_table if not exists
                if state_obj not in self.learner.Q_table:
                    self.learner.Q_table[state_obj] = {}

                for action_str, q_value_data in actions.items():
                    action_obj = QTableAction.from_str(action_str)
                    # Handle both direct float values and QTableQValue objects
                    if isinstance(q_value_data, dict):
                        q_value_obj = QTableQValue(**q_value_data)
                    else:
                        # FIX (Latent Bug): Use keyword argument for QTableQValue as well
                        q_value_obj = QTableQValue(q_value=float(q_value_data))

                    self.learner.Q_table[state_obj][action_obj] = q_value_obj

            except Exception as e:
                logger.warning(
                    "Error processing Q-table entry for state string %s: %s",
                    state_str,
                    str(e),
                )

        seen_states_list = q_table_data.get("seen_states", [])
        for state_str in seen_states_list:
            try:
                state_obj = self._parse_state_to_qtable_state(state_str)
                if state_obj is None:
                    logger.warning(f"Failed to parse seen_state {state_str}, skipping.")
                    continue
                self.learner.seen_states.add(state_obj)
            except Exception as e:
                logger.warning(
                    f"Error adding seen state from string {state_str}: {str(e)}"
                )

        logger.info(f"Loaded Q-table with {len(self.learner.Q_table)} states")
        logger.info(f"Loaded {len(self.learner.seen_states)} unique seen states")

    def _parse_state_to_qtable_state(self, state_str: str) -> QTableState:
        """
        Parse a state string back to QTableState object.
        Expects format from QTableState.to_tuple() joined with separator.

        Args:
            state_str: String representation of state

        Returns:
            QTableState object or None if parsing fails
        """
        try:
            # Try to split by the separator used in QTableManager.serialize_q_table
            # Looking at the serialize method, it uses "".join(state.to_tuple())
            # But to_tuple returns 4 strings, so we need to figure out how to split them back

            # Since the original serialize method uses "".join() without separator,
            # we need a different approach. Let's try to parse it as a tuple first
            if state_str.startswith("(") and state_str.endswith(")"):
                # Try to evaluate as tuple
                state_tuple = ast.literal_eval(state_str)
                if len(state_tuple) == 4:
                    return QTableState.from_tuple(state_tuple)

            # If it's a direct concatenation, we need to assume a separator
            # Let's try splitting by "||" which is commonly used
            if "||" in state_str:
                parts = state_str.split("||")
                if len(parts) == 4:
                    return QTableState.from_tuple(tuple(parts))

            # If no separator, try to parse as literal
            try:
                parsed = ast.literal_eval(state_str)
                if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                    return QTableState.from_tuple(tuple(parsed))
            except (ValueError, SyntaxError):
                pass

            logger.warning(f"Could not parse state string: {state_str}")
            return None

        except Exception as e:
            logger.error(f"Error parsing state string {state_str}: {str(e)}")
            return None

    def _state_equals(self, state1: QTableState, state2: QTableState) -> bool:
        """
        Compare two QTableState objects for equality.

        Args:
            state1: First state
            state2: Second state

        Returns:
            bool: True if states are equivalent
        """
        return (
            state1.current_subtask == state2.current_subtask
            and state1.last_action_taken == state2.last_action_taken
            and state1.last_outcome == state2.last_outcome
            and state1.key_context == state2.key_context
        )

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
        self,
        state_classifications: List[ClassificationEntry],
        reward_execs: List[LogItem],
    ):
        """
        Update the Q-table based on state classifications and reward executions.

        Args:
            state_classifications: List of state classification dictionaries
            reward_execs: List of reward execution dictionaries

        """
        for i, classification in enumerate(state_classifications):
            if i >= len(reward_execs):
                logger.warning(
                    f"No reward execution data for classification at index {i}"
                )
                continue

            reward_exec = reward_execs[i]

            if classification.classification.is_known_state:
                matched_state_repr = classification.classification.matched_state
                # Parse the matched state representation to QTableState
                matched_state = self._parse_state_to_qtable_state(matched_state_repr)

                if matched_state is None:
                    logger.warning(
                        f"Could not parse matched state: {matched_state_repr}"
                    )
                    continue

                input_action = classification.input_state.action
                reward = reward_exec.reward_exec

                available_actions = self.learner.get_actions_for_state(matched_state)

                if not available_actions:
                    # Add state to seen_states and create new action
                    if matched_state not in self.learner.seen_states:
                        logger.info(f"Adding new state {matched_state} to seen_states.")
                        self.learner.seen_states.add(matched_state)

                    logger.info(
                        f"State {matched_state} has no actions in Q-table. Adding new action {input_action}."
                    )
                    # FIX: Use keyword argument `action=`
                    self.learner.set_q_value(
                        matched_state, QTableAction(action=input_action), reward
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
                            and state_classifications[
                                i + 1
                            ].classification.is_known_state
                        ):
                            next_state_repr = state_classifications[
                                i + 1
                            ].classification.matched_state

                            next_state_parsed = self._parse_state_to_qtable_state(
                                next_state_repr
                            )
                            if next_state_parsed:
                                next_actions_for_next_state = (
                                    self.learner.get_actions_for_state(
                                        next_state_parsed
                                    )
                                )

                        self.learner.update_policy(
                            s=matched_state,
                            # FIX: Use keyword argument `action=`
                            a=QTableAction(action=input_action),
                            R=reward,
                            s_prime=next_state_parsed,
                            actions_prime=[
                                # FIX: Use keyword argument `action=` in list comprehension
                                QTableAction(action=a)
                                for a in next_actions_for_next_state
                            ],
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
                            # FIX: Use keyword argument `action=`
                            self.learner.set_q_value(
                                matched_state, QTableAction(action=input_action), reward
                            )
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
                                and state_classifications[
                                    i + 1
                                ].classification.is_known_state
                            ):
                                next_state_repr = state_classifications[
                                    i + 1
                                ].classification.matched_state
                                next_state_parsed = self._parse_state_to_qtable_state(
                                    next_state_repr
                                )
                                if next_state_parsed:
                                    next_actions_for_next_state = (
                                        self.learner.get_actions_for_state(
                                            next_state_parsed
                                        )
                                    )

                            self.learner.update_policy(
                                s=matched_state,
                                # FIX: Use keyword argument `action=`
                                a=QTableAction(action=action_to_update),
                                R=reward,
                                s_prime=next_state_parsed,
                                actions_prime=[
                                    # FIX: Use keyword argument `action=` in list comprehension
                                    QTableAction(action=a)
                                    for a in next_actions_for_next_state
                                ],
                            )
                        else:
                            logger.info(
                                f"No action similar enough (max sim: {most_similar_action_tuple[1]:.2f} < {self.similarity_threshold}) for '{input_action}'. Adding as new action to state {matched_state}."
                            )
                            # FIX: Use keyword argument `action=`
                            self.learner.set_q_value(
                                matched_state, QTableAction(action=input_action), reward
                            )
            # else:
            #     logger.info(
            #         f"Skipping unknown state at index {i}: {classification.input_state.model_dump()}"
            #     )

    def process_data(
        self,
        state_classifications_data: List[ClassificationEntry],
        reward_execs_data: List[LogItem],
        q_table_data: Dict,
    ) -> Tuple[Dict, str]:
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
        self.update_q_table(state_classifications_data, reward_execs_data)
        self.learner.save_q_table(prefix_version="post_run")

        return (
            self.learner.get_q_table_dict(),
            self.learner.extract_q_table_insights_to_str(),
        )

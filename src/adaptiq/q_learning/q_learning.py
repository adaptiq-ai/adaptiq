#!/usr/bin/env python3
"""
ADAPTIQ Offline Learning Loop Implementation

This script implements the ADAPTIQ (Agent Reinforcement through Iterative Configuration) methodology
for offline analysis of agent execution traces to improve an agent's prompt/configuration.
"""

import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ADAPTIQ-Qtable")


class AdaptiqOfflineLearner:
    """
    ADAPTIQ Offline Learner - Analyzes agent execution traces and learns to improve agent configurations
    based on Q-learning principles.
    """

    def __init__(self, alpha=0.8, gamma=0.8):
        """
        Initialize the ADAPTIQ Offline Learner.

        Args:
            config_path: Path to the ADAPTIQ configuration YAML file
            alpha: Learning rate (default 0.8)
            gamma: Discount factor (default 0.8)
        """
        # Core Q-learning components
        self.Q_table = {}
        self.seen_states = set()
        self.alpha = alpha
        self.gamma = gamma

        # Load environment variables for API access
        load_dotenv()

    def Q(self, s: Any, a: str) -> float:
        """
        Get the current Q-value for a state-action pair

        Args:
            s: State key (any hashable representation)
            a: Action

        Returns:
            float: Q-value for the state-action pair
        """
        if (s, a) not in self.Q_table:
            # Initialize with fixed value 0.0 to avoid random bias
            self.Q_table[(s, a)] = 0.0
        return self.Q_table[(s, a)]

    def update_policy(
        self,
        s: Tuple[str, str, str, str],
        a: str,
        R: float,
        s_prime: Any,
        actions_prime: List[str],
    ) -> float:
        """
        Update policy function with Q-learning formula

        Args:
            s: Current state key
            a: Action taken
            R: Reward received
            s_prime: Next state key
            actions_prime: List of possible actions from next state

        Returns:
            float: New Q-value
        """
        # Get current Q-value
        Q_sa = self.Q(s, a)

        # Calculate max Q-value among possible actions from NEXT state
        max_Q_s_prime = (
            max([self.Q(s_prime, a_prime) for a_prime in actions_prime])
            if actions_prime
            else 0
        )

        # Standard Q-learning update formula
        new_Q_sa = Q_sa + self.alpha * (R + self.gamma * max_Q_s_prime - Q_sa)

        # Update Q-table
        self.Q_table[(s, a)] = new_Q_sa

        # Add state to seen states
        self.seen_states.add(s)

        # Log the update
        logger.debug(f"Q-update: {Q_sa:.4f} → {new_Q_sa:.4f}")
        logger.debug(
            f"Formula: {Q_sa:.4f} + {self.alpha:.2f} * ({R:.2f} + {self.gamma:.2f} * {max_Q_s_prime:.4f} - {Q_sa:.4f})"
        )

        return new_Q_sa

    def save_q_table(self, file_path: str, prefix_version: str = None) -> bool:
        """
        Save the Q-table to a JSON file

        Args:
            file_path: Path where Q-table should be saved
            prefix_version: The version ID of the current Qtable

        Returns:
            bool: Success flag
        """
        try:
            # Convert Q-table to nested dict format
            nested_q = {}

            for (state, action), value in self.Q_table.items():
                # Convert state to consistent string representation
                if isinstance(state, tuple):
                    state_key = str(state)
                elif isinstance(state, list):
                    state_key = str(tuple(state))
                else:
                    state_key = str(state)

                if state_key not in nested_q:
                    nested_q[state_key] = {}

                nested_q[state_key][action] = value

            # Ensure consistent state representation in seen_states
            normalized_seen_states = []
            for state in self.seen_states:
                if isinstance(state, tuple):
                    normalized_seen_states.append(str(state))
                elif isinstance(state, list):
                    normalized_seen_states.append(str(tuple(state)))
                else:
                    normalized_seen_states.append(str(state))

            # Save metadata
            data = {
                "Q_table": nested_q,
                "seen_states": normalized_seen_states,
                "timestamp": datetime.now().isoformat(),
                "version": (
                    prefix_version + "_" + str(uuid.uuid4()).replace("-", "")[:5]
                    if prefix_version
                    else "1.0"
                ),
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Successfully saved Q-table to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving Q-table: {str(e)}")
            return False

    def load_q_table(self, file_path: str) -> bool:
        """
        Load the Q-table from a JSON file

        Args:
            file_path: Path to the Q-table file

        Returns:
            bool: Success flag
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Reset current Q-table
            self.Q_table = {}
            self.seen_states = set()

            # Load Q-table (convert from serialized format)
            serialized_q = data.get("Q_table", {})

            for state_str, actions in serialized_q.items():
                # Improved state key deserialization
                try:
                    state_key = None

                    # Handle tuple format like "(1, 2, 3)" or "('a', 'b', 'c')"
                    if state_str.startswith("(") and state_str.endswith(")"):
                        # Parse tuple elements with proper type handling
                        elements = []
                        raw_elements = state_str.strip("()").split(", ")

                        for element in raw_elements:
                            if (element.startswith("'") and element.endswith("'")) or (
                                element.startswith('"') and element.endswith('"')
                            ):
                                # String element
                                elements.append(element.strip("'\""))
                            elif element.lower() == "true":
                                elements.append(True)
                            elif element.lower() == "false":
                                elements.append(False)
                            elif element.lower() == "none" or element.lower() == "null":
                                elements.append(None)
                            else:
                                # Try numeric conversion
                                try:
                                    if "." in element:
                                        elements.append(float(element))
                                    else:
                                        elements.append(int(element))
                                except ValueError:
                                    # Keep as string if not convertible
                                    elements.append(element)

                        state_key = tuple(elements)
                    else:
                        # Keep as string if not recognizable format
                        state_key = state_str

                    # Add to seen states
                    self.seen_states.add(state_key)

                    # Load action values
                    for action, value in actions.items():
                        self.Q_table[(state_key, action)] = float(value)

                except Exception as e:
                    logger.warning(
                        f"Skipping state due to parsing error: {state_str}, Error: {str(e)}"
                    )
                    continue

            logger.info(f"Successfully loaded Q-table from {file_path}")
            logger.info(f"Loaded {len(self.Q_table)} state-action pairs")
            logger.info(f"Loaded {len(self.seen_states)} unique states")
            return True

        except FileNotFoundError:
            logger.warning(f"Q-table file not found: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading Q-table: {str(e)}")
            return False

    def get_q_table(self) -> Dict:
        """
        Get a copy of the current Q-table

        Returns:
            dict: The Q-table
        """
        return dict(self.Q_table)

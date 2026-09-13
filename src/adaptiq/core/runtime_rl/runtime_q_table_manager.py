#!/usr/bin/env python3
"""
Runtime Q-Table Manager - Online learning variant with epsilon-greedy exploration

This module extends the offline QTableManager to support online learning during
agent execution with exploration strategies.

Key Differences from Offline QTableManager:
- Lower alpha (0.1 vs 0.8) for stable online learning
- Higher gamma (0.9 vs 0.8) for long-term planning
- Epsilon-greedy action selection for exploration
- Separate storage path to avoid conflicts
"""

import logging
import random
from typing import List, Tuple

from adaptiq.core.entities.q_table import QTableAction, QTableState
from adaptiq.core.q_table.q_table_manager import QTableManager

# Configure logging
logger = logging.getLogger("ADAPTIQ-RuntimeRL")


class RuntimeQTableManager(QTableManager):
    """
    Runtime RL variant of QTableManager with epsilon-greedy exploration.

    Inherits all Q-Learning functionality from QTableManager:
    - update_policy() with Bellman equation (line 67 of q_table_manager.py)
    - save_q_table() and load_q_table() for persistence
    - Q() getter for Q-values
    - All state management methods

    Adds:
    - Epsilon-greedy action selection
    - Online learning hyperparameters
    - Separate storage path

    Example:
        >>> manager = RuntimeQTableManager()
        >>> state = QTableState(current_subtask="estimate_price", ...)
        >>> actions = [QTableAction(action="method_a"), QTableAction(action="method_b")]
        >>> action, explored = manager.select_action_epsilon_greedy(state, actions)
        >>> # Execute action, get reward
        >>> next_state = QTableState(...)
        >>> manager.update_policy(state, action, reward, next_state, actions)
        >>> manager.save_q_table(prefix_version="runtime")
    """

    def __init__(
        self,
        file_path: str = "storage/qtables/runtime_q_table.json",
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        """
        Initialize Runtime Q-Table Manager.

        Args:
            file_path: Path to save/load Q-table (default: runtime_q_table.json)
            alpha: Learning rate (0.1 for stable online learning vs 0.8 offline)
            gamma: Discount factor (0.9 for long-term vs 0.8 offline)
            epsilon: Exploration rate (0.1 = 10% exploration, 90% exploitation)

        Note:
            - alpha lower than offline (0.8) for gradual online updates
            - gamma higher than offline (0.8) to value long-term rewards
            - Storage path separate from offline Q-table to avoid conflicts
        """
        # Call parent constructor with runtime hyperparameters
        super().__init__(file_path=file_path, alpha=alpha, gamma=gamma)

        # Validate epsilon
        if not 0 <= epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")

        self.epsilon = epsilon

        logger.info(
            f"Runtime Q-Table Manager initialized: "
            f"alpha={self.alpha}, gamma={self.gamma}, epsilon={self.epsilon}"
        )
        logger.info(f"Storage path: {self.file_path}")

    def select_action_epsilon_greedy(
        self, state: QTableState, available_actions: List[QTableAction]
    ) -> Tuple[QTableAction, bool]:
        """
        Select action using epsilon-greedy strategy.

        With probability epsilon, select a random action (exploration).
        With probability (1-epsilon), select the best action (exploitation).

        Args:
            state: Current state
            available_actions: List of actions available in this state

        Returns:
            Tuple of (selected_action, exploration_used)
                - selected_action: The chosen action
                - exploration_used: True if exploration was used, False if exploitation

        Raises:
            ValueError: If no available actions provided

        Example:
            >>> state = QTableState(current_subtask="task", ...)
            >>> actions = [QTableAction(action="a"), QTableAction(action="b")]
            >>> action, explored = manager.select_action_epsilon_greedy(state, actions)
            >>> if explored:
            ...     print("Random exploration")
            ... else:
            ...     print("Greedy exploitation")
        """
        if not available_actions:
            raise ValueError("No available actions provided")

        # Exploration: random action with probability epsilon
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
            logger.debug(
                f"EXPLORATION: Randomly selected action '{action.action}' "
                f"(epsilon={self.epsilon})"
            )
            return action, True

        # Exploitation: best action based on Q-values (inherited method)
        action = self.get_best_action(state, available_actions)
        logger.debug(
            f"EXPLOITATION: Selected best action '{action.action}' "
            f"(Q-value={self.Q(state, action):.4f})"
        )
        return action, False

    def set_epsilon(self, new_epsilon: float) -> None:
        """
        Update epsilon (e.g., for epsilon decay strategies).

        Args:
            new_epsilon: New exploration rate in [0, 1]

        Raises:
            ValueError: If epsilon not in [0, 1]

        Example:
            >>> manager = RuntimeQTableManager(epsilon=0.1)
            >>> # Decay epsilon over time
            >>> manager.set_epsilon(0.05)
        """
        if not 0 <= new_epsilon <= 1:
            raise ValueError(f"Epsilon must be in [0, 1], got {new_epsilon}")

        old_epsilon = self.epsilon
        self.epsilon = new_epsilon
        logger.info(f"Epsilon updated: {old_epsilon:.4f} → {new_epsilon:.4f}")

    def get_epsilon(self) -> float:
        """
        Get current epsilon value.

        Returns:
            Current exploration rate
        """
        return self.epsilon

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"RuntimeQTableManager("
            f"alpha={self.alpha}, "
            f"gamma={self.gamma}, "
            f"epsilon={self.epsilon}, "
            f"states={len(self.Q_table)}, "
            f"storage='{self.file_path}'"
            f")"
        )

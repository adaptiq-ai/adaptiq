#!/usr/bin/env python3
"""
ADAPTIQ Offline Learning Loop Implementation

This script implements the ADAPTIQ (Agent Reinforcement through Iterative Configuration) methodology
for offline analysis of agent execution traces to improve an agent's prompt/configuration.
"""

import logging

import warnings
from typing import Any, List, Tuple

from adaptiq.core.abstract.q_table.base_q_table_manager import BaseQTableManager

warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ADAPTIQ-Qtable")



class QTableManager(BaseQTableManager):
    """
    ADAPTIQ Offline Learner - Analyzes agent execution traces and learns to improve agent configurations
    based on Q-learning principles.
    """

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


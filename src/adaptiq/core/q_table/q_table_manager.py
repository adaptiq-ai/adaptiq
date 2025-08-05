#!/usr/bin/env python3
"""
ADAPTIQ Offline Learning Loop Implementation

This script implements the ADAPTIQ (Agent Reinforcement through Iterative Configuration) methodology
for offline analysis of agent execution traces to improve an agent's prompt/configuration.
"""

import logging
import warnings
from typing import List, Tuple

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
        agent_id: str,
        s: Tuple[str, str, str, str],
        a: str,
        R: float,
        s_prime: Tuple[str, str, str, str],
        actions_prime: List[str],
    ) -> float:
        """
        Update policy function with Q-learning formula

        Args:
            agent_id: Identifier for the Q-table owner
            s: Current state key (tuple)
            a: Action taken
            R: Reward received
            s_prime: Next state key
            actions_prime: List of possible actions from next state

        Returns:
            float: Updated Q-value for (s, a)
        """
        # Get Q(s, a) or default to 0.0
        Q_sa = self.Q_table.get((s, a), 0.0)

        # Calculate max Q-value for next state
        max_Q_s_prime = 0.0
        if actions_prime:
            q_values = [self.Q_table.get((s_prime, a_prime), 0.0) for a_prime in actions_prime]
            max_Q_s_prime = max(q_values) if q_values else 0.0

        # Q-learning update
        new_Q_sa = Q_sa + self.alpha * (R + self.gamma * max_Q_s_prime - Q_sa)

        # Update the Q-table
        self.Q_table[(s, a)] = new_Q_sa

        # Mark state as seen
        self.seen_states.add(s)

        # Log
        logger.debug(f"[{agent_id}] Q-update: {Q_sa:.4f} → {new_Q_sa:.4f}")
        logger.debug(
            f"Formula: {Q_sa:.4f} + {self.alpha:.2f} * ({R:.2f} + {self.gamma:.2f} * {max_Q_s_prime:.4f} - {Q_sa:.4f})"
        )

        return new_Q_sa
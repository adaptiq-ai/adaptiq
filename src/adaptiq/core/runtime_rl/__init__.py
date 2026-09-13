"""
Runtime RL - Online Decision Making for AdaptIQ

This module provides runtime reinforcement learning capabilities for making
online decisions during agent execution.

Components:
- RuntimeQTableManager: Q-Learning with epsilon-greedy exploration
- RuntimeDecisionEngine: Orchestrates runtime decisions
- RuntimeRewardCalculator: Domain-specific reward calculation
- RuntimeRLHelper: Simplified YAML-based integration (recommended)
"""

from adaptiq.core.runtime_rl.runtime_decision_engine import (
    RuntimeDecision,
    RuntimeDecisionEngine,
)
from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
from adaptiq.core.runtime_rl.runtime_rewards import (
    AccuracyRewardCalculator,
    BaseRuntimeRewardCalculator,
    ClassificationRewardCalculator,
    CustomRewardCalculator,
    create_reward_calculator,
)
from adaptiq.core.runtime_rl.runtime_rl_helper import RuntimeRLHelper

__all__ = [
    "RuntimeQTableManager",
    "BaseRuntimeRewardCalculator",
    "AccuracyRewardCalculator",
    "ClassificationRewardCalculator",
    "CustomRewardCalculator",
    "create_reward_calculator",
    "RuntimeDecisionEngine",
    "RuntimeDecision",
    "RuntimeRLHelper",
]

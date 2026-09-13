#!/usr/bin/env python3
"""
Runtime Decision Engine - Orchestrator for Runtime RL

This module provides the main orchestration logic for runtime reinforcement learning,
integrating the RuntimeQTableManager and RuntimeRewardCalculator to make online
decisions during agent execution.

Architecture:
    User Code
        ↓
    RuntimeDecisionEngine (this file)
        ├─> RuntimeQTableManager (epsilon-greedy selection)
        ├─> RuntimeRewardCalculator (reward calculation)
        └─> User-defined Executors (action execution)

Flow:
    1. decide(context) → selects action via epsilon-greedy
    2. User executes action → gets result
    3. update(result) → calculates reward, updates Q-table

Example:
    >>> # Setup
    >>> engine = RuntimeDecisionEngine(
    ...     q_table_manager=RuntimeQTableManager(),
    ...     reward_calculator=AccuracyRewardCalculator()
    ... )
    >>>
    >>> # Define available actions
    >>> engine.register_actions([
    ...     QTableAction(action="method_a"),
    ...     QTableAction(action="method_b")
    ... ])
    >>>
    >>> # Runtime decision-making
    >>> context = {"subtask": "price_estimation", "last_outcome": "success"}
    >>> decision = engine.decide(context)
    >>> print(f"Selected: {decision.action.action}, Explored: {decision.explored}")
    >>>
    >>> # Execute action (user code)
    >>> result = execute_method(decision.action.action)
    >>>
    >>> # Update Q-table with result
    >>> engine.update(decision, result, context)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from adaptiq.core.entities.q_table import QTableAction, QTableState
from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
from adaptiq.core.runtime_rl.runtime_rewards import BaseRuntimeRewardCalculator

# Configure logging
logger = logging.getLogger("ADAPTIQ-RuntimeRL")


@dataclass
class RuntimeDecision:
    """
    Result of a runtime decision.

    Attributes:
        action: The selected action to execute
        explored: Whether exploration was used (True) or exploitation (False)
        state: The state from which the decision was made
        q_value: The Q-value of the selected action in this state
    """

    action: QTableAction
    explored: bool
    state: QTableState
    q_value: float

    def __repr__(self) -> str:
        exploration_type = "EXPLORATION" if self.explored else "EXPLOITATION"
        return (
            f"RuntimeDecision("
            f"action='{self.action.action}', "
            f"{exploration_type}, "
            f"Q={self.q_value:.4f}"
            f")"
        )


class RuntimeDecisionEngine:
    """
    Runtime Decision Engine - Orchestrates online RL decisions.

    This engine integrates:
    - RuntimeQTableManager: For epsilon-greedy action selection
    - RuntimeRewardCalculator: For domain-specific reward calculation
    - Action Registry: Available actions for decision-making

    The engine maintains the last decision to enable proper Q-learning updates
    when the result becomes available.

    Example:
        >>> # Initialize components
        >>> q_manager = RuntimeQTableManager(epsilon=0.1)
        >>> reward_calc = AccuracyRewardCalculator()
        >>>
        >>> # Create engine
        >>> engine = RuntimeDecisionEngine(
        ...     q_table_manager=q_manager,
        ...     reward_calculator=reward_calc
        ... )
        >>>
        >>> # Register available actions
        >>> actions = [
        ...     QTableAction(action="linear_regression"),
        ...     QTableAction(action="neural_network")
        ... ]
        >>> engine.register_actions(actions)
        >>>
        >>> # Make decision
        >>> context = {
        ...     "subtask": "price_prediction",
        ...     "last_action": "None",
        ...     "last_outcome": "None",
        ...     "key_context": "small_dataset"
        ... }
        >>> decision = engine.decide(context)
        >>>
        >>> # Execute action (user code)
        >>> result = {"accuracy": 0.85, "error_rate": 0.15}
        >>>
        >>> # Update Q-table
        >>> new_context = {
        ...     "subtask": "price_prediction",
        ...     "last_action": decision.action.action,
        ...     "last_outcome": "success",
        ...     "key_context": "small_dataset"
        ... }
        >>> engine.update(decision, result, new_context)
    """

    def __init__(
        self,
        q_table_manager: RuntimeQTableManager,
        reward_calculator: BaseRuntimeRewardCalculator,
    ):
        """
        Initialize Runtime Decision Engine.

        Args:
            q_table_manager: RuntimeQTableManager instance for action selection
            reward_calculator: Reward calculator for result evaluation
        """
        self.q_manager = q_table_manager
        self.reward_calc = reward_calculator
        self.available_actions: List[QTableAction] = []

        logger.info(
            f"RuntimeDecisionEngine initialized with "
            f"Q-manager: {type(q_table_manager).__name__}, "
            f"Reward calculator: {type(reward_calculator).__name__}"
        )

    def register_actions(self, actions: List[QTableAction]) -> None:
        """
        Register available actions for decision-making.

        Args:
            actions: List of QTableAction objects representing available choices

        Raises:
            ValueError: If actions list is empty
        """
        if not actions:
            raise ValueError("Cannot register empty actions list")

        self.available_actions = actions

        logger.info(
            f"Registered {len(actions)} actions: " f"{[a.action for a in actions]}"
        )

    def _construct_context(self, metadata: Dict[str, Any]) -> str:
        """
        Construct key_context from metadata dictionary (fallback).

        This is a domain-agnostic fallback that concatenates the top 3-5
        metadata fields into a string. Users should prefer providing
        key_context directly for full control.

        Args:
            metadata: Dictionary with arbitrary metadata fields

        Returns:
            str: Constructed key_context string

        Example:
            >>> metadata = {"material": "concrete", "surface": 20, "region": "north"}
            >>> engine._construct_context(metadata)
            'concrete_20_north'

            >>> metadata = {"type": "contract", "jurisdiction": "france", "value": 50000}
            >>> engine._construct_context(metadata)
            'contract_france_50000'
        """
        if not metadata:
            return "unknown"

        # Extract up to 5 key-value pairs (domain-agnostic)
        # Sort keys alphabetically for consistency
        sorted_keys = sorted(metadata.keys())[:5]

        # Build key_context string
        context_parts = []
        for key in sorted_keys:
            value = metadata[key]
            # Convert value to string, handle different types
            if isinstance(value, (int, float)):
                value_str = (
                    str(int(value))
                    if isinstance(value, float) and value.is_integer()
                    else str(value)
                )
            else:
                value_str = str(value)

            # Clean value (remove spaces, special chars)
            value_str = value_str.replace(" ", "_").replace("-", "_")
            context_parts.append(value_str)

        key_context = "_".join(context_parts)

        logger.debug(
            f"Constructed key_context from metadata: {key_context} "
            f"(from keys: {sorted_keys})"
        )

        return key_context

    def build_state(self, context: Dict[str, Any]) -> QTableState:
        """
        Build QTableState from context dictionary.

        Supports three modes for key_context construction:
        1. Direct: key_context provided in context (preferred)
        2. Fallback: Constructed from metadata if key_context missing
        3. Default: "unknown" if both missing

        Args:
            context: Dictionary with keys:
                    - "subtask": Current subtask name
                    - "last_action": Last action taken (or "None")
                    - "last_outcome": Outcome of last action (or "None")
                    - "key_context": Domain-specific context (optional if metadata provided)
                    - "metadata": Dict for fallback construction (optional)

        Returns:
            QTableState: Constructed state object

        Raises:
            ValueError: If required keys missing from context

        Example:
            >>> # Mode 1: Direct key_context
            >>> context = {"subtask": "task", "last_action": "None",
            ...            "last_outcome": "None", "key_context": "my_context"}
            >>> state = engine.build_state(context)

            >>> # Mode 2: Fallback from metadata
            >>> context = {"subtask": "task", "last_action": "None",
            ...            "last_outcome": "None",
            ...            "metadata": {"material": "concrete", "surface": 20}}
            >>> state = engine.build_state(context)  # key_context="concrete_20"
        """
        # Validate base required keys
        base_required_keys = ["subtask", "last_action", "last_outcome"]
        missing_keys = [key for key in base_required_keys if key not in context]

        if missing_keys:
            raise ValueError(
                f"Missing required keys in context: {missing_keys}. "
                f"Expected keys: {base_required_keys} + (key_context OR metadata)"
            )

        # Determine key_context (priority order)
        if "key_context" in context:
            # Mode 1: Direct (preferred)
            key_context = context["key_context"]
        elif "metadata" in context and context["metadata"]:
            # Mode 2: Fallback from metadata
            key_context = self._construct_context(context["metadata"])
            logger.info(
                f"key_context not provided, constructed from metadata: '{key_context}'"
            )
        else:
            # Mode 3: Default
            key_context = "unknown"
            logger.warning(
                "Neither key_context nor metadata provided, using default: 'unknown'"
            )

        state = QTableState(
            current_subtask=context["subtask"],
            last_action_taken=context["last_action"],
            last_outcome=context["last_outcome"],
            key_context=key_context,
        )

        return state

    def decide(self, context: Dict[str, str]) -> RuntimeDecision:
        """
        Make a runtime decision using epsilon-greedy strategy.

        This method:
        1. Builds QTableState from context
        2. Selects action via epsilon-greedy (exploration/exploitation)
        3. Returns RuntimeDecision with selected action and metadata

        Args:
            context: Dictionary with state information (see build_state())

        Returns:
            RuntimeDecision: Decision object with selected action and metadata

        Raises:
            ValueError: If no actions registered
            ValueError: If context missing required keys

        Example:
            >>> context = {
            ...     "subtask": "classification",
            ...     "last_action": "None",
            ...     "last_outcome": "None",
            ...     "key_context": "balanced_dataset"
            ... }
            >>> decision = engine.decide(context)
            >>> print(f"Action: {decision.action.action}")
            >>> print(f"Explored: {decision.explored}")
        """
        # Validate actions registered
        if not self.available_actions:
            raise ValueError("No actions registered. Call register_actions() first.")

        # Build state from context
        state = self.build_state(context)

        # Select action using epsilon-greedy
        action, explored = self.q_manager.select_action_epsilon_greedy(
            state, self.available_actions
        )

        # Get Q-value for logging
        q_value = self.q_manager.Q(state, action)

        # Create decision object
        decision = RuntimeDecision(
            action=action,
            explored=explored,
            state=state,
            q_value=q_value,
        )

        logger.info(
            f"Decision made: {decision.action.action} "
            f"({'EXPLORE' if explored else 'EXPLOIT'}) "
            f"Q={q_value:.4f}"
        )

        return decision

    def update(
        self,
        decision: RuntimeDecision,
        result: Dict[str, Any],
        next_context: Dict[str, str],
    ) -> float:
        """
        Update Q-table based on action execution result.

        This method:
        1. Calculates reward from result using reward calculator
        2. Builds next state from next_context
        3. Updates Q-table using Q-learning (Bellman equation)
        4. Returns the new Q-value

        Args:
            decision: The RuntimeDecision that was executed
            result: Dictionary with execution results (reward calculator-specific)
            next_context: Context after action execution (see build_state())

        Returns:
            float: Updated Q-value for (state, action) pair

        Raises:
            ValueError: If reward calculation fails
            ValueError: If next_context missing required keys

        Example:
            >>> result = {"accuracy": 0.9, "error_rate": 0.1}
            >>> next_context = {
            ...     "subtask": "classification",
            ...     "last_action": decision.action.action,
            ...     "last_outcome": "success",
            ...     "key_context": "balanced_dataset"
            ... }
            >>> new_q = engine.update(decision, result, next_context)
            >>> print(f"Updated Q-value: {new_q:.4f}")
        """
        # Calculate reward
        reward = self.reward_calc.calculate_reward(result)

        # Build next state
        next_state = self.build_state(next_context)

        # Update Q-table using Bellman equation (inherited from QTableManager)
        new_q_value = self.q_manager.update_policy(
            s=decision.state,
            a=decision.action,
            R=reward,
            s_prime=next_state,
            actions_prime=self.available_actions,
        )

        logger.info(
            f"Q-table updated: {decision.action.action} "
            f"reward={reward:.3f}, "
            f"Q: {decision.q_value:.4f} → {new_q_value:.4f}"
        )

        return new_q_value

    def save_q_table(self, prefix_version: str = "runtime") -> bool:
        """
        Save Q-table to disk.

        Args:
            prefix_version: Prefix for versioning (default: "runtime")

        Returns:
            bool: True if save successful, False otherwise
        """
        success = self.q_manager.save_q_table(prefix_version=prefix_version)

        if success:
            logger.info(f"Q-table saved successfully with prefix '{prefix_version}'")
        else:
            logger.error("Q-table save failed")

        return success

    def load_q_table(self) -> bool:
        """
        Load Q-table from disk.

        Returns:
            bool: True if load successful, False otherwise
        """
        success = self.q_manager.load_q_table()

        if success:
            logger.info("Q-table loaded successfully")
        else:
            logger.warning("Q-table load failed (may not exist yet)")

        return success

    def set_epsilon(self, new_epsilon: float) -> None:
        """
        Update exploration rate (epsilon).

        Useful for epsilon decay strategies:
        - Start with high epsilon (e.g., 0.3) for initial exploration
        - Gradually decay to low epsilon (e.g., 0.05) for exploitation

        Args:
            new_epsilon: New exploration rate in [0, 1]

        Example:
            >>> # Epsilon decay strategy
            >>> engine.set_epsilon(0.3)  # Initial: 30% exploration
            >>> # ... after some decisions ...
            >>> engine.set_epsilon(0.1)  # Reduced: 10% exploration
            >>> # ... after more decisions ...
            >>> engine.set_epsilon(0.05)  # Final: 5% exploration
        """
        self.q_manager.set_epsilon(new_epsilon)
        logger.info(f"Epsilon updated to {new_epsilon:.4f}")

    def get_epsilon(self) -> float:
        """
        Get current exploration rate.

        Returns:
            float: Current epsilon value
        """
        return self.q_manager.get_epsilon()

    def get_hyperparameters(self) -> Dict[str, float]:
        """
        Get current RL hyperparameters.

        Returns:
            dict: Dictionary with alpha, gamma, epsilon values
        """
        return {
            "alpha": self.q_manager.alpha,
            "gamma": self.q_manager.gamma,
            "epsilon": self.q_manager.epsilon,
        }

    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"RuntimeDecisionEngine("
            f"actions={len(self.available_actions)}, "
            f"alpha={self.q_manager.alpha}, "
            f"gamma={self.q_manager.gamma}, "
            f"epsilon={self.q_manager.epsilon}"
            f")"
        )

"""
RuntimeRLHelper - Simplified YAML-based Runtime RL Integration

This module provides the simplest way for developers to integrate Runtime RL
into their agents using YAML configuration.

Developer usage (3 lines):
    ```python
    from adaptiq.core.runtime_rl import RuntimeRLHelper

    # Load from YAML config
    runtime_rl = RuntimeRLHelper.from_yaml("agents/my_agent/runtime_rl_config.yaml")

    # Use in agent
    decision = runtime_rl.decide(metadata={"key": "value"})
    result = execute_method(decision.action.action)
    runtime_rl.update(decision, result)
    ```

YAML config structure:
    ```yaml
    runtime_rl:
      # How to build key_context from metadata
      key_context_template: "{material}_{region}_{surface}"

      # Reward function (Python code as string or builtin name)
      reward_function: "accuracy_reward"  # or custom Python expression

      # Available actions
      actions:
        - method_db_standard
        - method_knn_historical
        - method_regional_adjust

      # Storage path for Q-table
      storage_path: "storage/qtables/my_agent_runtime.json"

      # Hyperparameters (optional)
      hyperparameters:
        alpha: 0.1
        gamma: 0.9
        epsilon: 0.1
    ```
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from adaptiq.core.entities.q_table import QTableAction
from adaptiq.core.runtime_rl.runtime_decision_engine import (
    RuntimeDecision,
    RuntimeDecisionEngine,
)
from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
from adaptiq.core.runtime_rl.runtime_rewards import (
    AccuracyRewardCalculator,
    ClassificationRewardCalculator,
    CustomRewardCalculator,
)

logger = logging.getLogger(__name__)


class RuntimeRLHelper:
    """
    Simplified helper for Runtime RL integration using YAML configuration.

    This class provides a minimalistic API for developers to integrate
    Runtime RL into their agents with zero boilerplate code.

    Features:
    - Load configuration from YAML file
    - Auto-construct key_context from metadata using templates
    - Builtin reward functions + custom Python expressions
    - Auto-manage Q-table storage
    - Simple 3-step API: decide() -> execute() -> update()
    """

    def __init__(
        self,
        key_context_builder: Callable[[Dict[str, Any]], str],
        reward_function: Callable[[Dict[str, Any]], float],
        actions: List[str],
        storage_path: str,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
    ):
        """
        Initialize RuntimeRLHelper.

        Args:
            key_context_builder: Function to build key_context from metadata
            reward_function: Function to calculate reward from result
            actions: List of action names (method names)
            storage_path: Path to store Q-table JSON file
            alpha: Learning rate (default: 0.1)
            gamma: Discount factor (default: 0.9)
            epsilon: Exploration rate (default: 0.1)
        """
        self.key_context_builder = key_context_builder
        self.reward_function = reward_function
        self.actions = actions
        self.storage_path = storage_path

        # Initialize Q-Table Manager
        self.q_manager = RuntimeQTableManager(
            file_path=storage_path,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

        # Initialize Reward Calculator
        self.reward_calculator = CustomRewardCalculator(reward_fn=reward_function)

        # Initialize Decision Engine
        self.decision_engine = RuntimeDecisionEngine(
            q_table_manager=self.q_manager,
            reward_calculator=self.reward_calculator,
        )

        # Register actions
        q_actions = [QTableAction(action=action) for action in actions]
        self.decision_engine.register_actions(q_actions)

        logger.info(f"RuntimeRLHelper initialized with {len(actions)} actions")
        logger.info(f"Q-table storage: {storage_path}")

    @classmethod
    def from_yaml(cls, config_path: str) -> "RuntimeRLHelper":
        """
        Create RuntimeRLHelper from YAML configuration file.

        This is the simplest way for developers to initialize Runtime RL.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            RuntimeRLHelper instance

        Example:
            ```python
            runtime_rl = RuntimeRLHelper.from_yaml("agents/btp_agent/runtime_rl_config.yaml")
            ```

        YAML structure:
            ```yaml
            runtime_rl:
              key_context_template: "{material}_{region}_{surface}"
              reward_function: "accuracy_reward"
              actions:
                - method_a
                - method_b
              storage_path: "storage/qtables/runtime.json"
              hyperparameters:
                alpha: 0.1
                gamma: 0.9
                epsilon: 0.1
            ```
        """
        # Load YAML file
        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"YAML config not found: {config_path}")

        with open(config_path_obj, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "runtime_rl" not in config:
            raise ValueError(
                f"YAML must contain 'runtime_rl' key. Found keys: {list(config.keys())}"
            )

        rl_config = config["runtime_rl"]

        # Parse key_context_template
        key_context_template = rl_config.get("key_context_template")
        if not key_context_template:
            raise ValueError("YAML must specify 'key_context_template'")

        def key_context_builder(metadata: Dict[str, Any]) -> str:
            """Build key_context from template using metadata fields."""
            try:
                return key_context_template.format(**metadata)
            except KeyError as e:
                missing_key = str(e).strip("'")
                raise ValueError(
                    f"key_context_template requires field '{missing_key}' "
                    f"but it's missing in metadata. Available: {list(metadata.keys())}"
                )

        # Parse reward_function
        reward_fn_name = rl_config.get("reward_function")
        if not reward_fn_name:
            raise ValueError("YAML must specify 'reward_function'")

        reward_function = cls._parse_reward_function(reward_fn_name)

        # Parse actions
        actions = rl_config.get("actions", [])
        if not actions:
            raise ValueError("YAML must specify at least one action in 'actions' list")

        # Parse storage_path
        storage_path = rl_config.get(
            "storage_path", "storage/qtables/runtime_q_table.json"
        )

        # Ensure storage directory exists
        storage_dir = Path(storage_path).parent
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Parse hyperparameters
        hyperparams = rl_config.get("hyperparameters", {})
        alpha = hyperparams.get("alpha", 0.1)
        gamma = hyperparams.get("gamma", 0.9)
        epsilon = hyperparams.get("epsilon", 0.1)

        logger.info(f"Loaded Runtime RL config from: {config_path}")
        logger.info(f"  - Actions: {actions}")
        logger.info(f"  - key_context template: {key_context_template}")
        logger.info(f"  - Reward function: {reward_fn_name}")
        logger.info(f"  - Hyperparameters: α={alpha}, γ={gamma}, ε={epsilon}")

        return cls(
            key_context_builder=key_context_builder,
            reward_function=reward_function,
            actions=actions,
            storage_path=storage_path,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    @staticmethod
    def _parse_reward_function(
        reward_fn_name: str,
    ) -> Callable[[Dict[str, Any]], float]:
        """
        Parse reward function from string name or Python expression.

        Supports:
        - Builtin: "accuracy_reward", "classification_reward"
        - Python expression: "lambda result: 1.0 - result['error']"
        - Custom module import: "my_module.my_reward_function"
        """
        # Builtin reward functions
        if reward_fn_name == "accuracy_reward":

            def accuracy_reward(result: Dict[str, Any]) -> float:
                """Reward based on error percentage: reward = 1 - (error / 100)"""
                error_percent = result.get("error_percent", 0.0)
                return max(-1.0, 1.0 - (error_percent / 100.0))

            return accuracy_reward

        elif reward_fn_name == "classification_reward":

            def classification_reward(result: Dict[str, Any]) -> float:
                """Binary reward: +1 if correct, -1 if wrong"""
                return 1.0 if result.get("correct", False) else -1.0

            return classification_reward

        # Python lambda expression
        elif reward_fn_name.startswith("lambda"):
            try:
                # Security note: eval() should only be used with trusted config files
                reward_fn = eval(reward_fn_name)
                return reward_fn
            except Exception as e:
                raise ValueError(
                    f"Failed to parse lambda expression: {reward_fn_name}. Error: {e}"
                )

        # Custom module import
        elif "." in reward_fn_name:
            try:
                module_path, function_name = reward_fn_name.rsplit(".", 1)
                import importlib

                module = importlib.import_module(module_path)
                reward_fn = getattr(module, function_name)
                return reward_fn
            except Exception as e:
                raise ValueError(
                    f"Failed to import reward function: {reward_fn_name}. Error: {e}"
                )

        else:
            raise ValueError(
                f"Unknown reward function: {reward_fn_name}. "
                f"Use 'accuracy_reward', 'classification_reward', lambda expression, or 'module.function'"
            )

    def decide(
        self,
        subtask: str,
        metadata: Dict[str, Any],
        last_action: str = "None",
        last_outcome: str = "None",
    ) -> RuntimeDecision:
        """
        Make a decision using Runtime RL.

        Args:
            subtask: Current subtask name (e.g., "estimate_price")
            metadata: Metadata dictionary for building key_context
            last_action: Last action taken (default: "None")
            last_outcome: Outcome of last action (default: "None")

        Returns:
            RuntimeDecision with selected action and Q-value

        Example:
            ```python
            decision = runtime_rl.decide(
                subtask="estimate_price",
                metadata={"material": "concrete", "region": "north", "surface": 20}
            )
            print(f"Selected method: {decision.action.action}")
            print(f"Q-value: {decision.q_value:.4f}")
            ```
        """
        # Build key_context from metadata
        key_context = self.key_context_builder(metadata)

        # Build context for decision engine
        context = {
            "subtask": subtask,
            "last_action": last_action,
            "last_outcome": last_outcome,
            "key_context": key_context,
        }

        # Make decision
        decision = self.decision_engine.decide(context)

        logger.info(
            f"Runtime RL decision: action={decision.action.action}, "
            f"q_value={decision.q_value:.4f}, explored={decision.explored}"
        )

        return decision

    def update(
        self,
        decision: RuntimeDecision,
        result: Dict[str, Any],
        next_subtask: Optional[str] = None,
        next_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update Q-table after executing action.

        Args:
            decision: The decision returned by decide()
            result: Result dictionary containing outcome metrics
            next_subtask: Next subtask name (optional, defaults to "terminal")
            next_metadata: Metadata for next state (optional)

        Example:
            ```python
            # Execute method and get result
            result = {
                "estimated": 5250.0,
                "actual": 5500.0,
                "error_percent": 4.5
            }

            # Update Q-table
            runtime_rl.update(decision, result)
            ```
        """
        # Calculate reward
        reward = self.reward_function(result)

        # Build next context
        if next_subtask and next_metadata:
            next_key_context = self.key_context_builder(next_metadata)
            next_context = {
                "subtask": next_subtask,
                "last_action": decision.action.action,
                "last_outcome": result.get("outcome", "success"),
                "key_context": next_key_context,
            }
        else:
            # Terminal state
            next_context = {
                "subtask": "terminal",
                "last_action": decision.action.action,
                "last_outcome": result.get("outcome", "success"),
                "key_context": "terminal",
            }

        # Update Q-table
        self.decision_engine.update(decision, result, next_context)

        logger.info(
            f"Q-table updated: reward={reward:.4f}, next_subtask={next_subtask or 'terminal'}"
        )

    def save(self) -> None:
        """
        Save Q-table to disk.

        Example:
            ```python
            # Save after training
            runtime_rl.save()
            ```
        """
        self.q_manager.save_q_table()
        logger.info(f"Q-table saved to: {self.storage_path}")

    def load(self) -> None:
        """
        Load Q-table from disk.

        Example:
            ```python
            # Load existing Q-table
            runtime_rl.load()
            ```
        """
        self.q_manager.load_q_table()
        logger.info(f"Q-table loaded from: {self.storage_path}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Runtime RL statistics.

        Returns:
            Dictionary with Q-table size, actions, hyperparameters

        Example:
            ```python
            stats = runtime_rl.get_stats()
            print(f"Q-table size: {stats['q_table_size']}")
            print(f"Actions: {stats['actions']}")
            ```
        """
        return {
            "q_table_size": len(self.q_manager.Q_table),
            "actions": self.actions,
            "storage_path": self.storage_path,
            "hyperparameters": {
                "alpha": self.q_manager.alpha,
                "gamma": self.q_manager.gamma,
                "epsilon": self.q_manager.epsilon,
            },
        }

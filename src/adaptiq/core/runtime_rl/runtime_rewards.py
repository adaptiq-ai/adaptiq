#!/usr/bin/env python3
"""
Runtime Reward Calculator - Domain-specific reward calculation for Runtime RL

This module provides reward calculation strategies for runtime decision-making,
focusing on business metrics (accuracy, error rate) rather than agent execution
metrics (tool success, token usage).

Key Differences from Offline Rewards (CrewRewards):
- Offline: Measures agent execution quality (tool calls, token usage, task completion)
- Runtime: Measures business outcomes (accuracy, precision, recall, error rates)

Architecture:
- BaseRuntimeRewardCalculator: Abstract base class
- AccuracyRewardCalculator: Simple accuracy-based rewards
- ClassificationRewardCalculator: Classification metrics (precision/recall/F1)
- CustomRewardCalculator: User-defined custom reward functions
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

# Configure logging
logger = logging.getLogger("ADAPTIQ-RuntimeRL")


class BaseRuntimeRewardCalculator(ABC):
    """
    Abstract base class for runtime reward calculation.

    All runtime reward calculators must implement calculate_reward() method
    which returns a float reward in range [-1, 1].

    The reward system should:
    - Return positive rewards for good outcomes
    - Return negative rewards for bad outcomes
    - Be normalized to [-1, 1] range for stable Q-learning

    Example:
        >>> class MyRewardCalculator(BaseRuntimeRewardCalculator):
        ...     def calculate_reward(self, result: Dict[str, Any]) -> float:
        ...         return 1.0 if result["success"] else -1.0
    """

    @abstractmethod
    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward based on action execution result.

        Args:
            result: Dictionary containing execution results with metrics
                   Expected keys vary by calculator type

        Returns:
            float: Reward value in range [-1, 1]
                  - Positive rewards: Good outcomes
                  - Negative rewards: Bad outcomes
                  - Zero: Neutral outcome

        Raises:
            ValueError: If required keys missing from result
        """
        pass

    def validate_result(self, result: Dict[str, Any], required_keys: list) -> None:
        """
        Validate that result contains required keys.

        Args:
            result: Result dictionary to validate
            required_keys: List of required key names

        Raises:
            ValueError: If any required key is missing
        """
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in result: {missing_keys}. "
                f"Expected keys: {required_keys}"
            )


class AccuracyRewardCalculator(BaseRuntimeRewardCalculator):
    """
    Simple accuracy-based reward calculator.

    Calculates reward based on accuracy metric with optional error penalty.

    Formula:
        reward = success_weight * accuracy - error_weight * error_rate

    Where:
        - accuracy: Correctness metric in [0, 1]
        - error_rate: Error rate in [0, 1]
        - success_weight: Weight for accuracy (default: 1.0)
        - error_weight: Weight for errors (default: 0.5)

    Example:
        >>> calculator = AccuracyRewardCalculator()
        >>> result = {"accuracy": 0.9, "error_rate": 0.1}
        >>> reward = calculator.calculate_reward(result)
        >>> # reward = 1.0 * 0.9 - 0.5 * 0.1 = 0.85
    """

    def __init__(
        self,
        success_weight: float = 1.0,
        error_weight: float = 0.5,
    ):
        """
        Initialize Accuracy Reward Calculator.

        Args:
            success_weight: Weight for accuracy term (default: 1.0)
            error_weight: Weight for error penalty term (default: 0.5)
        """
        self.success_weight = success_weight
        self.error_weight = error_weight

        logger.info(
            f"AccuracyRewardCalculator initialized: "
            f"success_weight={success_weight}, error_weight={error_weight}"
        )

    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward based on accuracy and error rate.

        Args:
            result: Dictionary with keys:
                   - "accuracy": float in [0, 1]
                   - "error_rate": float in [0, 1]

        Returns:
            float: Reward = success_weight * accuracy - error_weight * error_rate

        Raises:
            ValueError: If required keys missing or values out of range
        """
        # Validate required keys
        self.validate_result(result, ["accuracy", "error_rate"])

        accuracy = result["accuracy"]
        error_rate = result["error_rate"]

        # Validate ranges
        if not 0 <= accuracy <= 1:
            raise ValueError(f"Accuracy must be in [0, 1], got {accuracy}")
        if not 0 <= error_rate <= 1:
            raise ValueError(f"Error rate must be in [0, 1], got {error_rate}")

        # Calculate reward
        reward = self.success_weight * accuracy - self.error_weight * error_rate

        # Clip to [-1, 1] for safety
        reward = max(-1.0, min(1.0, reward))

        logger.debug(
            f"Accuracy reward: {self.success_weight}*{accuracy:.3f} - "
            f"{self.error_weight}*{error_rate:.3f} = {reward:.3f}"
        )

        return reward


class ClassificationRewardCalculator(BaseRuntimeRewardCalculator):
    """
    Classification-specific reward calculator.

    Calculates reward using precision, recall, and F1-score metrics.
    Useful for classification tasks where both precision and recall matter.

    Formula:
        reward = precision_weight * precision + recall_weight * recall
              or reward = f1_weight * f1_score (if use_f1=True)

    Where:
        - precision: TP / (TP + FP) - correctness of positive predictions
        - recall: TP / (TP + FN) - coverage of actual positives
        - f1_score: 2 * (precision * recall) / (precision + recall)

    Example:
        >>> calculator = ClassificationRewardCalculator(use_f1=True)
        >>> result = {"precision": 0.8, "recall": 0.9, "f1_score": 0.85}
        >>> reward = calculator.calculate_reward(result)
        >>> # reward = 0.85 (F1-score)
    """

    def __init__(
        self,
        precision_weight: float = 0.5,
        recall_weight: float = 0.5,
        f1_weight: float = 1.0,
        use_f1: bool = False,
    ):
        """
        Initialize Classification Reward Calculator.

        Args:
            precision_weight: Weight for precision (default: 0.5)
            recall_weight: Weight for recall (default: 0.5)
            f1_weight: Weight for F1-score (default: 1.0)
            use_f1: If True, use F1-score only; if False, use precision+recall
        """
        self.precision_weight = precision_weight
        self.recall_weight = recall_weight
        self.f1_weight = f1_weight
        self.use_f1 = use_f1

        logger.info(
            f"ClassificationRewardCalculator initialized: "
            f"precision_weight={precision_weight}, recall_weight={recall_weight}, "
            f"f1_weight={f1_weight}, use_f1={use_f1}"
        )

    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward based on classification metrics.

        Args:
            result: Dictionary with keys:
                   If use_f1=True: ["f1_score"]
                   If use_f1=False: ["precision", "recall"]

        Returns:
            float: Reward based on classification metrics

        Raises:
            ValueError: If required keys missing or values out of range
        """
        if self.use_f1:
            # Use F1-score only
            self.validate_result(result, ["f1_score"])
            f1_score = result["f1_score"]

            if not 0 <= f1_score <= 1:
                raise ValueError(f"F1-score must be in [0, 1], got {f1_score}")

            reward = self.f1_weight * f1_score

            logger.debug(f"Classification reward (F1): {self.f1_weight}*{f1_score:.3f} = {reward:.3f}")

        else:
            # Use precision and recall
            self.validate_result(result, ["precision", "recall"])
            precision = result["precision"]
            recall = result["recall"]

            if not 0 <= precision <= 1:
                raise ValueError(f"Precision must be in [0, 1], got {precision}")
            if not 0 <= recall <= 1:
                raise ValueError(f"Recall must be in [0, 1], got {recall}")

            reward = self.precision_weight * precision + self.recall_weight * recall

            logger.debug(
                f"Classification reward: {self.precision_weight}*{precision:.3f} + "
                f"{self.recall_weight}*{recall:.3f} = {reward:.3f}"
            )

        # Clip to [-1, 1] for safety
        reward = max(-1.0, min(1.0, reward))

        return reward


class CustomRewardCalculator(BaseRuntimeRewardCalculator):
    """
    Custom reward calculator using user-defined reward function.

    Allows users to define their own reward calculation logic for
    domain-specific metrics and business requirements.

    Example:
        >>> def my_reward_fn(result: Dict[str, Any]) -> float:
        ...     # Custom logic
        ...     return 1.0 if result["profit"] > 100 else -1.0
        >>>
        >>> calculator = CustomRewardCalculator(reward_fn=my_reward_fn)
        >>> result = {"profit": 150}
        >>> reward = calculator.calculate_reward(result)
        >>> # reward = 1.0
    """

    def __init__(self, reward_fn: Callable[[Dict[str, Any]], float]):
        """
        Initialize Custom Reward Calculator.

        Args:
            reward_fn: Function that takes result dict and returns reward float
                      Must return value in range [-1, 1]
        """
        if not callable(reward_fn):
            raise ValueError("reward_fn must be a callable function")

        self.reward_fn = reward_fn

        logger.info(
            f"CustomRewardCalculator initialized with function: {reward_fn.__name__}"
        )

    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """
        Calculate reward using custom reward function.

        Args:
            result: Dictionary with arbitrary keys (function-dependent)

        Returns:
            float: Reward from custom function, clipped to [-1, 1]

        Raises:
            Exception: If custom function raises an exception
        """
        try:
            reward = self.reward_fn(result)

            # Ensure reward is float
            reward = float(reward)

            # Clip to [-1, 1] for safety
            reward = max(-1.0, min(1.0, reward))

            logger.debug(f"Custom reward calculated: {reward:.3f}")

            return reward

        except Exception as e:
            logger.error(f"Custom reward function failed: {e}")
            raise


# Convenience factory function
def create_reward_calculator(
    calculator_type: str = "accuracy",
    **kwargs
) -> BaseRuntimeRewardCalculator:
    """
    Factory function to create reward calculators.

    Args:
        calculator_type: Type of calculator ("accuracy", "classification", "custom")
        **kwargs: Arguments to pass to calculator constructor

    Returns:
        BaseRuntimeRewardCalculator: Instance of requested calculator

    Raises:
        ValueError: If calculator_type is unknown

    Example:
        >>> calc = create_reward_calculator("accuracy", success_weight=1.0, error_weight=0.5)
        >>> calc = create_reward_calculator("classification", use_f1=True)
        >>> calc = create_reward_calculator("custom", reward_fn=my_custom_function)
    """
    calculators = {
        "accuracy": AccuracyRewardCalculator,
        "classification": ClassificationRewardCalculator,
        "custom": CustomRewardCalculator,
    }

    if calculator_type not in calculators:
        raise ValueError(
            f"Unknown calculator type: {calculator_type}. "
            f"Valid types: {list(calculators.keys())}"
        )

    calculator_class = calculators[calculator_type]
    return calculator_class(**kwargs)

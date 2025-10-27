#!/usr/bin/env python3
"""
Tests for Runtime Reward Calculators

Validates:
1. Abstract base class structure
2. AccuracyRewardCalculator formula and validation
3. ClassificationRewardCalculator with precision/recall/F1
4. CustomRewardCalculator with user-defined functions
5. Factory function create_reward_calculator()
6. Reward normalization to [-1, 1]
"""

import pytest
from adaptiq.core.runtime_rl.runtime_rewards import (
    BaseRuntimeRewardCalculator,
    AccuracyRewardCalculator,
    ClassificationRewardCalculator,
    CustomRewardCalculator,
    create_reward_calculator,
)


class TestBaseRuntimeRewardCalculator:
    """Test abstract base class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that BaseRuntimeRewardCalculator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseRuntimeRewardCalculator()

    def test_subclass_must_implement_calculate_reward(self):
        """Test that subclasses must implement calculate_reward()"""

        # Missing implementation
        with pytest.raises(TypeError):

            class IncompleteCalculator(BaseRuntimeRewardCalculator):
                pass

            IncompleteCalculator()

        # Correct implementation
        class CompleteCalculator(BaseRuntimeRewardCalculator):
            def calculate_reward(self, result):
                return 1.0

        calc = CompleteCalculator()
        assert calc.calculate_reward({}) == 1.0

    def test_validate_result_helper(self):
        """Test validate_result() helper method"""

        class TestCalculator(BaseRuntimeRewardCalculator):
            def calculate_reward(self, result):
                self.validate_result(result, ["accuracy", "error_rate"])
                return 1.0

        calc = TestCalculator()

        # Valid result
        result = {"accuracy": 0.9, "error_rate": 0.1}
        assert calc.calculate_reward(result) == 1.0

        # Missing key
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"accuracy": 0.9})


class TestAccuracyRewardCalculator:
    """Test AccuracyRewardCalculator"""

    def test_initialization_default_params(self):
        """Test default initialization"""
        calc = AccuracyRewardCalculator()

        assert calc.success_weight == 1.0
        assert calc.error_weight == 0.5

    def test_initialization_custom_params(self):
        """Test custom initialization"""
        calc = AccuracyRewardCalculator(success_weight=0.8, error_weight=0.3)

        assert calc.success_weight == 0.8
        assert calc.error_weight == 0.3

    def test_calculate_reward_formula(self):
        """Test reward formula: success_weight * accuracy - error_weight * error_rate"""
        calc = AccuracyRewardCalculator(success_weight=1.0, error_weight=0.5)

        # Perfect result
        result = {"accuracy": 1.0, "error_rate": 0.0}
        reward = calc.calculate_reward(result)
        expected = 1.0 * 1.0 - 0.5 * 0.0  # = 1.0
        assert abs(reward - expected) < 0.001

        # Good result
        result = {"accuracy": 0.9, "error_rate": 0.1}
        reward = calc.calculate_reward(result)
        expected = 1.0 * 0.9 - 0.5 * 0.1  # = 0.85
        assert abs(reward - expected) < 0.001

        # Poor result
        result = {"accuracy": 0.3, "error_rate": 0.7}
        reward = calc.calculate_reward(result)
        expected = 1.0 * 0.3 - 0.5 * 0.7  # = -0.05
        assert abs(reward - expected) < 0.001

    def test_missing_keys(self):
        """Test error when required keys missing"""
        calc = AccuracyRewardCalculator()

        # Missing accuracy
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"error_rate": 0.1})

        # Missing error_rate
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"accuracy": 0.9})

        # Missing both
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({})

    def test_value_range_validation(self):
        """Test that accuracy and error_rate must be in [0, 1]"""
        calc = AccuracyRewardCalculator()

        # Invalid accuracy
        with pytest.raises(ValueError, match="Accuracy must be in"):
            calc.calculate_reward({"accuracy": 1.5, "error_rate": 0.1})

        with pytest.raises(ValueError, match="Accuracy must be in"):
            calc.calculate_reward({"accuracy": -0.1, "error_rate": 0.1})

        # Invalid error_rate
        with pytest.raises(ValueError, match="Error rate must be in"):
            calc.calculate_reward({"accuracy": 0.9, "error_rate": 1.5})

        with pytest.raises(ValueError, match="Error rate must be in"):
            calc.calculate_reward({"accuracy": 0.9, "error_rate": -0.1})

    def test_reward_clipping(self):
        """Test that rewards are clipped to [-1, 1]"""
        # High success weight to test upper bound
        calc = AccuracyRewardCalculator(success_weight=5.0, error_weight=0.0)
        result = {"accuracy": 1.0, "error_rate": 0.0}
        reward = calc.calculate_reward(result)
        assert reward <= 1.0

        # High error weight to test lower bound
        calc = AccuracyRewardCalculator(success_weight=0.0, error_weight=5.0)
        result = {"accuracy": 0.0, "error_rate": 1.0}
        reward = calc.calculate_reward(result)
        assert reward >= -1.0


class TestClassificationRewardCalculator:
    """Test ClassificationRewardCalculator"""

    def test_initialization_default_params(self):
        """Test default initialization"""
        calc = ClassificationRewardCalculator()

        assert calc.precision_weight == 0.5
        assert calc.recall_weight == 0.5
        assert calc.f1_weight == 1.0
        assert calc.use_f1 is False

    def test_initialization_custom_params(self):
        """Test custom initialization"""
        calc = ClassificationRewardCalculator(
            precision_weight=0.6,
            recall_weight=0.4,
            f1_weight=0.9,
            use_f1=True
        )

        assert calc.precision_weight == 0.6
        assert calc.recall_weight == 0.4
        assert calc.f1_weight == 0.9
        assert calc.use_f1 is True

    def test_precision_recall_mode(self):
        """Test reward calculation using precision and recall"""
        calc = ClassificationRewardCalculator(
            precision_weight=0.5,
            recall_weight=0.5,
            use_f1=False
        )

        result = {"precision": 0.8, "recall": 0.9}
        reward = calc.calculate_reward(result)
        expected = 0.5 * 0.8 + 0.5 * 0.9  # = 0.85
        assert abs(reward - expected) < 0.001

    def test_f1_mode(self):
        """Test reward calculation using F1-score"""
        calc = ClassificationRewardCalculator(
            f1_weight=1.0,
            use_f1=True
        )

        result = {"f1_score": 0.85, "precision": 0.8, "recall": 0.9}
        reward = calc.calculate_reward(result)
        expected = 1.0 * 0.85  # = 0.85
        assert abs(reward - expected) < 0.001

    def test_missing_keys_precision_recall_mode(self):
        """Test error when keys missing in precision/recall mode"""
        calc = ClassificationRewardCalculator(use_f1=False)

        # Missing precision
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"recall": 0.9})

        # Missing recall
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"precision": 0.8})

    def test_missing_keys_f1_mode(self):
        """Test error when keys missing in F1 mode"""
        calc = ClassificationRewardCalculator(use_f1=True)

        # Missing f1_score
        with pytest.raises(ValueError, match="Missing required keys"):
            calc.calculate_reward({"precision": 0.8, "recall": 0.9})

    def test_value_range_validation(self):
        """Test that metrics must be in [0, 1]"""
        calc_pr = ClassificationRewardCalculator(use_f1=False)

        # Invalid precision
        with pytest.raises(ValueError, match="Precision must be in"):
            calc_pr.calculate_reward({"precision": 1.5, "recall": 0.8})

        # Invalid recall
        with pytest.raises(ValueError, match="Recall must be in"):
            calc_pr.calculate_reward({"precision": 0.8, "recall": -0.1})

        calc_f1 = ClassificationRewardCalculator(use_f1=True)

        # Invalid F1
        with pytest.raises(ValueError, match="F1-score must be in"):
            calc_f1.calculate_reward({"f1_score": 1.5})

    def test_reward_clipping(self):
        """Test that rewards are clipped to [-1, 1]"""
        # High weights to test clipping
        calc = ClassificationRewardCalculator(
            precision_weight=5.0,
            recall_weight=5.0,
            use_f1=False
        )
        result = {"precision": 1.0, "recall": 1.0}
        reward = calc.calculate_reward(result)
        assert reward <= 1.0


class TestCustomRewardCalculator:
    """Test CustomRewardCalculator"""

    def test_initialization(self):
        """Test initialization with custom function"""

        def my_reward_fn(result):
            return 1.0

        calc = CustomRewardCalculator(reward_fn=my_reward_fn)
        assert calc.reward_fn == my_reward_fn

    def test_initialization_requires_callable(self):
        """Test that reward_fn must be callable"""
        with pytest.raises(ValueError, match="must be a callable"):
            CustomRewardCalculator(reward_fn="not_a_function")

        with pytest.raises(ValueError, match="must be a callable"):
            CustomRewardCalculator(reward_fn=123)

    def test_custom_reward_function(self):
        """Test reward calculation with custom function"""

        def profit_reward(result):
            profit = result.get("profit", 0)
            return 1.0 if profit > 100 else -1.0

        calc = CustomRewardCalculator(reward_fn=profit_reward)

        # High profit
        result = {"profit": 150}
        reward = calc.calculate_reward(result)
        assert reward == 1.0

        # Low profit
        result = {"profit": 50}
        reward = calc.calculate_reward(result)
        assert reward == -1.0

    def test_complex_custom_function(self):
        """Test complex custom reward logic"""

        def complex_reward(result):
            # Multi-factor reward
            accuracy = result.get("accuracy", 0.0)
            speed = result.get("speed", 0.0)
            cost = result.get("cost", 1.0)

            # Weighted combination
            reward = 0.5 * accuracy + 0.3 * speed - 0.2 * cost
            return reward

        calc = CustomRewardCalculator(reward_fn=complex_reward)

        result = {
            "accuracy": 0.9,
            "speed": 0.8,
            "cost": 0.3
        }
        reward = calc.calculate_reward(result)
        expected = 0.5 * 0.9 + 0.3 * 0.8 - 0.2 * 0.3  # = 0.63
        assert abs(reward - expected) < 0.001

    def test_reward_clipping(self):
        """Test that rewards from custom functions are clipped"""

        def extreme_reward(result):
            # Returns value > 1
            return 10.0

        calc = CustomRewardCalculator(reward_fn=extreme_reward)
        reward = calc.calculate_reward({})
        assert reward == 1.0  # Clipped

        def negative_extreme_reward(result):
            # Returns value < -1
            return -10.0

        calc = CustomRewardCalculator(reward_fn=negative_extreme_reward)
        reward = calc.calculate_reward({})
        assert reward == -1.0  # Clipped

    def test_custom_function_exception_handling(self):
        """Test error handling when custom function raises exception"""

        def buggy_reward(result):
            # Will raise KeyError
            return result["missing_key"]

        calc = CustomRewardCalculator(reward_fn=buggy_reward)

        with pytest.raises(Exception):
            calc.calculate_reward({})


class TestRewardCalculatorFactory:
    """Test create_reward_calculator() factory function"""

    def test_create_accuracy_calculator(self):
        """Test creating AccuracyRewardCalculator via factory"""
        calc = create_reward_calculator(
            "accuracy",
            success_weight=1.0,
            error_weight=0.5
        )

        assert isinstance(calc, AccuracyRewardCalculator)
        assert calc.success_weight == 1.0
        assert calc.error_weight == 0.5

    def test_create_classification_calculator(self):
        """Test creating ClassificationRewardCalculator via factory"""
        calc = create_reward_calculator(
            "classification",
            precision_weight=0.6,
            recall_weight=0.4,
            use_f1=False
        )

        assert isinstance(calc, ClassificationRewardCalculator)
        assert calc.precision_weight == 0.6
        assert calc.recall_weight == 0.4
        assert calc.use_f1 is False

    def test_create_custom_calculator(self):
        """Test creating CustomRewardCalculator via factory"""

        def my_fn(result):
            return 1.0

        calc = create_reward_calculator("custom", reward_fn=my_fn)

        assert isinstance(calc, CustomRewardCalculator)
        assert calc.reward_fn == my_fn

    def test_unknown_calculator_type(self):
        """Test error when calculator type unknown"""
        with pytest.raises(ValueError, match="Unknown calculator type"):
            create_reward_calculator("unknown_type")

    def test_factory_default_type(self):
        """Test factory with default type (accuracy)"""
        calc = create_reward_calculator()
        assert isinstance(calc, AccuracyRewardCalculator)


class TestRewardCalculatorIntegration:
    """Integration tests for reward calculators"""

    def test_all_calculators_implement_base_class(self):
        """Test that all calculators implement BaseRuntimeRewardCalculator"""
        calc_acc = AccuracyRewardCalculator()
        calc_class = ClassificationRewardCalculator()
        calc_custom = CustomRewardCalculator(reward_fn=lambda r: 1.0)

        assert isinstance(calc_acc, BaseRuntimeRewardCalculator)
        assert isinstance(calc_class, BaseRuntimeRewardCalculator)
        assert isinstance(calc_custom, BaseRuntimeRewardCalculator)

    def test_all_calculators_return_float(self):
        """Test that all calculators return float rewards"""
        calc_acc = AccuracyRewardCalculator()
        reward = calc_acc.calculate_reward({"accuracy": 0.9, "error_rate": 0.1})
        assert isinstance(reward, float)

        calc_class = ClassificationRewardCalculator(use_f1=True)
        reward = calc_class.calculate_reward({"f1_score": 0.85})
        assert isinstance(reward, float)

        calc_custom = CustomRewardCalculator(reward_fn=lambda r: 0.5)
        reward = calc_custom.calculate_reward({})
        assert isinstance(reward, float)

    def test_reward_normalization_across_calculators(self):
        """Test that all calculators normalize rewards to [-1, 1]"""

        # Accuracy
        calc = AccuracyRewardCalculator()
        for _ in range(10):
            reward = calc.calculate_reward({
                "accuracy": 0.5,
                "error_rate": 0.5
            })
            assert -1.0 <= reward <= 1.0

        # Classification
        calc = ClassificationRewardCalculator(use_f1=False)
        reward = calc.calculate_reward({"precision": 1.0, "recall": 1.0})
        assert -1.0 <= reward <= 1.0

        # Custom
        calc = CustomRewardCalculator(reward_fn=lambda r: 0.75)
        reward = calc.calculate_reward({})
        assert -1.0 <= reward <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

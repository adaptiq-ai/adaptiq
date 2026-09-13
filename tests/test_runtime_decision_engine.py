#!/usr/bin/env python3
"""
Tests for RuntimeDecisionEngine

Validates:
1. Engine initialization with components
2. Action registration
3. State building from context
4. Decision making (epsilon-greedy)
5. Q-table updates after execution
6. Save/load functionality
7. Epsilon management
8. Full decision-update cycle
"""

import os
import tempfile

import pytest

from adaptiq.core.entities.q_table import QTableAction, QTableState
from adaptiq.core.runtime_rl.runtime_decision_engine import (
    RuntimeDecision,
    RuntimeDecisionEngine,
)
from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
from adaptiq.core.runtime_rl.runtime_rewards import (
    AccuracyRewardCalculator,
    CustomRewardCalculator,
)


class TestRuntimeDecisionEngineInit:
    """Test initialization and component integration"""

    def test_initialization(self):
        """Test engine initialization with components"""
        q_manager = RuntimeQTableManager()
        reward_calc = AccuracyRewardCalculator()

        engine = RuntimeDecisionEngine(
            q_table_manager=q_manager,
            reward_calculator=reward_calc
        )

        assert engine.q_manager == q_manager
        assert engine.reward_calc == reward_calc
        assert engine.available_actions == []

    def test_repr(self):
        """Test string representation"""
        q_manager = RuntimeQTableManager()
        reward_calc = AccuracyRewardCalculator()
        engine = RuntimeDecisionEngine(q_manager, reward_calc)

        repr_str = repr(engine)
        assert "RuntimeDecisionEngine" in repr_str
        assert "actions=0" in repr_str
        assert "alpha=0.1" in repr_str
        assert "gamma=0.9" in repr_str
        assert "epsilon=0.1" in repr_str


class TestActionRegistration:
    """Test action registration"""

    def setup_method(self):
        """Setup engine for each test"""
        self.engine = RuntimeDecisionEngine(
            RuntimeQTableManager(),
            AccuracyRewardCalculator()
        )

    def test_register_actions(self):
        """Test registering actions"""
        actions = [
            QTableAction(action="method_a"),
            QTableAction(action="method_b"),
            QTableAction(action="method_c")
        ]

        self.engine.register_actions(actions)

        assert len(self.engine.available_actions) == 3
        assert self.engine.available_actions == actions

    def test_register_empty_actions_fails(self):
        """Test that registering empty list raises error"""
        with pytest.raises(ValueError, match="Cannot register empty"):
            self.engine.register_actions([])

    def test_register_actions_overwrites_previous(self):
        """Test that re-registering actions overwrites previous"""
        actions1 = [QTableAction(action="a"), QTableAction(action="b")]
        actions2 = [QTableAction(action="c")]

        self.engine.register_actions(actions1)
        assert len(self.engine.available_actions) == 2

        self.engine.register_actions(actions2)
        assert len(self.engine.available_actions) == 1
        assert self.engine.available_actions[0].action == "c"


class TestStateBuildinging:
    """Test state building from context"""

    def setup_method(self):
        """Setup engine for each test"""
        self.engine = RuntimeDecisionEngine(
            RuntimeQTableManager(),
            AccuracyRewardCalculator()
        )

    def test_build_state_valid_context(self):
        """Test building state from valid context"""
        context = {
            "subtask": "price_estimation",
            "last_action": "linear_model",
            "last_outcome": "success",
            "key_context": "small_dataset"
        }

        state = self.engine.build_state(context)

        assert isinstance(state, QTableState)
        assert state.current_subtask == "price_estimation"
        assert state.last_action_taken == "linear_model"
        assert state.last_outcome == "success"
        assert state.key_context == "small_dataset"

    def test_build_state_missing_keys(self):
        """Test error when context missing required keys"""
        # Missing subtask
        context = {
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "ctx"
        }

        with pytest.raises(ValueError, match="Missing required keys"):
            self.engine.build_state(context)

        # Missing multiple keys
        with pytest.raises(ValueError, match="Missing required keys"):
            self.engine.build_state({"subtask": "test"})

    def test_build_state_initial_context(self):
        """Test building initial state (no previous action)"""
        context = {
            "subtask": "classification",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "balanced_data"
        }

        state = self.engine.build_state(context)

        assert state.last_action_taken == "None"
        assert state.last_outcome == "None"


class TestDecisionMaking:
    """Test decision-making logic"""

    def setup_method(self):
        """Setup engine with actions for each test"""
        self.engine = RuntimeDecisionEngine(
            RuntimeQTableManager(epsilon=0.1),
            AccuracyRewardCalculator()
        )

        self.actions = [
            QTableAction(action="method_a"),
            QTableAction(action="method_b"),
            QTableAction(action="method_c")
        ]
        self.engine.register_actions(self.actions)

        self.context = {
            "subtask": "test_task",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "test_ctx"
        }

    def test_decide_returns_runtime_decision(self):
        """Test that decide() returns RuntimeDecision"""
        decision = self.engine.decide(self.context)

        assert isinstance(decision, RuntimeDecision)
        assert decision.action in self.actions
        assert isinstance(decision.explored, bool)
        assert isinstance(decision.state, QTableState)
        assert isinstance(decision.q_value, float)

    def test_decide_without_actions_fails(self):
        """Test that decide() fails if no actions registered"""
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(),
            AccuracyRewardCalculator()
        )

        with pytest.raises(ValueError, match="No actions registered"):
            engine.decide(self.context)

    def test_decide_with_invalid_context(self):
        """Test that decide() fails with invalid context"""
        invalid_context = {"subtask": "test"}

        with pytest.raises(ValueError, match="Missing required keys"):
            self.engine.decide(invalid_context)

    def test_decide_exploitation(self):
        """Test that exploitation selects best action"""
        # Set epsilon=0 for pure exploitation
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(epsilon=0.0),
            AccuracyRewardCalculator()
        )
        engine.register_actions(self.actions)

        # Set Q-values to make method_b best
        state = engine.build_state(self.context)
        engine.q_manager.set_q_value(state, self.actions[0], 0.3)
        engine.q_manager.set_q_value(state, self.actions[1], 0.9)  # Best
        engine.q_manager.set_q_value(state, self.actions[2], 0.5)

        # Should always select method_b
        for _ in range(5):
            decision = engine.decide(self.context)
            assert decision.action.action == "method_b"
            assert decision.explored is False

    def test_decide_exploration(self):
        """Test that exploration occurs with epsilon > 0"""
        # Set epsilon=1.0 for pure exploration
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(epsilon=1.0),
            AccuracyRewardCalculator()
        )
        engine.register_actions(self.actions)

        # Even with best Q-value, should still explore
        state = engine.build_state(self.context)
        engine.q_manager.set_q_value(state, self.actions[1], 0.9)

        explored_count = 0
        for _ in range(10):
            decision = engine.decide(self.context)
            if decision.explored:
                explored_count += 1

        # All should be exploration
        assert explored_count == 10


class TestQTableUpdate:
    """Test Q-table update after action execution"""

    def setup_method(self):
        """Setup engine for each test"""
        self.engine = RuntimeDecisionEngine(
            RuntimeQTableManager(alpha=0.1, gamma=0.9, epsilon=0.0),
            AccuracyRewardCalculator()
        )

        self.actions = [QTableAction(action="method_a")]
        self.engine.register_actions(self.actions)

        self.context = {
            "subtask": "test",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "ctx"
        }

    def test_update_calculates_reward(self):
        """Test that update() calculates reward correctly"""
        decision = self.engine.decide(self.context)

        result = {"accuracy": 0.9, "error_rate": 0.1}
        next_context = {
            "subtask": "test",
            "last_action": decision.action.action,
            "last_outcome": "success",
            "key_context": "ctx"
        }

        new_q = self.engine.update(decision, result, next_context)

        # Reward should be: 1.0 * 0.9 - 0.5 * 0.1 = 0.85
        # Q-update: 0 + 0.1 * (0.85 + 0.9 * 0 - 0) = 0.085
        assert abs(new_q - 0.085) < 0.001

    def test_update_modifies_q_table(self):
        """Test that update() modifies Q-table"""
        decision = self.engine.decide(self.context)

        # Initial Q-value should be 0
        initial_q = self.engine.q_manager.Q(decision.state, decision.action)
        assert initial_q == 0.0

        # Update with positive result
        result = {"accuracy": 1.0, "error_rate": 0.0}
        next_context = {
            "subtask": "test",
            "last_action": decision.action.action,
            "last_outcome": "success",
            "key_context": "ctx"
        }

        self.engine.update(decision, result, next_context)

        # Q-value should have increased
        updated_q = self.engine.q_manager.Q(decision.state, decision.action)
        assert updated_q > initial_q

    def test_update_with_negative_reward(self):
        """Test update with poor result (negative reward)"""
        decision = self.engine.decide(self.context)

        # Poor result
        result = {"accuracy": 0.1, "error_rate": 0.9}
        next_context = {
            "subtask": "test",
            "last_action": decision.action.action,
            "last_outcome": "failure",
            "key_context": "ctx"
        }

        new_q = self.engine.update(decision, result, next_context)

        # Reward should be negative: 1.0 * 0.1 - 0.5 * 0.9 = -0.35
        # Q-update: 0 + 0.1 * (-0.35 + 0 - 0) = -0.035
        assert new_q < 0.0


class TestSaveLoad:
    """Test save/load functionality"""

    def setup_method(self):
        """Setup temp file for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def teardown_method(self):
        """Cleanup temp file"""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_save_load_q_table(self):
        """Test saving and loading Q-table"""
        # Create engine and make some decisions
        engine1 = RuntimeDecisionEngine(
            RuntimeQTableManager(file_path=self.file_path, epsilon=0.0),
            AccuracyRewardCalculator()
        )

        actions = [QTableAction(action="method_a")]
        engine1.register_actions(actions)

        context = {
            "subtask": "test",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "ctx"
        }

        # Make decision and update
        decision = engine1.decide(context)
        result = {"accuracy": 0.9, "error_rate": 0.1}
        next_context = {**context, "last_action": "method_a", "last_outcome": "success"}
        engine1.update(decision, result, next_context)

        # Save Q-table
        success = engine1.save_q_table(prefix_version="test")
        assert success

        # Load into new engine
        engine2 = RuntimeDecisionEngine(
            RuntimeQTableManager(file_path=self.file_path),
            AccuracyRewardCalculator()
        )
        engine2.register_actions(actions)

        success = engine2.load_q_table()
        assert success

        # Verify Q-values match
        q1 = engine1.q_manager.Q(decision.state, decision.action)
        q2 = engine2.q_manager.Q(decision.state, decision.action)
        assert abs(q1 - q2) < 0.001


class TestEpsilonManagement:
    """Test epsilon getter/setter"""

    def setup_method(self):
        """Setup engine for each test"""
        self.engine = RuntimeDecisionEngine(
            RuntimeQTableManager(epsilon=0.1),
            AccuracyRewardCalculator()
        )

    def test_get_epsilon(self):
        """Test getting epsilon value"""
        epsilon = self.engine.get_epsilon()
        assert epsilon == 0.1

    def test_set_epsilon(self):
        """Test setting epsilon value"""
        self.engine.set_epsilon(0.05)
        assert self.engine.get_epsilon() == 0.05

    def test_epsilon_decay_strategy(self):
        """Test epsilon decay over time"""
        # Start with high exploration
        self.engine.set_epsilon(0.3)
        assert self.engine.get_epsilon() == 0.3

        # Gradual decay
        self.engine.set_epsilon(0.2)
        assert self.engine.get_epsilon() == 0.2

        self.engine.set_epsilon(0.1)
        assert self.engine.get_epsilon() == 0.1

        # Final low exploration
        self.engine.set_epsilon(0.05)
        assert self.engine.get_epsilon() == 0.05


class TestHyperparameters:
    """Test hyperparameter access"""

    def test_get_hyperparameters(self):
        """Test getting all hyperparameters"""
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(alpha=0.1, gamma=0.9, epsilon=0.15),
            AccuracyRewardCalculator()
        )

        params = engine.get_hyperparameters()

        assert params["alpha"] == 0.1
        assert params["gamma"] == 0.9
        assert params["epsilon"] == 0.15


class TestFullCycle:
    """Integration test: full decision-update cycle"""

    def test_complete_decision_cycle(self):
        """Test complete cycle: decide → execute → update"""
        # Setup
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(alpha=0.1, gamma=0.9, epsilon=0.1),
            AccuracyRewardCalculator()
        )

        actions = [
            QTableAction(action="linear_regression"),
            QTableAction(action="neural_network")
        ]
        engine.register_actions(actions)

        # Step 1: Make decision
        context = {
            "subtask": "price_prediction",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "small_dataset"
        }

        decision = engine.decide(context)

        assert decision.action in actions
        assert isinstance(decision.q_value, float)

        # Step 2: Simulate action execution
        # (In real code, user would execute the action here)
        execution_result = {"accuracy": 0.85, "error_rate": 0.15}

        # Step 3: Update Q-table
        next_context = {
            "subtask": "price_prediction",
            "last_action": decision.action.action,
            "last_outcome": "success",
            "key_context": "small_dataset"
        }

        new_q = engine.update(decision, execution_result, next_context)

        # Verify update occurred
        assert isinstance(new_q, float)
        assert new_q != decision.q_value  # Q-value should have changed

    def test_multiple_decision_cycles(self):
        """Test multiple sequential decision-update cycles"""
        engine = RuntimeDecisionEngine(
            RuntimeQTableManager(alpha=0.1, gamma=0.9, epsilon=0.0),
            AccuracyRewardCalculator()
        )

        actions = [QTableAction(action="method_a"), QTableAction(action="method_b")]
        engine.register_actions(actions)

        context = {
            "subtask": "task",
            "last_action": "None",
            "last_outcome": "None",
            "key_context": "ctx"
        }

        q_values = []

        # Run 5 cycles
        for i in range(5):
            # Decide
            decision = engine.decide(context)

            # Execute (simulated)
            result = {"accuracy": 0.8, "error_rate": 0.2}

            # Update
            next_context = {
                "subtask": "task",
                "last_action": decision.action.action,
                "last_outcome": "success",
                "key_context": "ctx"
            }

            new_q = engine.update(decision, result, next_context)
            q_values.append(new_q)

            # Update context for next iteration
            context = next_context

        # Q-values should be learning (changing over time)
        assert len(set(q_values)) > 1  # Not all the same


class TestRuntimeDecisionDataclass:
    """Test RuntimeDecision dataclass"""

    def test_runtime_decision_creation(self):
        """Test creating RuntimeDecision"""
        state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        action = QTableAction(action="test_action")

        decision = RuntimeDecision(
            action=action,
            explored=True,
            state=state,
            q_value=0.75
        )

        assert decision.action == action
        assert decision.explored is True
        assert decision.state == state
        assert decision.q_value == 0.75

    def test_runtime_decision_repr(self):
        """Test RuntimeDecision string representation"""
        state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        action = QTableAction(action="method_a")

        decision = RuntimeDecision(
            action=action,
            explored=False,
            state=state,
            q_value=0.85
        )

        repr_str = repr(decision)
        assert "RuntimeDecision" in repr_str
        assert "method_a" in repr_str
        assert "EXPLOITATION" in repr_str
        assert "0.85" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

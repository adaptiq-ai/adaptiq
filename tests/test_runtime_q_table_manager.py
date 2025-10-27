#!/usr/bin/env python3
"""
Tests for RuntimeQTableManager

Validates:
1. Inheritance from QTableManager
2. Epsilon-greedy action selection
3. Update policy inherited correctly
4. Save/load functionality
5. Separate storage from offline Q-table
"""

import os
import tempfile
import pytest
from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
from adaptiq.core.q_table.q_table_manager import QTableManager
from adaptiq.core.entities.q_table import QTableState, QTableAction


class TestRuntimeQTableManagerInheritance:
    """Test inheritance and basic functionality"""

    def test_inheritance(self):
        """Test that RuntimeQTableManager inherits from QTableManager"""
        manager = RuntimeQTableManager()
        assert isinstance(manager, QTableManager)
        assert issubclass(RuntimeQTableManager, QTableManager)

    def test_initialization_default_params(self):
        """Test default initialization parameters"""
        manager = RuntimeQTableManager()

        assert manager.alpha == 0.1  # Runtime default
        assert manager.gamma == 0.9  # Runtime default
        assert manager.epsilon == 0.1  # Runtime default
        assert "runtime_q_table.json" in manager.file_path

    def test_initialization_custom_params(self):
        """Test custom initialization parameters"""
        manager = RuntimeQTableManager(
            file_path="custom_path.json",
            alpha=0.2,
            gamma=0.85,
            epsilon=0.15
        )

        assert manager.alpha == 0.2
        assert manager.gamma == 0.85
        assert manager.epsilon == 0.15
        assert manager.file_path == "custom_path.json"

    def test_epsilon_validation(self):
        """Test epsilon must be in [0, 1]"""
        # Valid epsilon
        manager = RuntimeQTableManager(epsilon=0.0)
        assert manager.epsilon == 0.0

        manager = RuntimeQTableManager(epsilon=1.0)
        assert manager.epsilon == 1.0

        # Invalid epsilon
        with pytest.raises(ValueError, match="Epsilon must be in"):
            RuntimeQTableManager(epsilon=-0.1)

        with pytest.raises(ValueError, match="Epsilon must be in"):
            RuntimeQTableManager(epsilon=1.1)


class TestEpsilonGreedySelection:
    """Test epsilon-greedy action selection"""

    def setup_method(self):
        """Setup test state and actions"""
        self.state = QTableState(
            current_subtask="test_task",
            last_action_taken="None",
            last_outcome="None",
            key_context="test_context"
        )
        self.actions = [
            QTableAction(action="action_a"),
            QTableAction(action="action_b"),
            QTableAction(action="action_c")
        ]

    def test_epsilon_greedy_with_epsilon_zero(self):
        """Test epsilon=0 always exploits (never explores)"""
        manager = RuntimeQTableManager(epsilon=0.0)

        # Set Q-values to make action_b best
        manager.set_q_value(self.state, self.actions[0], 0.5)
        manager.set_q_value(self.state, self.actions[1], 0.9)  # Best
        manager.set_q_value(self.state, self.actions[2], 0.3)

        # Test multiple times - should always select action_b (exploitation)
        for _ in range(10):
            action, explored = manager.select_action_epsilon_greedy(
                self.state, self.actions
            )
            assert action.action == "action_b"
            assert explored is False

    def test_epsilon_greedy_with_epsilon_one(self):
        """Test epsilon=1.0 always explores (never exploits)"""
        manager = RuntimeQTableManager(epsilon=1.0)

        # Set Q-values to make action_b best
        manager.set_q_value(self.state, self.actions[0], 0.5)
        manager.set_q_value(self.state, self.actions[1], 0.9)  # Best
        manager.set_q_value(self.state, self.actions[2], 0.3)

        # Test multiple times - should explore (random selection)
        exploration_count = 0
        for _ in range(10):
            action, explored = manager.select_action_epsilon_greedy(
                self.state, self.actions
            )
            assert explored is True
            exploration_count += 1

        assert exploration_count == 10

    def test_epsilon_greedy_exploration_distribution(self):
        """Test epsilon=0.5 explores ~50% of the time"""
        manager = RuntimeQTableManager(epsilon=0.5)

        # Set Q-values
        manager.set_q_value(self.state, self.actions[0], 0.5)
        manager.set_q_value(self.state, self.actions[1], 0.9)  # Best
        manager.set_q_value(self.state, self.actions[2], 0.3)

        # Run many trials
        exploration_count = 0
        exploitation_count = 0
        trials = 100

        for _ in range(trials):
            action, explored = manager.select_action_epsilon_greedy(
                self.state, self.actions
            )
            if explored:
                exploration_count += 1
            else:
                exploitation_count += 1
                # Exploitation should always select action_b
                assert action.action == "action_b"

        # Check distribution (should be ~50/50 with some variance)
        exploration_ratio = exploration_count / trials
        assert 0.3 < exploration_ratio < 0.7  # Allow variance

    def test_select_action_no_actions(self):
        """Test error when no actions provided"""
        manager = RuntimeQTableManager()

        with pytest.raises(ValueError, match="No available actions"):
            manager.select_action_epsilon_greedy(self.state, [])

    def test_select_action_single_action(self):
        """Test with single action (should always select it)"""
        manager = RuntimeQTableManager(epsilon=0.5)
        single_action = [QTableAction(action="only_action")]

        action, explored = manager.select_action_epsilon_greedy(
            self.state, single_action
        )
        assert action.action == "only_action"


class TestUpdatePolicyInherited:
    """Test that update_policy is correctly inherited"""

    def setup_method(self):
        """Setup test components"""
        self.manager = RuntimeQTableManager()
        self.state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        self.action = QTableAction(action="test_action")
        self.next_state = QTableState(
            current_subtask="test",
            last_action_taken="test_action",
            last_outcome="success",
            key_context="ctx"
        )
        self.actions = [self.action]

    def test_update_policy_exists(self):
        """Test update_policy method exists"""
        assert hasattr(self.manager, "update_policy")
        assert callable(self.manager.update_policy)

    def test_update_policy_formula(self):
        """Test Q-learning formula: Q(s,a) ← Q(s,a) + α(R + γ·max Q(s',a') - Q(s,a))"""
        # Initial Q-value should be 0
        initial_q = self.manager.Q(self.state, self.action)
        assert initial_q == 0.0

        # Update with reward=1.0
        reward = 1.0
        new_q = self.manager.update_policy(
            self.state, self.action, reward, self.next_state, self.actions
        )

        # Check formula: Q = 0 + 0.1 * (1.0 + 0.9 * 0 - 0) = 0.1
        expected_q = 0.0 + 0.1 * (1.0 + 0.9 * 0.0 - 0.0)
        assert abs(new_q - expected_q) < 0.001

        # Verify Q-table updated
        assert self.manager.Q(self.state, self.action) == new_q

    def test_update_policy_multiple_updates(self):
        """Test Q-values converge with multiple updates"""
        reward = 1.0

        # Multiple updates
        q_values = []
        for _ in range(10):
            q_val = self.manager.update_policy(
                self.state, self.action, reward, self.next_state, self.actions
            )
            q_values.append(q_val)

        # Q-values should increase
        assert q_values[-1] > q_values[0]

        # Should converge (difference between last two small)
        assert abs(q_values[-1] - q_values[-2]) < 0.1


class TestSaveLoadFunctionality:
    """Test save/load Q-table functionality"""

    def setup_method(self):
        """Setup temporary file for testing"""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        self.temp_file.close()
        self.file_path = self.temp_file.name

    def teardown_method(self):
        """Cleanup temporary file"""
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_save_load_q_table(self):
        """Test save and load Q-table"""
        # Create manager and populate Q-table
        manager1 = RuntimeQTableManager(file_path=self.file_path)

        state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        action = QTableAction(action="test_action")

        manager1.set_q_value(state, action, 0.75)

        # Save
        success = manager1.save_q_table(prefix_version="runtime_test")
        assert success

        # Load into new manager
        manager2 = RuntimeQTableManager(file_path=self.file_path)
        success = manager2.load_q_table()
        assert success

        # Verify Q-value preserved
        assert manager2.Q(state, action) == 0.75

    def test_separate_storage_from_offline(self):
        """Test that runtime Q-table uses separate storage path"""
        runtime_manager = RuntimeQTableManager()
        offline_manager = QTableManager(
            file_path="storage/qtables/adaptiq_q_table.json",
            alpha=0.8,
            gamma=0.8
        )

        # Different storage paths
        assert runtime_manager.file_path != offline_manager.file_path
        assert "runtime" in runtime_manager.file_path.lower()
        assert "adaptiq" in offline_manager.file_path.lower()


class TestEpsilonManagement:
    """Test epsilon getter/setter"""

    def test_set_epsilon(self):
        """Test set_epsilon method"""
        manager = RuntimeQTableManager(epsilon=0.1)
        assert manager.get_epsilon() == 0.1

        manager.set_epsilon(0.05)
        assert manager.get_epsilon() == 0.05

        manager.set_epsilon(0.0)
        assert manager.get_epsilon() == 0.0

    def test_set_epsilon_validation(self):
        """Test epsilon validation in set_epsilon"""
        manager = RuntimeQTableManager()

        # Valid values
        manager.set_epsilon(0.0)
        manager.set_epsilon(1.0)
        manager.set_epsilon(0.5)

        # Invalid values
        with pytest.raises(ValueError):
            manager.set_epsilon(-0.1)

        with pytest.raises(ValueError):
            manager.set_epsilon(1.5)

    def test_epsilon_decay_strategy(self):
        """Test epsilon decay over time (simulated)"""
        manager = RuntimeQTableManager(epsilon=1.0)

        # Simulate epsilon decay
        decay_rate = 0.9
        for _ in range(10):
            current = manager.get_epsilon()
            manager.set_epsilon(current * decay_rate)

        # Epsilon should have decayed significantly
        assert manager.get_epsilon() < 0.5


class TestRuntimeVsOfflineComparison:
    """Test differences between Runtime and Offline Q-Table Managers"""

    def test_hyperparameter_differences(self):
        """Test that runtime uses different hyperparameters than offline"""
        runtime = RuntimeQTableManager()
        offline = QTableManager(
            file_path="offline.json",
            alpha=0.8,  # Offline default
            gamma=0.8   # Offline default
        )

        # Alpha comparison
        assert runtime.alpha == 0.1  # Online learning
        assert offline.alpha == 0.8  # Offline learning
        assert runtime.alpha < offline.alpha

        # Gamma comparison
        assert runtime.gamma == 0.9  # Long-term
        assert offline.gamma == 0.8  # Standard
        assert runtime.gamma > offline.gamma

    def test_exploration_capability(self):
        """Test that runtime has exploration, offline does not"""
        runtime = RuntimeQTableManager()

        # Runtime has epsilon-greedy
        assert hasattr(runtime, "select_action_epsilon_greedy")
        assert hasattr(runtime, "epsilon")
        assert hasattr(runtime, "set_epsilon")

        # Offline uses get_best_action (pure exploitation)
        offline = QTableManager(file_path="offline.json")
        assert hasattr(offline, "get_best_action")
        assert not hasattr(offline, "epsilon")


class TestRepr:
    """Test string representation"""

    def test_repr(self):
        """Test __repr__ method"""
        manager = RuntimeQTableManager()
        repr_str = repr(manager)

        assert "RuntimeQTableManager" in repr_str
        assert "alpha=0.1" in repr_str
        assert "gamma=0.9" in repr_str
        assert "epsilon=0.1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

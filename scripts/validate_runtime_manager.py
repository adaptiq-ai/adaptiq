#!/usr/bin/env python3
"""
Manual validation script for RuntimeQTableManager
This script validates all key requirements before proceeding to Component 2
"""

import os
import sys

# Add src to path
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

def test_imports():
    """Test 1: Verify imports work"""
    print("=" * 70)
    print("TEST 1: Verifying imports...")
    print("=" * 70)

    try:
        from adaptiq.core.entities.q_table import QTableAction, QTableState
        from adaptiq.core.q_table.q_table_manager import QTableManager
        from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
        print("✓ All imports successful\n")
        return True, (RuntimeQTableManager, QTableManager, QTableState, QTableAction)
    except Exception as e:
        print(f"✗ Import failed: {e}\n")
        return False, None

def test_inheritance(classes):
    """Test 2: Verify inheritance"""
    print("=" * 70)
    print("TEST 2: Verifying inheritance...")
    print("=" * 70)

    RuntimeQTableManager, QTableManager, _, _ = classes

    try:
        manager = RuntimeQTableManager()

        # Check isinstance
        if not isinstance(manager, QTableManager):
            print(f"✗ Instance check failed: {type(manager)}")
            return False

        # Check issubclass
        if not issubclass(RuntimeQTableManager, QTableManager):
            print(f"✗ Subclass check failed")
            return False

        print(f"✓ RuntimeQTableManager correctly inherits from QTableManager")
        print(f"✓ Instance check: {isinstance(manager, QTableManager)}")
        print(f"✓ Subclass check: {issubclass(RuntimeQTableManager, QTableManager)}\n")
        return True
    except Exception as e:
        print(f"✗ Inheritance test failed: {e}\n")
        return False

def test_initialization(classes):
    """Test 3: Verify initialization and hyperparameters"""
    print("=" * 70)
    print("TEST 3: Verifying initialization and hyperparameters...")
    print("=" * 70)

    RuntimeQTableManager, _, _, _ = classes

    try:
        # Default params
        manager = RuntimeQTableManager()

        assert manager.alpha == 0.1, f"Expected alpha=0.1, got {manager.alpha}"
        assert manager.gamma == 0.9, f"Expected gamma=0.9, got {manager.gamma}"
        assert manager.epsilon == 0.1, f"Expected epsilon=0.1, got {manager.epsilon}"
        assert "runtime_q_table.json" in manager.file_path

        print(f"✓ Default hyperparameters correct:")
        print(f"  - alpha: {manager.alpha} (online learning)")
        print(f"  - gamma: {manager.gamma} (long-term planning)")
        print(f"  - epsilon: {manager.epsilon} (exploration rate)")
        print(f"  - storage: {manager.file_path}")

        # Custom params
        manager2 = RuntimeQTableManager(alpha=0.2, gamma=0.85, epsilon=0.15)
        assert manager2.alpha == 0.2
        assert manager2.gamma == 0.85
        assert manager2.epsilon == 0.15

        print(f"✓ Custom parameters work correctly\n")
        return True
    except Exception as e:
        print(f"✗ Initialization test failed: {e}\n")
        return False

def test_epsilon_validation(classes):
    """Test 4: Verify epsilon validation"""
    print("=" * 70)
    print("TEST 4: Verifying epsilon validation...")
    print("=" * 70)

    RuntimeQTableManager, _, _, _ = classes

    try:
        # Valid epsilon values
        RuntimeQTableManager(epsilon=0.0)
        RuntimeQTableManager(epsilon=1.0)
        RuntimeQTableManager(epsilon=0.5)
        print("✓ Valid epsilon values (0.0, 0.5, 1.0) accepted")

        # Invalid epsilon - negative
        try:
            RuntimeQTableManager(epsilon=-0.1)
            print("✗ Negative epsilon should raise ValueError")
            return False
        except ValueError as e:
            print(f"✓ Negative epsilon rejected: {e}")

        # Invalid epsilon - > 1
        try:
            RuntimeQTableManager(epsilon=1.5)
            print("✗ Epsilon > 1 should raise ValueError")
            return False
        except ValueError as e:
            print(f"✓ Epsilon > 1 rejected: {e}\n")

        return True
    except Exception as e:
        print(f"✗ Epsilon validation test failed: {e}\n")
        return False

def test_epsilon_greedy(classes):
    """Test 5: Verify epsilon-greedy action selection"""
    print("=" * 70)
    print("TEST 5: Verifying epsilon-greedy action selection...")
    print("=" * 70)

    RuntimeQTableManager, _, QTableState, QTableAction = classes

    try:
        state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        actions = [
            QTableAction(action="action_a"),
            QTableAction(action="action_b"),
            QTableAction(action="action_c")
        ]

        # Test epsilon=0 (pure exploitation)
        manager_exploit = RuntimeQTableManager(epsilon=0.0)
        manager_exploit.set_q_value(state, actions[0], 0.5)
        manager_exploit.set_q_value(state, actions[1], 0.9)  # Best
        manager_exploit.set_q_value(state, actions[2], 0.3)

        exploit_count = 0
        for _ in range(10):
            action, explored = manager_exploit.select_action_epsilon_greedy(state, actions)
            if action.action == "action_b" and not explored:
                exploit_count += 1

        if exploit_count == 10:
            print(f"✓ Epsilon=0: Always exploits (selected best action 10/10 times)")
        else:
            print(f"✗ Epsilon=0: Should always exploit, but only {exploit_count}/10")
            return False

        # Test epsilon=1.0 (pure exploration)
        manager_explore = RuntimeQTableManager(epsilon=1.0)
        manager_explore.set_q_value(state, actions[0], 0.5)
        manager_explore.set_q_value(state, actions[1], 0.9)
        manager_explore.set_q_value(state, actions[2], 0.3)

        explore_count = 0
        for _ in range(10):
            action, explored = manager_explore.select_action_epsilon_greedy(state, actions)
            if explored:
                explore_count += 1

        if explore_count == 10:
            print(f"✓ Epsilon=1.0: Always explores (random selection 10/10 times)")
        else:
            print(f"✗ Epsilon=1.0: Should always explore, but only {explore_count}/10")
            return False

        # Test epsilon=0.5 (mixed)
        manager_mixed = RuntimeQTableManager(epsilon=0.5)
        manager_mixed.set_q_value(state, actions[0], 0.5)
        manager_mixed.set_q_value(state, actions[1], 0.9)
        manager_mixed.set_q_value(state, actions[2], 0.3)

        explore_count = 0
        exploit_count = 0
        for _ in range(100):
            action, explored = manager_mixed.select_action_epsilon_greedy(state, actions)
            if explored:
                explore_count += 1
            else:
                exploit_count += 1
                # Exploitation should always select action_b
                if action.action != "action_b":
                    print(f"✗ Exploitation selected wrong action: {action.action}")
                    return False

        exploration_ratio = explore_count / 100
        if 0.3 < exploration_ratio < 0.7:
            print(f"✓ Epsilon=0.5: Balanced exploration/exploitation ({explore_count}% explore, {exploit_count}% exploit)")
        else:
            print(f"✗ Epsilon=0.5: Expected ~50% exploration, got {exploration_ratio*100}%")
            return False

        print()
        return True
    except Exception as e:
        print(f"✗ Epsilon-greedy test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_update_policy_inherited(classes):
    """Test 6: Verify update_policy is correctly inherited"""
    print("=" * 70)
    print("TEST 6: Verifying update_policy inheritance (Bellman equation)...")
    print("=" * 70)

    RuntimeQTableManager, _, QTableState, QTableAction = classes

    try:
        manager = RuntimeQTableManager()

        state = QTableState(
            current_subtask="test",
            last_action_taken="None",
            last_outcome="None",
            key_context="ctx"
        )
        action = QTableAction(action="test_action")
        next_state = QTableState(
            current_subtask="test",
            last_action_taken="test_action",
            last_outcome="success",
            key_context="ctx"
        )
        actions = [action]

        # Initial Q-value should be 0
        initial_q = manager.Q(state, action)
        assert initial_q == 0.0, f"Expected initial Q=0, got {initial_q}"

        # Update with reward=1.0
        reward = 1.0
        new_q = manager.update_policy(state, action, reward, next_state, actions)

        # Check formula: Q = 0 + 0.1 * (1.0 + 0.9 * 0 - 0) = 0.1
        expected_q = 0.0 + 0.1 * (1.0 + 0.9 * 0.0 - 0.0)

        if abs(new_q - expected_q) < 0.001:
            print(f"✓ Bellman equation correctly inherited from QTableManager")
            print(f"  Formula: Q(s,a) ← Q(s,a) + α(R + γ·maxQ(s',a') - Q(s,a))")
            print(f"  Calculated: 0.0 + 0.1 * (1.0 + 0.9 * 0.0 - 0.0) = {new_q}")
            print(f"  Expected: {expected_q}")
        else:
            print(f"✗ Q-value calculation incorrect: got {new_q}, expected {expected_q}")
            return False

        # Verify Q-table updated
        updated_q = manager.Q(state, action)
        if updated_q == new_q:
            print(f"✓ Q-table correctly updated with new value: {updated_q}\n")
        else:
            print(f"✗ Q-table not updated correctly: {updated_q} != {new_q}")
            return False

        return True
    except Exception as e:
        print(f"✗ Update policy test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_separate_storage(classes):
    """Test 7: Verify separate storage from offline Q-table"""
    print("=" * 70)
    print("TEST 7: Verifying separate storage paths...")
    print("=" * 70)

    RuntimeQTableManager, QTableManager, _, _ = classes

    try:
        runtime_manager = RuntimeQTableManager()
        offline_manager = QTableManager(
            file_path="storage/qtables/adaptiq_q_table.json",
            alpha=0.8,
            gamma=0.8
        )

        # Different storage paths
        if runtime_manager.file_path != offline_manager.file_path:
            print(f"✓ Storage paths are different:")
            print(f"  Runtime: {runtime_manager.file_path}")
            print(f"  Offline: {offline_manager.file_path}")
        else:
            print(f"✗ Storage paths should be different")
            return False

        # Verify "runtime" in path
        if "runtime" in runtime_manager.file_path.lower():
            print(f"✓ Runtime Q-table path contains 'runtime'")
        else:
            print(f"✗ Runtime path should contain 'runtime': {runtime_manager.file_path}")
            return False

        print()
        return True
    except Exception as e:
        print(f"✗ Separate storage test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_epsilon_management(classes):
    """Test 8: Verify epsilon getter/setter"""
    print("=" * 70)
    print("TEST 8: Verifying epsilon management...")
    print("=" * 70)

    RuntimeQTableManager, _, _, _ = classes

    try:
        manager = RuntimeQTableManager(epsilon=0.1)

        # Test getter
        if manager.get_epsilon() == 0.1:
            print(f"✓ get_epsilon() returns correct value: {manager.get_epsilon()}")
        else:
            print(f"✗ get_epsilon() returned {manager.get_epsilon()}, expected 0.1")
            return False

        # Test setter
        manager.set_epsilon(0.05)
        if manager.get_epsilon() == 0.05:
            print(f"✓ set_epsilon() works: {manager.get_epsilon()}")
        else:
            print(f"✗ set_epsilon() failed")
            return False

        # Test validation in setter
        try:
            manager.set_epsilon(-0.1)
            print(f"✗ set_epsilon() should reject negative values")
            return False
        except ValueError:
            print(f"✓ set_epsilon() correctly rejects negative values")

        try:
            manager.set_epsilon(1.5)
            print(f"✗ set_epsilon() should reject values > 1")
            return False
        except ValueError:
            print(f"✓ set_epsilon() correctly rejects values > 1")

        print()
        return True
    except Exception as e:
        print(f"✗ Epsilon management test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  RuntimeQTableManager Validation Suite".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")

    results = []

    # Test 1: Imports
    success, classes = test_imports()
    results.append(("Imports", success))
    if not success:
        print("\n❌ VALIDATION FAILED: Cannot proceed without successful imports\n")
        sys.exit(1)

    # Test 2: Inheritance
    success = test_inheritance(classes)
    results.append(("Inheritance", success))

    # Test 3: Initialization
    success = test_initialization(classes)
    results.append(("Initialization", success))

    # Test 4: Epsilon validation
    success = test_epsilon_validation(classes)
    results.append(("Epsilon Validation", success))

    # Test 5: Epsilon-greedy
    success = test_epsilon_greedy(classes)
    results.append(("Epsilon-Greedy Selection", success))

    # Test 6: Update policy
    success = test_update_policy_inherited(classes)
    results.append(("Update Policy (Bellman)", success))

    # Test 7: Separate storage
    success = test_separate_storage(classes)
    results.append(("Separate Storage", success))

    # Test 8: Epsilon management
    success = test_epsilon_management(classes)
    results.append(("Epsilon Management", success))

    # Summary
    print("\n")
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:10} - {test_name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 70)

    if passed == total:
        print("\n✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("\nComponent 1 (RuntimeQTableManager) is VALIDATED and READY.")
        print("Proceeding to Component 2 (RuntimeRewardCalculator) is approved.\n")
        return 0
    else:
        print("\n❌ VALIDATION FAILED")
        print(f"\n{total - passed} test(s) failed. Fix issues before proceeding to Component 2.\n")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

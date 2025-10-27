#!/usr/bin/env python3
"""Quick test script for BTP example"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing BTP Runtime RL Example...")
    print("=" * 70)

    # Test 1: Import modules
    print("\n1. Testing imports...")
    from adaptiq.core.runtime_rl import RuntimeDecisionEngine, RuntimeQTableManager, CustomRewardCalculator
    from adaptiq.core.entities.q_table import QTableAction
    print("   ✓ All imports successful")

    # Test 2: Create components
    print("\n2. Creating Runtime RL components...")
    q_manager = RuntimeQTableManager(
        file_path="storage/qtables/test_btp.json",
        alpha=0.15,
        gamma=0.9,
        epsilon=0.2
    )
    print(f"   ✓ Q-Manager created: alpha={q_manager.alpha}, gamma={q_manager.gamma}, epsilon={q_manager.epsilon}")

    # Test 3: Custom reward function
    print("\n3. Testing custom reward function...")
    def test_reward(result):
        error_percent = result["error_percent"]
        if error_percent < 5:
            return 1.0
        elif error_percent < 15:
            return 0.5
        else:
            return 0.0

    reward_calc = CustomRewardCalculator(reward_fn=test_reward)
    test_result = {"error_percent": 8.0}
    reward = reward_calc.calculate_reward(test_result)
    print(f"   ✓ Reward calculator works: error=8% → reward={reward}")

    # Test 4: Decision engine
    print("\n4. Creating decision engine...")
    engine = RuntimeDecisionEngine(q_manager, reward_calc)
    print("   ✓ Decision engine created")

    # Test 5: Register actions
    print("\n5. Registering pricing methods...")
    actions = [
        QTableAction(action="method_db_standard"),
        QTableAction(action="method_ml_predict"),
        QTableAction(action="method_regional_adjust")
    ]
    engine.register_actions(actions)
    print(f"   ✓ Registered {len(actions)} actions")

    # Test 6: Test key_context construction from metadata
    print("\n6. Testing key_context construction from metadata...")
    context_with_metadata = {
        "subtask": "estimate_price",
        "last_action": "None",
        "last_outcome": "None",
        "metadata": {
            "material": "concrete",
            "region": "north",
            "surface": 20
        }
    }

    state = engine.build_state(context_with_metadata)
    print(f"   ✓ key_context auto-constructed: '{state.key_context}'")
    print(f"   ✓ State: current_subtask={state.current_subtask}")

    # Test 7: Make a decision
    print("\n7. Making a decision...")
    decision = engine.decide(context_with_metadata)
    print(f"   ✓ Decision: {decision.action.action}")
    print(f"   ✓ Explored: {decision.explored}")
    print(f"   ✓ Q-value: {decision.q_value:.4f}")

    # Test 8: Update Q-table
    print("\n8. Updating Q-table...")
    result = {"error_percent": 12.0}
    next_context = {
        "subtask": "estimate_price",
        "last_action": decision.action.action,
        "last_outcome": "success",
        "metadata": context_with_metadata["metadata"]
    }
    new_q = engine.update(decision, result, next_context)
    print(f"   ✓ Q-table updated: {decision.q_value:.4f} → {new_q:.4f}")

    print("\n" + "=" * 70)
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print("=" * 70)
    print("\nRuntime RL is working correctly!")
    print("\nYou can now run the full BTP example:")
    print("  python examples/runtime_rl_example.py")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

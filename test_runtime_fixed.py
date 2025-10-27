#!/usr/bin/env python3
"""Test that Runtime RL is working after fixes."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*60)
print("Testing Runtime RL after bug fixes...")
print("="*60)

try:
    # Test 1: Import
    print("\n[1/4] Testing imports...")
    from adaptiq.core.runtime_rl import RuntimeRLHelper
    from adaptiq.core.entities.q_table import QTableAction
    print("    ✓ Imports successful")

    # Test 2: Create temporary YAML config
    print("\n[2/4] Creating test YAML config...")
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            'runtime_rl': {
                'key_context_template': '{material}_{region}',
                'reward_function': 'accuracy_reward',
                'actions': ['method_a', 'method_b', 'method_c'],
                'storage_path': 'storage/qtables/test_runtime.json',
                'hyperparameters': {
                    'alpha': 0.1,
                    'gamma': 0.9,
                    'epsilon': 0.1
                }
            }
        }
        yaml.dump(config, f)
        temp_config_path = f.name

    print(f"    ✓ Test config created: {temp_config_path}")

    # Test 3: Initialize RuntimeRLHelper
    print("\n[3/4] Initializing RuntimeRLHelper from YAML...")
    runtime_rl = RuntimeRLHelper.from_yaml(temp_config_path)
    print("    ✓ RuntimeRLHelper initialized successfully")
    print(f"    - Actions: {runtime_rl.actions}")
    print(f"    - Storage: {runtime_rl.storage_path}")

    # Test 4: Make a decision
    print("\n[4/4] Testing decide() method...")
    decision = runtime_rl.decide(
        subtask="test_task",
        metadata={"material": "concrete", "region": "north"}
    )
    print(f"    ✓ Decision made successfully")
    print(f"    - Action: {decision.action.action}")
    print(f"    - Q-value: {decision.q_value:.4f}")
    print(f"    - Explored: {decision.explored}")

    # Cleanup
    os.unlink(temp_config_path)

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe RuntimeRLHelper bug has been fixed:")
    print("  - Changed 'storage_path' to 'file_path' parameter")
    print("  - Fixed import from 'adaptiq.core.learning' to 'adaptiq.core.entities'")
    print("\nYou can now run: python examples/runtime_rl_example_simplified.py")
    print("="*60)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

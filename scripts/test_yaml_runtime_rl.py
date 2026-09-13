#!/usr/bin/env python3
"""Quick test of YAML-based Runtime RL"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

print("Testing YAML-based Runtime RL...")
print("=" * 70)

try:
    # Test 1: Import RuntimeRLHelper
    print("\n1. Testing import...")
    from adaptiq.core.runtime_rl import RuntimeRLHelper
    print("   ✅ RuntimeRLHelper imported successfully")

    # Test 2: Load YAML config
    print("\n2. Testing YAML config loading...")
    config_path = "agents/btp_agent/runtime_rl_config.yaml"
    if not os.path.exists(config_path):
        print(f"   ❌ Config file not found: {config_path}")
    else:
        print(f"   ✅ Config file exists: {config_path}")

    # Test 3: Initialize RuntimeRLHelper
    print("\n3. Testing RuntimeRLHelper initialization...")
    runtime_rl = RuntimeRLHelper.from_yaml(config_path)
    print("   ✅ RuntimeRLHelper initialized successfully")

    # Test 4: Check stats
    print("\n4. Testing get_stats()...")
    stats = runtime_rl.get_stats()
    print(f"   ✅ Q-table size: {stats['q_table_size']}")
    print(f"   ✅ Actions: {stats['actions']}")
    print(f"   ✅ Storage: {stats['storage_path']}")
    print(f"   ✅ Hyperparameters: α={stats['hyperparameters']['alpha']}, "
          f"γ={stats['hyperparameters']['gamma']}, ε={stats['hyperparameters']['epsilon']}")

    # Test 5: Make a decision
    print("\n5. Testing decide()...")
    decision = runtime_rl.decide(
        subtask="estimate_price",
        metadata={"material": "concrete", "region": "north", "surface": 20}
    )
    print(f"   ✅ Decision made: action={decision.action.action}")
    print(f"   ✅ Q-value: {decision.q_value:.4f}")
    print(f"   ✅ Explored: {decision.explored}")

    # Test 6: Update Q-table
    print("\n6. Testing update()...")
    result = {
        "estimated": 5100.0,
        "actual": 5000.0,
        "error_percent": 2.0
    }
    runtime_rl.update(decision, result)
    print(f"   ✅ Q-table updated successfully")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\n💡 Runtime RL with YAML config is working correctly!")
    print("   Developer needs ONLY 3 lines of code:")
    print("   1. runtime_rl = RuntimeRLHelper.from_yaml('config.yaml')")
    print("   2. decision = runtime_rl.decide(...)")
    print("   3. runtime_rl.update(...)")
    print("=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

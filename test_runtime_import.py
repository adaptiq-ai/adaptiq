#!/usr/bin/env python3
"""Simple test to verify Runtime RL imports work correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing import of RuntimeRLHelper...")
    from adaptiq.core.runtime_rl import RuntimeRLHelper
    print("✓ RuntimeRLHelper imported successfully!")

    print("\nTesting import of RuntimeQTableManager...")
    from adaptiq.core.runtime_rl import RuntimeQTableManager
    print("✓ RuntimeQTableManager imported successfully!")

    print("\nTesting import of RuntimeDecisionEngine...")
    from adaptiq.core.runtime_rl import RuntimeDecisionEngine
    print("✓ RuntimeDecisionEngine imported successfully!")

    print("\nTesting import of QTableAction...")
    from adaptiq.core.entities.q_table import QTableAction
    print("✓ QTableAction imported successfully!")

    print("\n" + "="*60)
    print("✅ ALL IMPORTS SUCCESSFUL!")
    print("="*60)
    print("\nThe import error has been fixed.")
    print("You can now run: python examples/runtime_rl_example_simplified.py")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

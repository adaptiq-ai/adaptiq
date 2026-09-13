#!/usr/bin/env python3
"""Test imports for RuntimeQTableManager"""
import sys
import traceback

try:
    print("Testing import of RuntimeQTableManager...")
    from adaptiq.core.runtime_rl.runtime_q_table_manager import RuntimeQTableManager
    print("✓ RuntimeQTableManager imported successfully")

    print("\nTesting import of QTableManager...")
    from adaptiq.core.q_table.q_table_manager import QTableManager
    print("✓ QTableManager imported successfully")

    print("\nTesting import of QTable entities...")
    from adaptiq.core.entities.q_table import QTableAction, QTableState
    print("✓ QTable entities imported successfully")

    print("\nTesting RuntimeQTableManager initialization...")
    manager = RuntimeQTableManager()
    print(f"✓ RuntimeQTableManager created: {manager}")

    print("\nTesting inheritance...")
    print(f"Is instance of QTableManager: {isinstance(manager, QTableManager)}")
    print(f"Is subclass of QTableManager: {issubclass(RuntimeQTableManager, QTableManager)}")

    print("\n✓✓✓ All imports successful! ✓✓✓")

except Exception as e:
    print(f"\n✗ Error occurred:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

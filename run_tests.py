#!/usr/bin/env python3
"""Quick test runner for RuntimeQTableManager"""
import sys
import pytest

if __name__ == "__main__":
    # Run tests with verbose output
    exit_code = pytest.main([
        "tests/test_runtime_q_table_manager.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    sys.exit(exit_code)

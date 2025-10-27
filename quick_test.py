import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Step 1: Import")
try:
    from adaptiq.core.runtime_rl import RuntimeRLHelper
    print("SUCCESS: RuntimeRLHelper imported")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 2: Check YAML")
config_path = "agents/btp_agent/runtime_rl_config.yaml"
print(f"Path: {config_path}")
print(f"Exists: {os.path.exists(config_path)}")

print("\nStep 3: Load from YAML")
try:
    runtime_rl = RuntimeRLHelper.from_yaml(config_path)
    print("SUCCESS: Loaded from YAML")
    print(f"Actions: {runtime_rl.actions}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

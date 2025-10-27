# 🚀 Simplified Guide: Runtime RL with YAML

**Version:** AdaptIQ v0.12.8+
**Date:** 2025-10-27

---

## 🎯 Objective

This guide shows **THE SIMPLEST WAY** to integrate Runtime RL into your AdaptIQ agent.

**Developer code: ONLY 3 LINES!**

---

## 📋 YAML Configuration (Agent Side)

Create a YAML file in your agent's folder:

**File: `agents/my_agent/runtime_rl_config.yaml`**

```yaml
runtime_rl:
  # =============================================================================
  # 1️⃣ KEY_CONTEXT TEMPLATE
  # =============================================================================
  # How to build key_context from metadata?
  # Use Python format syntax: {field_name}
  #
  # BTP Example: "{material}_{region}_{surface}"
  # Result: "concrete_north_20" for concrete in north region, 20m²
  #
  key_context_template: "{material}_{region}_{surface}"

  # =============================================================================
  # 2️⃣ REWARD FUNCTION
  # =============================================================================
  # How to calculate reward from result dictionary?
  #
  # Available options:
  #   - "accuracy_reward": Builtin reward for error percentage
  #                        reward = 1 - (error_percent / 100)
  #   - "classification_reward": Builtin reward for binary classification
  #                              reward = +1 if correct, -1 if wrong
  #   - "lambda result: ...": Custom Python lambda expression
  #   - "my_module.my_function": Import custom function from Python module
  #
  # BTP Example (heavily penalizes errors > 25%):
  reward_function: "lambda result: 1.0 - (result['error_percent'] / 25.0) if result['error_percent'] < 25 else -1.0"

  # =============================================================================
  # 3️⃣ AVAILABLE ACTIONS
  # =============================================================================
  # List of method names (actions) that Runtime RL can choose from.
  # These must match the method names in your agent code.
  #
  actions:
    - method_db_standard      # Method 1
    - method_knn_historical   # Method 2
    - method_regional_adjust  # Method 3

  # =============================================================================
  # 4️⃣ Q-TABLE STORAGE
  # =============================================================================
  # Where to save/load the Q-table JSON file?
  # Directory will be created automatically if it doesn't exist.
  #
  storage_path: "storage/qtables/my_agent_runtime_q_table.json"

  # =============================================================================
  # 5️⃣ HYPERPARAMETERS (Optional)
  # =============================================================================
  # Q-Learning hyperparameters.
  #
  # alpha: Learning rate (0.0 to 1.0)
  #        Higher = faster learning, more sensitive to recent experiences
  #        Lower = slower learning, more stable
  #        Recommended: 0.1 - 0.2
  #
  # gamma: Discount factor (0.0 to 1.0)
  #        Higher = more weight on future rewards
  #        Lower = focus on immediate rewards
  #        Recommended: 0.9
  #
  # epsilon: Exploration rate (0.0 to 1.0)
  #          Higher = more exploration (random actions)
  #          Lower = more exploitation (best known actions)
  #          Recommended: 0.1 - 0.2
  #
  hyperparameters:
    alpha: 0.15      # Learning rate
    gamma: 0.9       # Discount factor
    epsilon: 0.15    # Exploration rate
```

---

## 💻 Python Code (Agent)

**ONLY 3 LINES OF CODE!**

### Initialization (1 line)

```python
from adaptiq.core.runtime_rl import RuntimeRLHelper

# Load YAML config
runtime_rl = RuntimeRLHelper.from_yaml("agents/my_agent/runtime_rl_config.yaml")
```

### Usage (2 lines)

```python
# 1️⃣ Decide which method to use
decision = runtime_rl.decide(
    subtask="estimate_price",
    metadata={"material": "concrete", "region": "north", "surface": 20}
)

# Execute the chosen method
pricing_method = my_methods[decision.action.action]
result = pricing_method(element)

# 2️⃣ Update Q-table ( only allowed by the Prerun or postrun not agents )
result_dict = {
    "estimated": result,
    "actual": actual_price,
    "error_percent": calculate_error(result, actual_price)
}
runtime_rl.update(decision, result_dict)
```

---

## 🏗️ Complete Example: BTP Agent

```python
from adaptiq.core.runtime_rl import RuntimeRLHelper

class BTPAgent:
    def __init__(self):
        # ✨ LINE 1: Load Runtime RL from YAML
        self.runtime_rl = RuntimeRLHelper.from_yaml(
            "agents/btp_agent/runtime_rl_config.yaml"
        )

        # Map of available methods
        self.pricing_methods = {
            "method_db_standard": self.method_db_standard,
            "method_knn_historical": self.method_knn_historical,
            "method_regional_adjust": self.method_regional_adjust,
        }

    def estimate_price(self, element: dict) -> float:
        """
        Estimate a price using Runtime RL.

        Args:
            element: {"material": str, "region": str, "surface": int}

        Returns:
            Estimated price in euros
        """
        # ✨ LINE 2: Decide which method to use
        decision = self.runtime_rl.decide(
            subtask="estimate_price",
            metadata=element  # key_context built automatically!
        )

        print(f"🎯 Method chosen: {decision.action.action}")

        # Execute the method
        pricing_method = self.pricing_methods[decision.action.action]
        estimated_price = pricing_method(element)

        # Get actual price (for learning)
        actual_price = self.get_actual_price(element)
        error_percent = abs(estimated_price - actual_price) / actual_price * 100



        # ✨ LINE 3: Update Q-table ( only allowed by the Prerun or postrun not agents )
        result = {
            "estimated": estimated_price,
            "actual": actual_price,
            "error_percent": error_percent
        }
        self.runtime_rl.update(decision, result)

        return estimated_price

    # Your business logic methods
    def method_db_standard(self, element: dict) -> float:
        """Standard catalog lookup"""
        return 250 * element["surface"]

    def method_knn_historical(self, element: dict) -> float:
        """KNN on past projects"""
        return self.knn_estimate(element)

    def method_regional_adjust(self, element: dict) -> float:
        """Regional adjustment on catalog"""
        base = 250 * element["surface"]
        regional_factor = {"paris": 1.1, "north": 1.0, "south": 0.95}
        return base * regional_factor.get(element["region"], 1.0)
```

---

## 🎮 Using the Agent

```python
# Create agent
agent = BTPAgent()

# Estimate a price (Runtime RL decides automatically!)
element = {"material": "concrete", "region": "north", "surface": 20}
price = agent.estimate_price(element)

print(f"Estimated price: {price}€")
```

**That's it!** Runtime RL automatically learns which method to use. 🎉

---

## 📊 Complete RuntimeRLHelper API

### `from_yaml(config_path: str)` → RuntimeRLHelper

Load configuration from a YAML file.

```python
runtime_rl = RuntimeRLHelper.from_yaml("agents/my_agent/runtime_rl_config.yaml")
```

### `decide(subtask: str, metadata: dict, ...)` → RuntimeDecision

Decide which action to take (epsilon-greedy).

```python
decision = runtime_rl.decide(
    subtask="estimate_price",
    metadata={"material": "concrete", "region": "north", "surface": 20},
    last_action="None",      # Optional
    last_outcome="None"      # Optional
)

print(f"Chosen action: {decision.action.action}")
print(f"Q-value: {decision.q_value:.4f}")
print(f"Explored?: {decision.explored}")
```

### `update(decision: RuntimeDecision, result: dict, ...)`

Update Q-table after execution.

```python
result = {
    "estimated": 5100.0,
    "actual": 5000.0,
    "error_percent": 2.0
}

#( only allowed by the Prerun or postrun not agents )

runtime_rl.update(
    decision,
    result,
    next_subtask=None,      # Optional (default: terminal)
    next_metadata=None      # Optional
)
```

### `save()`

Save Q-table to disk.

```python
runtime_rl.save()
```

### `load()`

Load Q-table from disk.

```python
runtime_rl.load()
```

### `get_stats()` → dict

Get Q-table statistics.

```python
stats = runtime_rl.get_stats()
print(f"Q-table size: {stats['q_table_size']}")
print(f"Actions: {stats['actions']}")
print(f"Hyperparameters: {stats['hyperparameters']}")
```

---

## 🔍 Builtin Reward Functions

### 1. accuracy_reward

For regression tasks (error percentage).

```yaml
reward_function: "accuracy_reward"
```

**Calculation:**
```python
reward = 1 - (error_percent / 100)
# Example: error_percent=5% → reward=0.95
```

**Required in result:**
- `error_percent`: Error percentage (0-100)

### 2. classification_reward

For binary classification tasks.

```yaml
reward_function: "classification_reward"
```

**Calculation:**
```python
reward = +1.0 if result["correct"] else -1.0
```

**Required in result:**
- `correct`: Boolean (True/False)

### 3. Custom Lambda Expression

For custom reward logic.

```yaml
reward_function: "lambda result: 1.0 - (result['error_percent'] / 25.0)"
```

**BTP Example (heavily penalizes errors > 25%):**
```yaml
reward_function: "lambda result: 1.0 - (result['error_percent'] / 25.0) if result['error_percent'] < 25 else -1.0"
```

### 4. Python Function Import

For complex logic in a separate module.

```yaml
reward_function: "my_module.my_custom_reward"
```

**Example:**
```python
# my_module.py
def my_custom_reward(result: dict) -> float:
    """Complex reward with multiple criteria"""
    error = result['error_percent']
    time = result['execution_time']

    # Penalize both error AND time
    reward_error = 1.0 - (error / 25.0)
    reward_time = 1.0 - (time / 10.0)

    return (reward_error + reward_time) / 2
```

---

## ✅ Integration Checklist

- [ ] Create YAML file `agents/my_agent/runtime_rl_config.yaml`
- [ ] Define `key_context_template` (e.g., `"{field1}_{field2}"`)
- [ ] Define `reward_function` (builtin or custom)
- [ ] List available `actions`
- [ ] Specify `storage_path` for Q-table
- [ ] (Optional) Adjust `hyperparameters`
- [ ] Import `RuntimeRLHelper` in your agent
- [ ] Call `from_yaml()` at initialization
- [ ] Call `decide()` before each decision 
- [ ] Call `update()` after each execution  
- [ ] Save with `save()` after training

---

## 🎓 Benefits of YAML Approach

| Benefit | Description |
|---------|-------------|
| **Zero boilerplate** | Only 3 lines of Python code |
| **Centralized config** | All config in 1 YAML file |
| **Code-free changes** | Change hyperparameters without touching Python |
| **Versionable** | YAML easy to version with Git |
| **Reusable** | Same Python code, different YAML configs |
| **Testable** | Easy to create multiple configs for tests |

---

## 💡 Best Practices

### 1. Name key_context Template Clearly

**❌ Bad:**
```yaml
key_context_template: "{a}_{b}_{c}"  # Which fields?
```

**✅ Good:**
```yaml
key_context_template: "{material}_{region}_{surface}"  # Clear!
```

### 2. Domain-Specific Reward Function

**❌ Generic:**
```yaml
reward_function: "accuracy_reward"  # Too simple?
```

**✅ Specific:**
```yaml
# Heavily penalizes large errors
reward_function: "lambda result: 1.0 - (result['error_percent'] / 25.0) if result['error_percent'] < 25 else -1.0"
```

### 3. Epsilon Adapted to Phase

**Training (exploration):**
```yaml
epsilon: 0.2  # 20% exploration
```

**Production (exploitation):**
```yaml
epsilon: 0.05  # 5% exploration (keep learning)
```

**Pure Evaluation:**
```python
runtime_rl.q_manager.epsilon = 0.0  # 0% exploration
```

### 4. Save Regularly

```python
# After each training batch
for batch in training_batches:
    train_on_batch(batch)
    runtime_rl.save()  # Incremental save
```

---

## 🐛 Troubleshooting

### Error: "YAML must contain 'runtime_rl' key"

**Cause:** Incorrect YAML structure.

**Solution:** Verify that YAML file starts with `runtime_rl:`

```yaml
runtime_rl:  # ← Don't forget!
  key_context_template: "..."
  reward_function: "..."
```

### Error: "key_context_template requires field 'XXX'"

**Cause:** Template requires a field that's not in `metadata`.

**Solution:** Verify that `metadata` contains all template fields.

```python
# Template: "{material}_{region}_{surface}"
metadata = {
    "material": "concrete",
    "region": "north",
    "surface": 20,  # Don't forget!
}
```

### Q-values stay at 0

**Cause:** Different key_context between `decide()` and `update()`.

**Solution:** Use EXACTLY the same `metadata` fields in decide() and update().

```python
# ✅ Good: Same metadata
metadata = {"material": "concrete", "region": "north", "surface": 20}
decision = runtime_rl.decide(subtask="...", metadata=metadata)
# ... execution ...
runtime_rl.update(decision, result)  # Uses same key_context

# ❌ Bad: Different metadata
decision = runtime_rl.decide(subtask="...", metadata={"material": "concrete"})
runtime_rl.update(decision, result, next_metadata={"material": "concrete", "region": "north"})
# → Different key_context!
```

---

## 🔧 Bug Fixes Applied (v0.12.8+)

### Bug 1: Import Error
**Fixed:** Changed import from `adaptiq.core.learning.q_table_manager` to `adaptiq.core.entities.q_table`

### Bug 2: Parameter Name
**Fixed:** Changed `storage_path` parameter to `file_path` in RuntimeQTableManager initialization

### Bug 3: Attribute Name
**Fixed:** Changed `q_table` attribute to `Q_table` (capital Q) in get_stats()

All bugs are now resolved! ✅

---

## 📦 Delivered Files

| File | Description |
|------|-------------|
| `src/adaptiq/core/runtime_rl/runtime_rl_helper.py` | RuntimeRLHelper class (simplified API) |
| `agents/btp_agent/runtime_rl_config.yaml` | YAML config example for BTP |
| `examples/runtime_rl_example_simplified.py` | Complete example with YAML |
| `RUNTIME_RL_SIMPLIFIED_GUIDE.md` | This guide |

---

## 🚀 Executable Example

To test the YAML approach:

```bash
cd adaptiq
python examples/runtime_rl_example_simplified.py
```

**What this example does:**
1. Loads config from `agents/btp_agent/runtime_rl_config.yaml`
2. Trains Runtime RL on 50 BTP estimations
3. Evaluates learned policy
4. Displays Q-values and errors

**Total developer code: 3 lines!** ✨

---

## 📞 Support

For more information:
- **Detailed guide:** `RUNTIME_RL_DEVELOPER_GUIDE.md`
- **Pricing methods explained:** `PRICING_METHODS_EXPLAINED.md`
- **Tests:** `tests/test_runtime_rl.py`

---

**Version:** 1.1
**Date:** 2025-10-27
**AdaptIQ:** v0.12.8+

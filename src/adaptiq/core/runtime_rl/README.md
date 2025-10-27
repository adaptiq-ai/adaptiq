# Runtime RL - Online Decision Making for AdaptIQ

**Runtime RL** extends AdaptIQ with online reinforcement learning capabilities for making real-time decisions during agent execution.

## Overview

### Offline RL vs Runtime RL

| Feature | Offline RL (Existing) | Runtime RL (New) |
|---------|----------------------|------------------|
| **When** | Pre-Run & Post-Run | During execution |
| **What it optimizes** | Agent prompts/tools | Business method selection |
| **Learning approach** | Batch learning from traces | Online learning from outcomes |
| **Hyperparameters** | α=0.8, γ=0.8 | α=0.1, γ=0.9 |
| **Exploration** | Pure exploitation | Epsilon-greedy (ε=0.1) |
| **Storage** | `adaptiq_q_table.json` | `runtime_q_table.json` |

### Architecture

```
User Code
    ↓
RuntimeDecisionEngine ←─────────────────┐
    ├─> RuntimeQTableManager            │
    │   └─> epsilon-greedy selection    │
    │                                    │
    ├─> RuntimeRewardCalculator         │
    │   └─> domain-specific rewards     │
    │                                    │
    └─> User-defined Executors          │
            └─> execution results ───────┘
```

## Components

### 1. RuntimeQTableManager

Manages Q-values with epsilon-greedy exploration for online learning.

**Inherits from:** `QTableManager` (reuses Bellman equation)

**Key Features:**
- Epsilon-greedy action selection
- Online learning hyperparameters (α=0.1, γ=0.9)
- Separate storage from offline Q-table

**Example:**
```python
from adaptiq.core.runtime_rl import RuntimeQTableManager
from adaptiq.core.entities.q_table import QTableState, QTableAction

# Initialize
manager = RuntimeQTableManager(
    file_path="storage/qtables/runtime_q_table.json",
    alpha=0.1,    # Learning rate
    gamma=0.9,    # Discount factor
    epsilon=0.1   # Exploration rate
)

# Define state and actions
state = QTableState(
    current_subtask="price_estimation",
    last_action_taken="None",
    last_outcome="None",
    key_context="small_dataset"
)

actions = [
    QTableAction(action="method_a"),
    QTableAction(action="method_b")
]

# Select action (epsilon-greedy)
action, explored = manager.select_action_epsilon_greedy(state, actions)

if explored:
    print(f"EXPLORATION: Random action {action.action}")
else:
    print(f"EXPLOITATION: Best action {action.action}")

# Execute action, get reward...

# Update Q-table
next_state = QTableState(...)
reward = 0.85
manager.update_policy(state, action, reward, next_state, actions)

# Save Q-table
manager.save_q_table(prefix_version="runtime")
```

### 2. RuntimeRewardCalculator

Calculates rewards based on business metrics (accuracy, error rate, precision, recall, etc.).

**Available Calculators:**
- `AccuracyRewardCalculator`: Simple accuracy-based rewards
- `ClassificationRewardCalculator`: Precision/recall/F1-based rewards
- `CustomRewardCalculator`: User-defined reward functions

**Example:**
```python
from adaptiq.core.runtime_rl import (
    AccuracyRewardCalculator,
    ClassificationRewardCalculator,
    CustomRewardCalculator,
    create_reward_calculator
)

# Option 1: Accuracy-based
calc = AccuracyRewardCalculator(
    success_weight=1.0,
    error_weight=0.5
)

result = {"accuracy": 0.9, "error_rate": 0.1}
reward = calc.calculate_reward(result)
# reward = 1.0 * 0.9 - 0.5 * 0.1 = 0.85

# Option 2: Classification metrics
calc = ClassificationRewardCalculator(use_f1=True)

result = {"f1_score": 0.85}
reward = calc.calculate_reward(result)
# reward = 0.85

# Option 3: Custom reward function
def profit_reward(result):
    profit = result["profit"]
    cost = result["cost"]
    return (profit - cost) / 1000  # Normalized

calc = CustomRewardCalculator(reward_fn=profit_reward)

result = {"profit": 1500, "cost": 500}
reward = calc.calculate_reward(result)
# reward = (1500 - 500) / 1000 = 1.0

# Option 4: Factory function
calc = create_reward_calculator("accuracy", success_weight=1.0, error_weight=0.5)
```

### 3. RuntimeDecisionEngine

Orchestrates the decision-making process, integrating Q-table manager and reward calculator.

**Example:**
```python
from adaptiq.core.runtime_rl import (
    RuntimeDecisionEngine,
    RuntimeQTableManager,
    AccuracyRewardCalculator
)
from adaptiq.core.entities.q_table import QTableAction

# Initialize components
q_manager = RuntimeQTableManager(epsilon=0.1)
reward_calc = AccuracyRewardCalculator()

# Create engine
engine = RuntimeDecisionEngine(
    q_table_manager=q_manager,
    reward_calculator=reward_calc
)

# Register available actions
actions = [
    QTableAction(action="linear_regression"),
    QTableAction(action="neural_network")
]
engine.register_actions(actions)

# ============================================================
# Runtime Decision Loop
# ============================================================

# Step 1: Make decision
context = {
    "subtask": "price_prediction",
    "last_action": "None",
    "last_outcome": "None",
    "key_context": "small_dataset"
}

decision = engine.decide(context)
print(f"Selected: {decision.action.action}")
print(f"Explored: {decision.explored}")
print(f"Q-value: {decision.q_value:.4f}")

# Step 2: Execute action (user code)
result = execute_method(decision.action.action)
# result = {"accuracy": 0.85, "error_rate": 0.15}

# Step 3: Update Q-table
next_context = {
    "subtask": "price_prediction",
    "last_action": decision.action.action,
    "last_outcome": "success",
    "key_context": "small_dataset"
}

new_q = engine.update(decision, result, next_context)
print(f"Q-value updated: {decision.q_value:.4f} → {new_q:.4f}")

# Step 4: Save learned Q-table
engine.save_q_table(prefix_version="runtime")
```

## Usage Patterns

### Pattern 1: Epsilon Decay Strategy

Gradually reduce exploration as the agent learns:

```python
engine = RuntimeDecisionEngine(q_manager, reward_calc)

# Initial phase: High exploration
engine.set_epsilon(0.3)  # 30% exploration

for i in range(1000):
    decision = engine.decide(context)
    # ... execute and update ...

    # Decay epsilon over time
    if i == 100:
        engine.set_epsilon(0.1)  # Reduce to 10%
    elif i == 500:
        engine.set_epsilon(0.05)  # Reduce to 5%
```

### Pattern 2: Multi-Context Decision Making

Learn different strategies for different contexts:

```python
# Small dataset context
context_small = {
    "subtask": "estimation",
    "last_action": "None",
    "last_outcome": "None",
    "key_context": "small_dataset"
}

# Large dataset context
context_large = {
    "subtask": "estimation",
    "last_action": "None",
    "last_outcome": "None",
    "key_context": "large_dataset"
}

# Agent learns different optimal actions for each context
decision_small = engine.decide(context_small)
decision_large = engine.decide(context_large)

# Different Q-values based on context
print(f"Small dataset: {decision_small.action.action}")
print(f"Large dataset: {decision_large.action.action}")
```

### Pattern 3: Custom Reward Functions

Define domain-specific reward logic:

```python
def trading_reward(result):
    """Reward function for trading decisions"""
    profit = result["profit"]
    risk = result["risk"]
    execution_time = result["execution_time"]

    # Multi-factor reward
    reward = (
        0.6 * (profit / 1000) +      # Profit component
        -0.2 * risk +                # Risk penalty
        -0.2 * (execution_time / 10) # Speed penalty
    )

    return max(-1.0, min(1.0, reward))  # Clip to [-1, 1]

calc = CustomRewardCalculator(reward_fn=trading_reward)
engine = RuntimeDecisionEngine(q_manager, calc)
```

## Integration with AdaptIQ

### Adding Runtime RL to AdaptIQ Agent

```python
from adaptiq import AdaptiqRun
from adaptiq.core.runtime_rl import (
    RuntimeDecisionEngine,
    RuntimeQTableManager,
    AccuracyRewardCalculator
)

class MyAdaptiqAgent(AdaptiqRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize Runtime RL
        self.runtime_engine = RuntimeDecisionEngine(
            q_table_manager=RuntimeQTableManager(),
            reward_calculator=AccuracyRewardCalculator()
        )

        # Register business methods as actions
        self.runtime_engine.register_actions([
            QTableAction(action="method_a"),
            QTableAction(action="method_b")
        ])

    def execute_task_with_runtime_rl(self, task_context):
        """Execute task using Runtime RL for method selection"""

        # Build RL context from task
        rl_context = {
            "subtask": task_context["task_name"],
            "last_action": task_context.get("last_method", "None"),
            "last_outcome": task_context.get("last_result", "None"),
            "key_context": task_context["data_size"]
        }

        # Decide which method to use
        decision = self.runtime_engine.decide(rl_context)

        # Execute selected method
        result = self.execute_method(decision.action.action, task_context)

        # Update Q-table with result
        next_context = {**rl_context, "last_action": decision.action.action}
        self.runtime_engine.update(decision, result, next_context)

        return result
```

## Testing

Run the comprehensive test suite:

```bash
# Test all components
pytest tests/test_runtime_q_table_manager.py -v
pytest tests/test_runtime_rewards.py -v
pytest tests/test_runtime_decision_engine.py -v

# Run example
python examples/runtime_rl_example.py
```

## Configuration

Runtime RL can be configured via YAML (future enhancement):

```yaml
# config/runtime_rl_config.yaml
runtime_rl:
  enabled: true

  q_table:
    file_path: "storage/qtables/runtime_q_table.json"
    alpha: 0.1
    gamma: 0.9
    epsilon: 0.1

  reward_calculator:
    type: "accuracy"  # or "classification", "custom"
    success_weight: 1.0
    error_weight: 0.5

  actions:
    - "linear_regression"
    - "neural_network"
    - "decision_tree"

  epsilon_decay:
    enabled: true
    schedule:
      - {decisions: 0, epsilon: 0.3}
      - {decisions: 100, epsilon: 0.1}
      - {decisions: 500, epsilon: 0.05}
```

## API Reference

### RuntimeQTableManager

- `__init__(file_path, alpha=0.1, gamma=0.9, epsilon=0.1)`
- `select_action_epsilon_greedy(state, actions) -> (action, explored)`
- `set_epsilon(new_epsilon)`
- `get_epsilon() -> float`
- `update_policy(s, a, R, s_prime, actions_prime) -> float` (inherited)
- `save_q_table(prefix_version) -> bool` (inherited)
- `load_q_table() -> bool` (inherited)

### RuntimeRewardCalculator

**AccuracyRewardCalculator:**
- `__init__(success_weight=1.0, error_weight=0.5)`
- `calculate_reward(result) -> float`

**ClassificationRewardCalculator:**
- `__init__(precision_weight=0.5, recall_weight=0.5, f1_weight=1.0, use_f1=False)`
- `calculate_reward(result) -> float`

**CustomRewardCalculator:**
- `__init__(reward_fn: Callable)`
- `calculate_reward(result) -> float`

### RuntimeDecisionEngine

- `__init__(q_table_manager, reward_calculator)`
- `register_actions(actions)`
- `build_state(context) -> QTableState`
- `decide(context) -> RuntimeDecision`
- `update(decision, result, next_context) -> float`
- `save_q_table(prefix_version) -> bool`
- `load_q_table() -> bool`
- `set_epsilon(new_epsilon)`
- `get_epsilon() -> float`
- `get_hyperparameters() -> dict`

## Examples

See [`examples/runtime_rl_example.py`](../../../../examples/runtime_rl_example.py) for a complete working example.

## References

- [Q-Learning Algorithm](https://en.wikipedia.org/wiki/Q-learning)
- [Epsilon-Greedy Exploration](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/)
- [AdaptIQ Documentation](../../../../README.md)

## License

MIT License - See LICENSE file for details.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.adaptiq.q_learning.q_learning import AdaptiqOfflineLearner # Assuming this is your class


learner = AdaptiqOfflineLearner(
    config_path="../src/adaptiq/config/adaptiq_config.yml", # Path to your config
    alpha=0.7,
    gamma=0.9,
    similarity_threshold=0.85 # Will only be used if represent_state uses embeddings
)

state_s = (
    "DraftEmailCopy",               # Current_SubTask (Inferred from 'Thought: write email...')
    "Use_Tool_LeadNameTool",        # Last_Action_Taken (ARIC predefined action)
    "Success_DataFound",            # Last_Outcome (Categorized outcome)
)

# Action A taken from State S
# This is an action from ARIC's predefined action_menu.
action_a = "Action_WriteEmail" # Agent decides to proceed with writing the email

# Reward R for taking Action A in State S
# This would be calculated by ARIC's RewardCalculator based on the trace segment
# where the email was written. Let's assume the email draft was okay but not perfectly personalized.
reward_r = 0.3 # Positive, but not perfect (e.g., basic draft, some personalization missing)

# Next State S' (after the email was drafted)
# This would be the state key for the situation *after* the WriteEmail action completed.
state_s_prime = (
    "AttemptingEmailSend",          # Current_SubTask (Inferred from 'Thought: now send email...')
    "Action_WriteEmail",            # Last_Action_Taken (the action we just evaluated)
    "Success_EmailDrafted",         # Last_Outcome (Categorized outcome of writing)
    ("LeadDataAvailable", True),    # Key_Context (LeadDataAvailable is still true)
    ("EmailDraftReady", True)       # New context: email is now drafted
)

# Possible next actions A' from State S' (from ARIC's action_menu)
next_actions_from_s_prime = [
    "Use_Tool_SendEmailTool",
    "Action_ReviseEmailDraft", # Hypothetical action if agent could revise
    "Action_GiveUp"
]

# Perform Q-learning update
new_q_value = learner.update_policy(state_s, action_a, reward_r, state_s_prime, next_actions_from_s_prime)

# Output the new Q-value
print(f"Updated Q-value for (State S, Action A): {new_q_value:.4f}")
print(f"State S: {state_s}")
print(f"Action A: {action_a}")

# Let's assume the save_q_table method in your class can take a path directly for testing.
q_table_path = "q_table_copywriter_test.json"
if hasattr(learner.config.get("q_learning_engine", {}), 'get'): # Check if q_learning_engine is dict-like
    q_table_path = learner.config.get("q_learning_engine", {}).get("q_table_persistence_path", q_table_path)

success = learner.save_q_table(q_table_path) # Assuming save_q_table uses its internal path or this as override
print(f"Q-table saved to {q_table_path}: {success}")

# Print out the Q-table
q_table = learner.get_q_table()
print("\nCurrent Q-table contents:")
for (s, a), q_val in q_table.items(): # Renamed q to q_val to avoid conflict
    print(f"State: {s}\nAction: {a}\nQ-value: {q_val:.4f}\n---")
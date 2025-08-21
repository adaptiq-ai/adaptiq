import ast
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict
from datetime import datetime, timezone

from adaptiq.core.entities.q_table import QTablePayload, QTableState, QTableAction, QTableQValue


class BaseQTableManager(ABC):

    def __init__(self, file_path: str, alpha: float = 0.8, gamma: float = 0.8):
        self.file_path = file_path
        self.alpha = alpha
        self.gamma = gamma

        # Internal structure: QTableState -> Dict[QTableAction, QTableQValue]
        self.Q_table: Dict[QTableState, Dict[QTableAction, QTableQValue]] = {}
        self.seen_states: set[QTableState] = set()

    @abstractmethod
    def update_policy(
        self,
        s: QTableState,
        a: QTableAction,
        R: float,
        s_prime: QTableState,
        actions_prime: List[QTableAction]
    ) -> float:
        pass

    def save_q_table(self, prefix_version: str = None) -> bool:
        """Save Q-table to file using the new data model"""
        try:
            version = prefix_version + "_" + str(uuid.uuid4())[:5] if prefix_version else "1.0"

            # Convert to serializable format
            serialized_q_table: Dict[str, Dict[str, QTableQValue]] = {}
            
            for state, actions_dict in self.Q_table.items():
                # FIX: Use the string representation of the tuple directly as the key.
                state_key = f"{state.to_tuple()}"  # No more ast.literal_eval
                serialized_q_table[state_key] = {}
                
                for action, q_value in actions_dict.items():
                    serialized_q_table[state_key][action.to_str()] = q_value

            # Serialize seen states
            serialized_seen_states = [
                f"{state.to_tuple()}" for state in self.seen_states
            ]

            payload = QTablePayload(
                Q_table=serialized_q_table,
                seen_states=serialized_seen_states,
                version=version,
                timestamp=datetime.now(timezone.utc)
            )

            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(payload.model_dump_json(indent=2))

            print(f"[INFO] Q-table saved successfully to: {self.file_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to save Q-table: {e}")
            return False

    def load_q_table(self) -> bool:
        """Load Q-table from file using the new data model"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                payload = QTablePayload.model_validate_json(f.read())

            self.Q_table.clear()
            self.seen_states.clear()

            # Load Q-values
            for state_str, actions_dict in payload.Q_table.items():
                # FIX: Parse the string key from JSON back into a tuple.
                state_tuple = ast.literal_eval(state_str)
                
                if len(state_tuple) == 4:
                    state = QTableState.from_tuple(state_tuple)
                    self.Q_table[state] = {}
                    
                    for action_str, q_value in actions_dict.items():
                        action = QTableAction.from_str(action_str)
                        # Handle both dict and QTableQValue objects
                        if isinstance(q_value, dict):
                            self.Q_table[state][action] = QTableQValue(**q_value)
                        else:
                            self.Q_table[state][action] = q_value

            # Load seen states
            for state_str in payload.seen_states:
                # FIX: Parse the string representation back into a tuple.
                state_tuple = ast.literal_eval(state_str)
                if len(state_tuple) == 4:
                    state = QTableState.from_tuple(state_tuple)
                    self.seen_states.add(state)

            print(f"[INFO] Q-table loaded successfully from: {self.file_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to load Q-table: {e}")
            return False
    def Q(self, s: QTableState, a: QTableAction) -> float:
        """Get Q-value for state-action pair"""
        if s not in self.Q_table:
            return 0.0
        if a not in self.Q_table[s]:
            return 0.0
        return self.Q_table[s][a].q_value

    def get_q_table(self) -> Dict[QTableState, Dict[QTableAction, QTableQValue]]:
        """Return copy of the Q-table"""
        return dict(self.Q_table)
    
    def set_q_value(self, state: QTableState, action: QTableAction, value: float) -> None:
        """
        Directly set Q-value for a state-action pair
        
        Args:
            state: The state
            action: The action
            value: The Q-value to set
        """
        if state not in self.Q_table:
            self.Q_table[state] = {}
        
        self.Q_table[state][action] = QTableQValue(q_value=value)
        self.seen_states.add(state)

    def get_actions_for_state(self, state: str) -> List[str]:
        """
        Get all actions available for a given state in the Q-table.

        Args:
            state: The state to look up (should be a serialized state string).

        Returns:
            List[str]: List of action strings associated with the state
        """
        actions = []
        for q_state in self.Q_table.keys():
            # Compare serialized state strings
            if q_state.to_tuple() == state:
                # Get all actions for this state
                for action in self.Q_table[q_state].keys():
                    actions.append(action.action)
                break  # Found the state, no need to continue
        return actions
    
    def get_actions_for_state_object(self, state: QTableState) -> List[QTableAction]:
        """
        Get all actions available for a given state object in the Q-table.

        Args:
            state: The QTableState object to look up.

        Returns:
            List[QTableAction]: List of actions associated with the state
        """
        if state in self.Q_table:
            return list(self.Q_table[state].keys())
        return []
   
    def get_best_action(self, state: QTableState, available_actions: List[QTableAction]) -> QTableAction:
        """
        Get the best action for a given state based on Q-values
        
        Args:
            state: Current state
            available_actions: List of actions available in this state
            
        Returns:
            QTableAction: Best action to take
        """
        if not available_actions:
            raise ValueError("No available actions provided")
        
        if state not in self.Q_table:
            # If state not seen before, return random action (first one)
            return available_actions[0]
        
        best_action = available_actions[0]
        best_q_value = self.Q(state, best_action)
        
        for action in available_actions[1:]:
            q_value = self.Q(state, action)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action

    def get_action_values(self, state: QTableState, actions: List[QTableAction]) -> List[tuple[QTableAction, float]]:
        """
        Get Q-values for all specified actions in a state
        
        Args:
            state: State to evaluate
            actions: List of actions to get values for
            
        Returns:
            List of (action, q_value) tuples sorted by Q-value descending
        """
        action_values = [(action, self.Q(state, action)) for action in actions]
        return sorted(action_values, key=lambda x: x[1], reverse=True)
    
    def serialize_q_table(self) -> QTablePayload:
        """
        Serialize the Q-table to a QTablePayload object for JSON storage.
        
        Returns:
            QTablePayload: Serialized Q-table data with stringified states and actions
        """
        output_q_table = {}
        for state, actions_dict in self.Q_table.items():
            # FIX: Use the string representation directly.
            state_key_str = f"{state.to_tuple()}"
            
            if state_key_str not in output_q_table:
                output_q_table[state_key_str] = {}
            
            # Convert each action and its Q-value
            for action, q_value in actions_dict.items():
                action_str = action.to_str()
                output_q_table[state_key_str][action_str] = q_value

        # This seems to have a bug as well, it should just be state.to_tuple()
        seen_states_output = [f"{s.to_tuple()}" for s in self.seen_states]

        # Create and return QTablePayload object
        return QTablePayload(
            Q_table=output_q_table,
            seen_states=seen_states_output,
            version="NA"
        )
    
    def get_q_table_dict(self) -> Dict:
        """
        Get Q-table data as a dictionary (for backward compatibility).
        
        Returns:
            dict: Q-table data in dictionary format
        """
        payload = self.serialize_q_table()

        # unwrap QTableQValue -> float
        q_table_unwrapped = {
            state_str: {
                action_str: (
                    q_val.q_value if isinstance(q_val, QTableQValue) else q_val
                )
                for action_str, q_val in actions.items()
            }
            for state_str, actions in payload.Q_table.items()
        }

        return {
            "Q_table": q_table_unwrapped,
            "seen_states": payload.seen_states,
            "timestamp": payload.timestamp.isoformat(),
            "version": payload.version,
        }
    
    def extract_q_table_insights_to_str(self) -> str:
        """
        Extracts insights from the Q-table output.
        Focuses on states and actions with non-zero Q-values.
        """

        q_table_output = self.get_q_table_dict()
        q_table:Dict = q_table_output.get("Q_table", {})
        if not q_table:
            return "No Q-table data available to analyze."

        insights = ["Q-Table Insights (States and Actions with Non-Zero Q-values):\n"]
        for state_str, actions_dict in q_table.items():
            if not actions_dict:
                continue

            # Filter actions with non-zero Q-values
            non_zero_actions = {a: q for a, q in actions_dict.items() if q != 0.0}
            if not non_zero_actions:
                continue

            insights.append(f"  State: {state_str}")
            for action, q_value in non_zero_actions.items():
                insights.append(f"    - Action: {action}, Q-Value: {q_value:.4f}")
            insights.append("")  # Newline for readability

        if len(insights) == 1:  # Only the header was added
            return "Q-table exists but contains no states with non-zero Q-values."
        
        return "\n".join(insights)
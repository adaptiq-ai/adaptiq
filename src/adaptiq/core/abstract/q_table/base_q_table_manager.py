import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict
from datetime import datetime, timezone

from adaptiq.core.entities.q_table import QTablePayload


class BaseQTableManager(ABC):

    def __init__(self, file_path: str, alpha: float = 0.8, gamma: float = 0.8):
        self.file_path = file_path
        self.alpha = alpha
        self.gamma = gamma

        # Internal structure: (state, action) -> value
        self.Q_table: Dict[Tuple[Any, str], float] = {}
        self.seen_states: set = set()

    @abstractmethod
    def update_policy(
        self,
        agent_id: str,
        s: Tuple[Any],
        a: str,
        R: float,
        s_prime: Tuple[Any],
        actions_prime: List[str]
    ) -> float:
        pass

    def Q(self, agent_id: str, s: Any, a: str) -> float:
        return self.Q_table.get((s, a), 0.0)

    def get_q_table(self) -> Dict:
        return dict(self.Q_table)

    def save_q_table(self, prefix_version: str = None) -> bool:
        try:
            version = prefix_version + "_" + str(uuid.uuid4())[:5] if prefix_version else "1.0"

            # Flatten into: { state_str: { action: value } }
            flat_q_table: Dict[str, Dict[str, float]] = {}
            for (state, action), value in self.Q_table.items():
                state_str = str(state)
                if state_str not in flat_q_table:
                    flat_q_table[state_str] = {}
                flat_q_table[state_str][action] = value

            payload = QTablePayload(
                Q_table=flat_q_table,
                seen_states=[str(s) for s in self.seen_states],
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
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                payload = QTablePayload.model_validate_json(f.read())

            self.Q_table.clear()
            self.seen_states.clear()

            # Load Q-values
            for state_str, actions_dict in payload.Q_table.items():
                state = self._deserialize_state(state_str)
                for action, value in actions_dict.items():
                    self.Q_table[(state, action)] = float(value)

            # Load seen states
            for state_str in payload.seen_states:
                state = self._deserialize_state(state_str)
                self.seen_states.add(state)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to load Q-table: {e}")
            return False

    def _deserialize_state(self, state_str: str) -> Any:
        try:
            if state_str.startswith("(") and state_str.endswith(")"):
                parts = state_str[1:-1].split(", ")
                result = []
                for part in parts:
                    if part.startswith("'") or part.startswith('"'):
                        result.append(part.strip("'\""))
                    elif part.lower() in {"true", "false"}:
                        result.append(part.lower() == "true")
                    elif part.lower() in {"none", "null"}:
                        result.append(None)
                    else:
                        try:
                            result.append(float(part) if '.' in part else int(part))
                        except:
                            result.append(part)
                return tuple(result)
            return state_str
        except Exception as e:
            print(f"[WARN] Failed to deserialize state: {state_str}, error: {e}")
            return state_str

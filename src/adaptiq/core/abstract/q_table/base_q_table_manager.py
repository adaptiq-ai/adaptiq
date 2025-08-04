import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Dict
from pathlib import Path

from adaptiq.core.entities.q_table import QTablePayload, QTableStateActions

class  BaseQTableManager(ABC):

    def __init__(self,file_path: str, alpha: float = 0.8, gamma: float = 0.8):

        if not Path(self.file_path).exists():

            print(f"[WARN] Q-table file not found: {self.file_path}")
            raise FileNotFoundError(f"Q-table file not found: {self.file_path}")
        

        self.Q_table: Dict[Tuple[Any, str], float] = {}
        self.seen_states: set = set()
        self.file_path = file_path
        self.alpha = alpha
        self.gamma = gamma
        
    @abstractmethod
    def update_policy(
        self,
        s: Tuple[Any],
        a: str,
        R: float,
        s_prime: Tuple[Any],
        actions_prime: List[str]
    ) -> float:
        pass

    def Q(self, s: Any, a: str) -> float:
        if (s, a) not in self.Q_table:
            self.Q_table[(s, a)] = 0.0
        return self.Q_table[(s, a)]

    def get_q_table(self) -> Dict:
        return dict(self.Q_table) 

    def save_q_table(self, prefix_version: str = None) -> bool:
        try:
            version = prefix_version + "_" + str(uuid.uuid4())[:5] if prefix_version else "1.0"

            nested_q: Dict[str, Dict[str, float]] = {}
            for (state, action), value in self.Q_table.items():
                key = str(state)
                if key not in nested_q:
                    nested_q[key] = {}
                nested_q[key][action] = value

            payload = QTablePayload(
                Q_table={k: QTableStateActions(__root__=v) for k, v in nested_q.items()},
                seen_states=[str(s) for s in self.seen_states],
                version=version
            )

            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(payload.model_dump_json(indent=2))

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

            for state_str, actions in payload.Q_table.items():
                state = self._deserialize_state(state_str)
                self.seen_states.add(state)
                for action, value in actions.__root__.items():
                    self.Q_table[(state, action)] = float(value)

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

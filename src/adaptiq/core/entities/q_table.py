import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class QTableState(BaseModel):
    current_subtask: str
    last_action_taken: str
    last_outcome: str
    key_context: str

    def to_tuple(self) -> Tuple[str, str, str, str]:
        """Serialize state to a tuple (safe for JSON as str)."""
        return (
            self.current_subtask,
            self.last_action_taken,
            self.last_outcome,
            self.key_context,
        )

    @classmethod
    def from_tuple(cls, t: Tuple[str, str, str, str]) -> "QTableState":
        """Deserialize tuple back into QTableState."""
        return cls(
            current_subtask=t[0],
            last_action_taken=t[1],
            last_outcome=t[2],
            key_context=t[3],
        )

    def __hash__(self):
        """Custom hash method for using as dictionary key"""
        return hash(self.to_tuple())

    def __eq__(self, other):
        """Custom equality method"""
        if not isinstance(other, QTableState):
            return False
        return self.to_tuple() == other.to_tuple()


class QTableAction(BaseModel):
    action: str

    def to_str(self) -> str:
        return self.action

    @classmethod
    def from_str(cls, s: str) -> "QTableAction":
        return cls(action=s)

    def __hash__(self):
        """Custom hash method for using as dictionary key"""
        return hash(self.action)

    def __eq__(self, other):
        """Custom equality method"""
        if not isinstance(other, QTableAction):
            return False
        return self.action == other.action


class QTableQValue(BaseModel):
    q_value: float


class QTablePayload(BaseModel):
    # Store serialized keys instead of objects
    Q_table: Dict[str, Dict[str, QTableQValue]] = Field(
        ..., description="Q-table with stringified states and actions"
    )
    seen_states: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str

    def save_json(self, path: str):
        with open(path, "a") as f:
            f.write(self.model_dump_json(indent=2, default=str))

    @classmethod
    def load_json(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_native(self) -> Dict[QTableState, Dict[QTableAction, QTableQValue]]:
        """Convert stringified keys back into objects."""
        native: Dict[QTableState, Dict[QTableAction, QTableQValue]] = {}
        for state_str, actions in self.Q_table.items():
            state = QTableState.from_tuple(tuple(state_str.split("||")))
            native[state] = {}
            for action_str, qval in actions.items():
                native[state][QTableAction.from_str(action_str)] = QTableQValue(**qval)
        return native

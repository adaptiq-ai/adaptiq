from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, timezone


class QTableStateActions(BaseModel):
    __root__: Dict[str, float]  # action: Q-value


class QTablePayload(BaseModel):
    Q_table: Dict[str, QTableStateActions] = Field(
        ..., description="Nested Q-table with stringified states"
    )
    seen_states: List[str] = Field(
        default_factory=list, description="List of stringified seen states"
    )
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now(timezone.utc), description="When the Q-table was saved"
    )
    version: str = Field(
        ..., description="Q-table version identifier"
    )

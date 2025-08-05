
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

class QTablePayload(BaseModel):
    Q_table: Dict[str, Dict[str, float]] = Field(
        ..., description="Flat Q-table with stringified states"
    )
    seen_states: List[str] = Field(
        default_factory=list, description="List of stringified seen states"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the Q-table was saved"
    )
    version: str = Field(
        ..., description="Q-table version identifier"
    )

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from typing import Literal, Tuple, Dict, Any
from pydantic import BaseModel, Field, field_validator

class TaskIntent(BaseModel):
    intended_subtask: str = Field("Unnamed task", description="The subtask the prompt is targeting")
    intended_action: str = Field(..., description="The action to be performed for the subtask")
    preconditions_mentioned_in_prompt: Optional[str] = Field(
        None, description="Any preconditions explicitly stated in the prompt"
    )
    expected_ideal_outcome_mentioned_in_prompt: Optional[str] = Field(
        None, description="The ideal or expected outcome mentioned in the prompt"
    )

########################################################-----########################################################

class HypotheticalStateRepresentation(BaseModel):
    state: str
    action: str
    details: Optional[Dict[str, str]] = {}

########################################################-----########################################################

class ScenarioModel(BaseModel):
    original_state: str
    intended_action: str
    scenario_type: Literal["ideal_success", "common_failure", "partial_success"]
    simulated_action: str = Field(..., description="The action/tool actually used in this scenario")
    simulated_outcome: str
    reward_sim: float
    next_state: Tuple[str, str, str, str] = Field(
        ..., description="(next_subtask, simulated_action, outcome_type, context)"
    )
    key_context_changes: Dict[str, Any]
    source_details: Dict[str, str]

########################################################-----########################################################

class PromptParsingStatus(BaseModel):
    completed: bool
    steps_found: int


class HypotheticalRepresentationStatus(BaseModel):
    completed: bool
    states_generated: int


class ScenarioSimulationStatus(BaseModel):
    completed: bool
    scenarios_generated: int


class QTableInitializationStatus(BaseModel):
    completed: bool
    q_entries: int


class PromptAnalysisStatus(BaseModel):
    completed: bool
    weaknesses_found: int
    suggestions_provided: int


class StatusSummary(BaseModel):
    prompt_parsing: PromptParsingStatus
    hypothetical_representation: HypotheticalRepresentationStatus
    scenario_simulation: ScenarioSimulationStatus
    qtable_initialization: QTableInitializationStatus
    prompt_analysis: PromptAnalysisStatus

########################################################-----########################################################

class FormattedAnalysis(BaseModel):
    summary: str
    weaknesses: Optional[List[str]]
    suggested_modifications: Optional[List[str]]
    best_practices: Optional[List[str]]
    missing_components: Optional[List[str]]
    strengths: Optional[List[str]]

########################################################-----########################################################

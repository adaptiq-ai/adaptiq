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

class LogState(BaseModel):
    """Represents the state context of an agent at a given step."""
    current_sub_task_or_thought: str = Field(..., description="The agent’s current thought or sub-task")
    last_action_taken: str = Field(..., description="The previous action executed")
    last_outcome: Any = Field(..., description="The outcome of the last action")
    agent_context: str = Field(..., description="Identifier or name of the agent")


class LogKey(BaseModel):
    """Defines the key structure that links state and action."""
    state: LogState
    agent_action: str = Field(..., description="The action chosen by the agent in the current step")


class LogItem(BaseModel):
    """Full standardized log entry."""
    key: LogKey
    reward_exec: float = Field(..., description="Normalized reward value")


class ProcessedLogs(BaseModel):
    """Collection of processed logs after parsing."""
    processed_logs: List[LogItem]

########################################################-----########################################################


class RewardAssessment(BaseModel):
    """Assessment of whether the reward is valid and potentially adjusted."""
    original: float = Field(..., description="The raw reward before validation")
    is_appropriate: bool = Field(..., description="Whether the reward value is deemed appropriate")
    adjusted: float = Field(..., description="The validated or adjusted reward value")
    reason: str = Field(..., description="Reason for validation decision or adjustment")


class ValidatedEntry(BaseModel):
    """Represents a log entry after validation with reward assessment."""
    reward_assessment: RewardAssessment
    corrected_entry: LogItem

########################################################-----########################################################

class ValidationSummary(BaseModel):
    total_entries: int = Field(..., description="Total number of validated entries")
    entries_with_appropriate_rewards: int = Field(..., description="Count of entries with appropriate rewards")
    entries_with_reward_adjustments: int = Field(..., description="Count of entries where rewards were adjusted")
    appropriate_reward_rate: float = Field(..., description="Proportion of entries with appropriate rewards")
    reward_adjustment_rate: float = Field(..., description="Proportion of entries with reward adjustments")
    average_adjustment_magnitude: float = Field(..., description="Average magnitude of reward adjustments")

class ValidationResults(BaseModel):
    """Represents the results of the validation process."""
    summary: ValidationSummary
    validated_entries: List[ValidatedEntry]

########################################################-----########################################################

class StateActionMapping(BaseModel):
    state: List[str] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Compact state representation: [sub_task, last_action, last_outcome, context]"
    )
    action: str = Field(
        ...,
        description="Clean tool name only, e.g. 'FileReadTool'"
    )

########################################################-----########################################################

class Classification(BaseModel):
    is_known_state: bool
    matched_state: Optional[str]  # null if not found
    reasoning: str


class ClassificationResponse(BaseModel):
    classification: Classification

class ClassicationEntry(BaseModel):
    index: int
    input_state: StateActionMapping
    classification: Classification

########################################################-----########################################################

class ReconciliationSummary(BaseModel):
    total_extracted_pairs: int
    total_classified_states: int
    known_states_found: int
    unknown_states_found: int
    task_key: Optional[str] = None
    new_prompt: str = None

class ReconciliationResults(BaseModel):
    pipeline_status: str = None
    extracted_data: List[StateActionMapping] = None
    state_classifications: List[ClassicationEntry] = None
    updated_qtable: Dict = None
    report_content: str = None
    summary: ReconciliationSummary = None

########################################################-----########################################################

class Outputs(BaseModel):
    parsed_logs_path: str
    validated_logs_path: str
    validation_summary_path: str


class Stats(BaseModel):
    parsed_entries_count: int
    validation_results: ValidationResults


class ValidationData(BaseModel):
    outputs: Outputs
    stats: Stats


class PostRunResults(BaseModel):
    validation_data: ValidationData
    reconciliation_results: ReconciliationResults


class PreRunResults(BaseModel):
    parsed_steps: List[TaskIntent]
    hypothetical_states: List[HypotheticalStateRepresentation]
    simulated_scenarios: List[ScenarioModel]
    q_table_size: int
    prompt_analysis: FormattedAnalysis
    new_prompt: str
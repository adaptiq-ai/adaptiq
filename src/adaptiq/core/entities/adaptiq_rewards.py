from enum import Enum


class CrewRewards(Enum):
    """
    Enum containing all reward constants and configuration values for CrewAI log parsing.
    
    This enum centralizes all the reward values, penalties, thresholds, and string constants
    used in the reward calculation system for different CrewAI log entry types (AgentAction,
    AgentFinish, TaskLog).
    
    The reward system is designed to encourage:
    - Meaningful thoughts and descriptions (>250 characters)
    - Successful tool usage with non-empty results
    - Comprehensive final outputs
    - Complete task logs with descriptions and raw outputs
    - Error-free operations
    
    Attributes are organized into categories:
    - Thresholds: Length-based criteria for quality assessment
    - General: Basic thought quality rewards/penalties
    - AgentAction: Tool usage and thinking action rewards
    - AgentFinish: Final output quality rewards
    - TaskLog: Task completion and documentation rewards
    - Keywords: Error detection and placeholder identification
    - Actions: String representations for different action types
    """
    # Thresholds for thought/output quality
    MIN_MEANINGFUL_THOUGHT_LEN = 250
    SHORT_OUTPUT_LEN_THRESHOLD = 500
    MEDIUM_OUTPUT_LEN_THRESHOLD = 1000

    # General
    BONUS_GOOD_THOUGHT = 0.15
    PENALTY_POOR_THOUGHT = -0.15  # For empty/placeholder/very short thoughts

    # AgentAction: Tool Usage
    REWARD_TOOL_SUCCESS = 1.0
    REWARD_TOOL_SUCCESS_EMPTY_RESULT = 0.25  # Tool worked, but result was empty
    PENALTY_TOOL_ERROR = -1.0
    PENALTY_TOOL_NO_RESULT_FIELD = -0.75  # Tool was called, but 'result' key is missing
    PENALTY_TOOL_NAME_EMPTY_STRING = -1.0  # If 'tool' field is an empty string

    # AgentAction: Thinking (No Tool)
    REWARD_AGENT_THINK_ACTION_GOOD_THOUGHT = 0.3
    PENALTY_AGENT_THINK_ACTION_POOR_THOUGHT = -0.3

    # AgentFinish: Final Output
    REWARD_FINAL_OUTPUT_LONG = 0.75
    REWARD_FINAL_OUTPUT_MEDIUM = 0.5
    REWARD_FINAL_OUTPUT_SHORT = 0.2
    PENALTY_FINAL_OUTPUT_EMPTY_OR_PLACEHOLDER = -0.5

    # TaskLog
    REWARD_TASKLOG_HAS_DESCRIPTION = 0.25
    PENALTY_TASKLOG_NO_DESCRIPTION = -0.25
    REWARD_TASKLOG_HAS_RAW = 0.5
    PENALTY_TASKLOG_NO_RAW = -0.5
    PENALTY_TASKLOG_RAW_CONTAINS_ERROR = -0.75

    # Keywords and Placeholder strings
    ERROR_KEYWORDS = [
        "error:",
        "traceback:",
        "failed to execute",
        "exception:",
        "failure:",
    ]
    PLACEHOLDER_STRINGS_LOWER = [
        "none",
        "n/a",
        "missing thought",
        "empty thought",
        "task log content",
        "null",
    ]

    # Action representations for keys
    ACTION_AGENT_THOUGHT_PROCESS = "AgentThoughtProcess"
    ACTION_INVALID_TOOL_EMPTY_NAME = "InvalidTool(EmptyName)"
    ACTION_FINAL_ANSWER = "FinalAnswer"
    TASKLOG_NO_RAW_OUTPUT_REPR = "NoRawOutputInTaskLog"
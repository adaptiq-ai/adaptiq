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

    # Time-based thresholds and rewards (in seconds)
    MAX_REASONABLE_STEP_TIME = 30.0
    FAST_STEP_TIME_THRESHOLD = 5.0   
    SLOW_STEP_TIME_THRESHOLD = 15.0

    REWARD_FAST_EXECUTION = 0.2
    PENALTY_SLOW_EXECUTION = -0.3
    PENALTY_EXCESSIVE_TIME = -0.8

    # Token-based thresholds and rewards
    MAX_REASONABLE_TOKENS = 2000
    EFFICIENT_TOKEN_THRESHOLD = 500
    VERBOSE_TOKEN_THRESHOLD = 1200
    EXCESSIVE_TOKEN_THRESHOLD = 1800

    REWARD_EFFICIENT_TOKENS = 0.15
    PENALTY_VERBOSE_TOKENS = -0.2
    PENALTY_EXCESSIVE_TOKENS = -0.5


class FeedbackRewards(Enum):
    """
    Enum containing reward constants for feedback-based adjustments to agent actions.

    This enum defines rewards and penalties based on user feedback events and sentiment analysis.
    Feedback can come from explicit events (VALUE_INCREASED, VALUE_DECREASED, TEXT_EDITED)
    or human-submitted feedback with sentiment analysis (positive, negative, neutral).

    The reward system integrates user feedback to:
    - Reinforce actions that led to positive outcomes (increased values, positive sentiment)
    - Penalize actions that led to negative outcomes (decreased values, negative sentiment)
    - Apply moderate adjustments for text edits and neutral sentiment
    - Scale rewards based on sentiment polarity scores (-1.0 to 1.0)

    Event-based rewards are fixed, while sentiment-based rewards are scaled by polarity score.

    Categories:
    - Event Feedback: Rewards for VALUE_INCREASED, VALUE_DECREASED, TEXT_EDITED events
    - Sentiment Feedback: Base rewards multiplied by polarity_score for HUMAN_SUBMITTED_FEEDBACK
    - Thresholds: Polarity score ranges for sentiment interpretation
    """

    # Event-based feedback rewards (fixed values)
    REWARD_VALUE_INCREASED = 1.0  # Strong positive signal: user improved/increased a value
    PENALTY_VALUE_DECREASED = -1.0  # Strong negative signal: user decreased a value
    REWARD_TEXT_EDITED = 0.3  # Moderate signal: user edited text (could be improvement or correction)

    # Sentiment-based feedback rewards (base values, scaled by polarity_score)
    # Formula: reward = base_reward * polarity_score
    # polarity_score ranges from -1.0 (most negative) to +1.0 (most positive)
    SENTIMENT_BASE_REWARD_POSITIVE = 1.2  # Base for positive sentiment (scaled: 0.12 to 1.2)
    SENTIMENT_BASE_REWARD_NEUTRAL = 0.0  # Neutral sentiment provides no reward/penalty
    SENTIMENT_BASE_PENALTY_NEGATIVE = -1.2  # Base for negative sentiment (scaled: -0.12 to -1.2)

    # Polarity score thresholds for sentiment classification
    POLARITY_THRESHOLD_POSITIVE = 0.1  # Scores > 0.1 considered positive
    POLARITY_THRESHOLD_NEGATIVE = -0.1  # Scores < -0.1 considered negative
    # Scores between -0.1 and 0.1 are considered neutral

    # Combined feedback multipliers (when multiple feedback signals agree)
    MULTIPLIER_CONSISTENT_POSITIVE = 1.5  # When event and sentiment both positive
    MULTIPLIER_CONSISTENT_NEGATIVE = 1.5  # When event and sentiment both negative
    MULTIPLIER_CONFLICTING = 0.5  # When event and sentiment contradict

    # Relevance-based scaling (from reranking relevance_score)
    MIN_RELEVANCE_THRESHOLD = 0.3  # Minimum relevance score to apply feedback reward
    RELEVANCE_SCALE_FACTOR = 1.0  # Scale feedback reward by relevance_score
from datetime import datetime
from typing import Any, Literal, Optional, List
from pydantic import BaseModel, Field


class FeedbackContext(BaseModel):
    """Context information about where the feedback originated."""
    ui_location: str = Field(..., description="UI location where feedback was generated")
    item_id: str = Field(..., description="Identifier for the specific item")


class FeedbackDetails(BaseModel):
    """Details of the feedback containing original and new values."""
    original_value: Optional[Any] = Field(None, description="Original value before feedback")
    new_value: Any = Field(..., description="New value or feedback content")


class FeedbackPayload(BaseModel):
    """Payload containing the feedback event data."""
    event_name: Literal["VALUE_DECREASED", "VALUE_INCREASED", "TEXT_EDITED", "HUMAN_SUBMITTED_FEEDBACK"] = Field(
        ..., 
        description="Name of the feedback event"
    )
    context: FeedbackContext = Field(..., description="Context of the feedback")
    details: FeedbackDetails = Field(..., description="Feedback details")


class FeedbackEvent(BaseModel):
    """Individual feedback event."""
    event_id: str = Field(..., description="Unique identifier for the event")
    trace_id: str = Field(..., description="Trace ID linking to the original run")
    user_id: str = Field(..., description="User who generated the feedback")
    timestamp_utc: datetime = Field(..., description="UTC timestamp of the event")
    feedback_type: Literal["HUMAN", "EVENT"] = Field(..., description="Type of feedback")
    payload: FeedbackPayload = Field(..., description="Feedback payload data")


class FeedbackMetadata(BaseModel):
    """Metadata about the feedback batch."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    fetch_timestamp_utc: datetime = Field(..., description="UTC timestamp when batch was fetched")
    source_api_endpoint: str = Field(..., description="API endpoint source")
    event_count: int = Field(..., description="Number of events in the batch")


class Feedback(BaseModel):
    """Main feedback structure containing metadata and events."""
    metadata: FeedbackMetadata = Field(..., description="Batch metadata")
    feedback_events: List[FeedbackEvent] = Field(..., description="List of feedback events")


class FeedbackDocument(BaseModel):
    """Document representation of a processed log entry for reranking."""
    agent_task: str = Field(..., description="The agent's current thought or sub-task")
    agent_action: str = Field(..., description="The action chosen by the agent")
    relevance_score: Optional[float] = Field(None, description="Relevance score from reranking (0.0 to 1.0)")


class SentimentAnalysisResults(BaseModel):
    """Results from sentiment analysis on feedback text."""
    class_label: Literal["positive", "negative", "neutral"] = Field(
        ...,
        description="The sentiment class/label of the text"
    )
    polarity_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Polarity score ranging from -1 (most negative) to +1 (most positive)"
    )


class RerankResult(BaseModel):
    """Results from reranking feedback documents."""
    feedback_documents: List[FeedbackDocument] = Field(..., description="List of feedback documents with relevance scores")


class FeedbackResult(BaseModel):
    """Overall feedback results including reranked documents and sentiment analysis."""
    event: FeedbackEvent = Field(..., description="The original feedback event")
    document: FeedbackDocument = Field(..., description="Top feedback document after reranking")
    reward_feedback: Optional[float] = Field(None, description="Numerical reward value derived from feedback")


class FeedbackMap(BaseModel):
    """Batch results containing multiple feedback results."""
    batch_id: str = Field(..., description="Unique identifier for the batch")
    results: List[FeedbackResult] = Field(..., description="List of individual feedback results")







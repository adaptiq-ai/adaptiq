from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Any, Dict, List, Optional
from datetime import datetime


class CallInfo(BaseModel):
    """Information about a single function call with validation"""
    function_name: str = Field(..., min_length=1, description="Name of the function that was called")
    timestamp: str = Field(..., description="ISO format timestamp of the call")
    input_tokens: int = Field(ge=0, description="Number of input tokens used")
    output_tokens: int = Field(ge=0, description="Number of output tokens generated")
    total_tokens: int = Field(ge=0, description="Total tokens used (input + output)")
    llm_calls: int = Field(ge=0, description="Number of LLM API calls made")
    session_id: str = Field(default="", description="Session identifier")
    thread_id: str = Field(default="", description="Thread identifier")
    execution_time: Optional[float] = Field(default=None, ge=0, description="Function execution time in seconds")
    model_name: Optional[str] = Field(default=None, description="LLM model used")
    
    @field_validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError('timestamp must be in ISO format')
    
    @model_validator(mode='before')
    def validate_token_consistency(cls, values):
        """Validate that total_tokens >= input_tokens + output_tokens"""
        input_tokens = values.get('input_tokens', 0)
        output_tokens = values.get('output_tokens', 0)
        total_tokens = values.get('total_tokens', 0)
        
        if total_tokens < input_tokens + output_tokens:
            # Auto-correct if total is less than sum
            values['total_tokens'] = input_tokens + output_tokens
        
        return values
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "function_name": "analyze_text",
                "timestamp": "2024-01-15T10:30:00",
                "input_tokens": 150,
                "output_tokens": 75,
                "total_tokens": 225,
                "llm_calls": 1,
                "session_id": "session_123",
                "thread_id": "thread_456",
                "execution_time": 2.5,
                "model_name": "gpt-4"
            }
        }


class TokenStats(BaseModel):
    """Token statistics for a mode with validation and computed properties"""
    total_input_tokens: int = Field(default=0, ge=0, description="Total input tokens across all calls")
    total_output_tokens: int = Field(default=0, ge=0, description="Total output tokens across all calls")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens across all calls")
    total_calls: int = Field(default=0, ge=0, description="Total number of function calls")
    call_history: List[CallInfo] = Field(default_factory=list, description="History of all calls")
    
    # Computed properties
    @property
    def average_tokens_per_call(self) -> float:
        """Calculate average tokens per call"""
        return self.total_tokens / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def average_input_tokens_per_call(self) -> float:
        """Calculate average input tokens per call"""
        return self.total_input_tokens / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def average_output_tokens_per_call(self) -> float:
        """Calculate average output tokens per call"""
        return self.total_output_tokens / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def total_execution_time(self) -> float:
        """Calculate total execution time from call history"""
        return sum(call.execution_time or 0 for call in self.call_history)
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time per call"""
        if self.total_calls == 0:
            return 0.0
        total_time = sum(call.execution_time or 0 for call in self.call_history if call.execution_time)
        calls_with_time = len([call for call in self.call_history if call.execution_time])
        return total_time / calls_with_time if calls_with_time > 0 else 0.0
    
    def add_call(self, call_info: CallInfo) -> None:
        """Thread-safe method to add call information with validation"""
        # Validate the call_info
        if not isinstance(call_info, CallInfo):
            raise TypeError("call_info must be an instance of CallInfo")
        
        # Update totals
        self.total_input_tokens += call_info.input_tokens
        self.total_output_tokens += call_info.output_tokens
        self.total_tokens += call_info.total_tokens
        self.total_calls += 1
        self.call_history.append(call_info)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of statistics"""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "average_tokens_per_call": self.average_tokens_per_call,
            "average_input_tokens_per_call": self.average_input_tokens_per_call,
            "average_output_tokens_per_call": self.average_output_tokens_per_call,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": self.average_execution_time,
            "first_call": self.call_history[0].timestamp if self.call_history else None,
            "last_call": self.call_history[-1].timestamp if self.call_history else None
        }
    
    def filter_calls(self, 
                    function_name: Optional[str] = None,
                    session_id: Optional[str] = None,
                    thread_id: Optional[str] = None,
                    min_tokens: Optional[int] = None,
                    max_tokens: Optional[int] = None) -> List[CallInfo]:
        """Filter call history based on criteria"""
        filtered_calls = self.call_history
        
        if function_name:
            filtered_calls = [call for call in filtered_calls if call.function_name == function_name]
        
        if session_id:
            filtered_calls = [call for call in filtered_calls if call.session_id == session_id]
        
        if thread_id:
            filtered_calls = [call for call in filtered_calls if call.thread_id == thread_id]
        
        if min_tokens is not None:
            filtered_calls = [call for call in filtered_calls if call.total_tokens >= min_tokens]
        
        if max_tokens is not None:
            filtered_calls = [call for call in filtered_calls if call.total_tokens <= max_tokens]
        
        return filtered_calls
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "total_input_tokens": 1500,
                "total_output_tokens": 750,
                "total_tokens": 2250,
                "total_calls": 10,
                "call_history": []
            }
        }

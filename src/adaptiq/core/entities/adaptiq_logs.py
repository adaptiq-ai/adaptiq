
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class CrewLogs(BaseModel):
    """
    Pydantic model for representing agent execution logs from CrewAI or similar agent frameworks.
    
    This model captures both intermediate agent actions and final agent outputs, providing
    a structured way to parse and validate agent execution logs.
    
    Attributes:
        timestamp (str): Timestamp of the log entry in 'YYYY-MM-DD HH:MM:SS' format.
        type (Literal["AgentAction", "AgentFinish"]): Type of log entry.
            - "AgentAction": Represents an intermediate action taken by the agent
            - "AgentFinish": Represents the final output/completion of the agent
        thought (Optional[str]): Agent's internal reasoning or thought process.
        text (Optional[str]): Main content of the log entry, often containing action 
            descriptions, observations, or final answers.
        tool (Optional[str]): Name of the tool used by the agent. Only present for 
            "AgentAction" type entries.
        tool_input (Optional[str]): Input parameters passed to the tool as a JSON string.
            Only present for "AgentAction" type entries.
        result (Optional[str]): Output/result returned by the tool execution.
            Only present for "AgentAction" type entries.
        output (Optional[str]): Final output produced by the agent. Only present for
            "AgentFinish" type entries.
    
    Example:
        >>> # AgentAction log entry
        >>> action_log = CrewLogs(
        ...     timestamp="2025-07-19 13:13:15",
        ...     type="AgentAction",
        ...     thought="I need to gather company information",
        ...     text="Action: Read a file's content",
        ...     tool="Read a file's content",
        ...     tool_input='{"file_path": "knowledge/company.txt"}',
        ...     result="Company information retrieved successfully"
        ... )
        
        >>> # AgentFinish log entry
        >>> finish_log = CrewLogs(
        ...     timestamp="2025-07-19 13:13:29",
        ...     type="AgentFinish",
        ...     thought="I now have the final answer",
        ...     text="Final response with email content",
        ...     output="Subject: Welcome to our services!"
        ... )
    
    """
    timestamp: str = Field(..., description="Timestamp in format 'YYYY-MM-DD HH:MM:SS'")
    type: Literal["AgentAction", "AgentFinish"] = Field(..., description="Type of log entry")
    thought: Optional[str] = Field(None, description="Agent's thought process")
    text: Optional[str] = Field(None, description="Main content/action description")
    tool: Optional[str] = Field(None, description="Tool used (only for AgentAction)")
    tool_input: Optional[str] = Field(None, description="Tool input as JSON string (only for AgentAction)")
    result: Optional[str] = Field(None, description="Tool execution result (only for AgentAction)")
    output: Optional[str] = Field(None, description="Final output (only for AgentFinish)")
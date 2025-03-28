from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class Property(BaseModel):
    type: str
    description: str
    enum: Optional[List[Any]] = None    

class Parameters(BaseModel):
    type: str
    properties: Dict[str, Property]
    required: List[str]
    additionalProperties: bool = False

class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters

class ToolCall(BaseModel):
    name: str
    arguments: str

class ToolCallRequest(BaseModel):
    """Standardized tool call request that can be returned by any AI provider"""
    tool_calls: List[ToolCall]
    content: str
    finish_reason: Optional[str] = None

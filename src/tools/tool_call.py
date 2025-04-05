"""
Tool call model for handling tool invocations.
"""
from typing import Dict, Any
from pydantic import BaseModel


class ToolCall(BaseModel):
    """Model for a tool call request."""
    name: str
    arguments: Dict[str, Any]
    id: str = None 
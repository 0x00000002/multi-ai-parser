"""
Interfaces for tool implementations.
"""
from typing import Protocol, Dict, Any
from typing_extensions import runtime_checkable


@runtime_checkable
class ToolStrategy(Protocol):
    """Interface for tool implementations."""
    
    def execute(self, **args) -> Any:
        """
        Execute the tool with the provided arguments.
        
        Args:
            args: Tool-specific arguments
            
        Returns:
            Tool execution result
        """
        ...
    
    def get_description(self) -> str:
        """
        Get a description of the tool.
        
        Returns:
            Tool description string
        """
        ...
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool parameters.
        
        Returns:
            JSON schema as a dictionary
        """
        ... 
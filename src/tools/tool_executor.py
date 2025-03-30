"""
Tool executor for handling tool execution.
"""
from typing import Any, Dict, Optional
from .interfaces import ToolStrategy
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError
from .models import ToolResult


class ToolExecutor:
    """Executor for handling tool execution."""
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the tool executor.
        
        Args:
            logger: Logger instance
        """
        self._logger = logger
    
    def execute(self, tool: ToolStrategy, **args) -> ToolResult:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool: The tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            AIToolError: If execution fails
        """
        try:
            # Execute the tool
            result = tool.execute(**args)
            
            return ToolResult(
                success=True,
                result=result,
                tool_name=tool.get_name() if hasattr(tool, 'get_name') else None
            )
            
        except Exception as e:
            self._logger.error(f"Tool execution failed: {str(e)}")
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                tool_name=tool.get_name() if hasattr(tool, 'get_name') else None
            ) 
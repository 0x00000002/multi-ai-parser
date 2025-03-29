from typing import Any
from src.Logger import Logger
from src.ai.tools.tools_list import Tool
import json

class ToolExecutor:
    """Handles the execution of tools with proper error handling and logging."""
    
    def __init__(self, logger: Logger):
        self._logger = logger
    
    def execute(self, tool_name: str, tool_args: str) -> Any:
        """
        Execute a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool in JSON string format
            
        Returns:
            The result of the tool execution
        """
        try:
            # Find the tool in the available tools
            for tool in Tool:
                if tool.value[0].name == tool_name:
                    # Execute the tool function with the arguments
                    tool_func = tool.value[1]
                    args = json.loads(tool_args)
                    result = tool_func(**args)
                    self._logger.info(f"Tool {tool_name} executed successfully with result: {result}")
                    return result
            self._logger.warning(f"Tool {tool_name} not found")
            return None
        except Exception as e:
            self._logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return None 
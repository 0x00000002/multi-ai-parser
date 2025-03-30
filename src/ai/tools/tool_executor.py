from typing import Any, Dict, Optional, Union
from src.logger import Logger
from src.ai.tools.tools_list import Tool
import json
import re

class ToolResult:
    """Structured result from a tool execution."""
    
    def __init__(self, 
                 success: bool, 
                 result: Any = None, 
                 message: str = "", 
                 tool_name: str = None):
        """
        Initialize a tool result.
        
        Args:
            success: Whether the tool execution was successful
            result: The result data from the tool (if successful)
            message: An explanatory message, especially for errors
            tool_name: The name of the tool that was executed
        """
        self.success = success
        self.result = result
        self.message = message
        self.tool_name = tool_name
    
    def __str__(self) -> str:
        """String representation of the result."""
        if self.success:
            return f"Success ({self.tool_name}): {self.result}"
        return f"Failed: {self.message}"
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "message": self.message,
            "tool_name": self.tool_name
        }


class ToolExecutor:
    """Handles the execution of tools with integrated parsing capabilities."""
    
    def __init__(self, logger: Logger):
        """Initialize the tool executor with a logger."""
        self._logger = logger
        self._result_cache = {}  # Simple cache for tool results
    
    def execute_from_response(self, content: str) -> ToolResult:
        """
        Parse and execute a tool call from a response string.
        
        Args:
            content: The response content to parse for tool calls
        
        Returns:
            ToolResult with the execution outcome
        """
        tool_call = self._parse_tool_call(content)
        if not tool_call:
            return ToolResult(
                success=False, 
                message="No valid tool call found in response"
            )
            
        return self.execute(tool_call["name"], json.dumps(tool_call["arguments"]))
    
    def execute(self, tool_name: str, tool_args_str: str) -> ToolResult:
        """
        Execute a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args_str: Arguments for the tool in JSON string format
            
        Returns:
            ToolResult with execution outcome
        """
        # Parse arguments
        try:
            args = json.loads(tool_args_str)
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse tool arguments: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Invalid tool arguments: {str(e)}",
                tool_name=tool_name
            )
            
        # Check cache for identical calls
        cache_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        if cache_key in self._result_cache:
            self._logger.info(f"Using cached result for tool {tool_name}")
            return self._result_cache[cache_key]
        
        # Find and execute the tool
        try:
            for tool in Tool:
                if tool.value[0].name == tool_name:
                    # Execute the tool function with the arguments
                    tool_func = tool.value[1]
                    result = tool_func(**args)
                    
                    tool_result = ToolResult(
                        success=True,
                        result=result,
                        tool_name=tool_name
                    )
                    
                    # Cache the result
                    self._result_cache[cache_key] = tool_result
                    self._logger.info(f"Tool {tool_name} executed successfully")
                    return tool_result
                    
            # No matching tool found
            self._logger.warning(f"Tool {tool_name} not found")
            return ToolResult(
                success=False,
                message=f"Tool '{tool_name}' not found",
                tool_name=tool_name
            )
            
        except Exception as e:
            self._logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return ToolResult(
                success=False,
                message=f"Tool execution error: {str(e)}",
                tool_name=tool_name
            )
    
    def clear_cache(self) -> None:
        """Clear the tool result cache."""
        self._result_cache = {}
        self._logger.info("Tool result cache cleared")
    
    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Parse a tool call from the given content.
        Uses multiple strategies to extract and validate the tool call.
        
        Args:
            content: The text content to parse
            
        Returns:
            Dictionary containing tool name and arguments if found, None otherwise
        """
        # Strategy 1: Try to find a complete JSON object after TOOL_CALL:
        tool_call_match = re.search(r'TOOL_CALL:\s*(.+)', content)
        if tool_call_match:
            try:
                json_str = tool_call_match.group(1).strip()
                return self._parse_json_str(json_str)
            except Exception as e:
                self._logger.error(f"Error processing tool call from text: {type(e).__name__} - {str(e)}")
        
        # Strategy 2: Try to extract name and arguments separately
        name_match = re.search(r'"name":\s*"([^"]+)"', content)
        args_match = re.search(r'"arguments":\s*({[^}]+})', content)
        if name_match and args_match:
            try:
                tool_name = name_match.group(1)
                args_str = args_match.group(1)
                if not args_str.endswith('}'):
                    args_str += '}'
                args_str = re.sub(r'\s+', ' ', args_str)
                return {
                    "name": tool_name,
                    "arguments": json.loads(args_str)
                }
            except Exception as e:
                self._logger.error(f"Error in final fallback attempt: {type(e).__name__} - {str(e)}")
        
        return None
    
    def _parse_json_str(self, json_str: str) -> Dict[str, Any]:
        """Parse and validate a JSON string."""
        # Clean up the JSON string
        json_str = json_str.strip()
        json_str = re.sub(r',\s*}', '}', json_str)
        if not json_str.endswith('}'):
            json_str += '}'
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # Parse and validate
        tool_call_data = json.loads(json_str)
        if not isinstance(tool_call_data, dict) or 'name' not in tool_call_data or 'arguments' not in tool_call_data:
            raise ValueError("Invalid tool call structure")
        return tool_call_data


# from typing import Any
# from src.logger import Logger
# from src.ai.tools.tools_list import Tool
# import json

# class ToolExecutor:
#     """Handles the execution of tools with proper error handling and logging."""
    
#     def __init__(self, logger: Logger):
#         self._logger = logger
    
#     def execute(self, tool_name: str, tool_args: str) -> Any:
#         """
#         Execute a tool with the given name and arguments.
        
#         Args:
#             tool_name: Name of the tool to execute
#             tool_args: Arguments for the tool in JSON string format
            
#         Returns:
#             The result of the tool execution
#         """
#         try:
#             # Find the tool in the available tools
#             for tool in Tool:
#                 if tool.value[0].name == tool_name:
#                     # Execute the tool function with the arguments
#                     tool_func = tool.value[1]
#                     args = json.loads(tool_args)
#                     result = tool_func(**args)
#                     self._logger.info(f"Tool {tool_name} executed successfully with result: {result}")
#                     return result
#             self._logger.warning(f"Tool {tool_name} not found")
#             return None
#         except Exception as e:
#             self._logger.error(f"Error executing tool {tool_name}: {str(e)}")
#             return None 
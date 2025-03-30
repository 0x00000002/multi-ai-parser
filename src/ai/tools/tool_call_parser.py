from typing import Dict, Any, Optional
from src.logger import Logger
import json
import re

class ToolCallParser:
    """Handles parsing of tool calls from text responses."""
    
    def __init__(self, logger: Logger):
        self._logger = logger
    
    def parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
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
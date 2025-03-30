from typing import Dict, List, Any, Optional
from src.ai.ai_config import Provider, Model, DEFAULT_TOOL_FINDER_MODEL
from src.ai.tools.tools_list import Tool
from src.ai.tools.tool_executor import ToolExecutor
from src.ai.tools.tool_call_parser import ToolCallParser
from src.ai.tools.tools_registry import ToolsRegistry
from src.ai.tools.tool_finder import ToolFinder
from src.logger import Logger, NullLogger
import json
from unittest.mock import Mock

class ToolManager:
    """Manages tool finding, selection and execution functionality."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NullLogger()
        self._tool_finder = None
        self._tool_executor = ToolExecutor(self._logger)
        self._tool_call_parser = ToolCallParser(self._logger)
        self._auto_find_tools = False
    
    def set_tool_finder(self, tool_finder: ToolFinder) -> None:
        """Set a tool finder instance."""
        self._tool_finder = tool_finder
        self._logger.info("ToolFinder has been set")
    
    def create_tool_finder(self, model: Model = DEFAULT_TOOL_FINDER_MODEL) -> None:
        """Create and set a new tool finder instance."""
        try:
            self._tool_finder = ToolFinder(model=model, logger=self._logger)
            self._logger.info(f"Created ToolFinder using {model.name}")
        except Exception as e:
            self._logger.error(f"Failed to create ToolFinder: {str(e)}")
            raise
    
    def enable_auto_tool_finding(self, enabled: bool = True) -> None:
        """Enable or disable automatic tool finding."""
        if self._tool_finder is None:
            self.create_tool_finder()
        self._auto_find_tools = enabled
        self._logger.info(f"Auto tool finding {'enabled' if enabled else 'disabled'}")
    
    def find_tools(self, user_prompt: str, conversation_history: List[str] = None) -> List[Tool]:
        """Find relevant tools for a user prompt."""
        try:
            if self._tool_finder is None:
                self.create_tool_finder()
            return self._tool_finder.find_tools(user_prompt, conversation_history)
        except Exception as e:
            self._logger.error(f"Failed to find tools: {str(e)}")
            return []
    
    def handle_tool_calls(self, messages: List[Dict[str, Any]], response: Any, ai_requester) -> Any:
        """Handle tool calls, using the provided ai_requester for follow-up requests."""
        try:
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_result = self._tool_executor.execute(tool_call.name, tool_call.arguments)
                    messages = ai_requester.add_tool_message(messages, tool_call.name, tool_result)
                    response = ai_requester.request(messages)
            else:
                # Check if there's a tool call in the response text
                tool_call_data = self._tool_call_parser.parse_tool_call(response.content)
                if tool_call_data:
                    tool_result = self._tool_executor.execute(
                        tool_call_data["name"],
                        json.dumps(tool_call_data["arguments"])
                    )
                    messages = ai_requester.add_tool_message(messages, tool_call_data["name"], tool_result)
                    response = ai_requester.request(messages)
            
            return response
        except Exception as e:
            self._logger.error(f"Failed to handle tool calls: {str(e)}")
            return response

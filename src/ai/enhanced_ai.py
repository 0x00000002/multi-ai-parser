from typing import List, Dict, Any, Optional, Union
from src.ai.ai_config import Model, Quality, Speed, DEFAULT_TOOL_FINDER_MODEL
from src.ai.model_selector import UseCase
from src.ai.errors import AI_Processing_Error
from src.logger import Logger, NullLogger
from src.ai.tools.tools_list import Tool
from src.parser import Parser
from src.ai.tools.tool_executor import ToolExecutor
from src.ai.tools.tool_call_parser import ToolCallParser
from src.ai.tools.tool_prompt_builder import ToolPromptBuilder
from src.ai.tools.tools_manager import ToolManager
import json

# Import the base class and ToolFinder
from src.ai.base_ai import AIBase, Role
from src.ai.tools.tool_finder import ToolFinder


class AI(AIBase):
    """
    Enhanced AI class that inherits from AIBase and adds tool finding capabilities.
    """
    
    def __init__(self, 
                 model_param: Union[Model, Dict[str, Any], UseCase], 
                 quality: Optional[Quality] = None,
                 speed: Optional[Speed] = None,
                 use_local: bool = False,
                 system_prompt: str = "",
                 logger: Optional[Logger] = None):
        """
        Initialize the enhanced AI with all base functionality plus tool finding.
        """
        # Initialize the base class
        super().__init__(
            model_param=model_param,
            quality=quality,
            speed=speed,
            use_local=use_local,
            system_prompt=system_prompt,
            logger=logger or NullLogger()
        )
        
        # Initialize tool manager
        self._tool_manager = ToolManager(self._logger)
    
    def set_tool_finder(self, tool_finder: ToolFinder) -> None:
        """
        Set a tool finder for this AI instance.
        
        Args:
            tool_finder: A ToolFinder instance to use
        """
        self._tool_manager.set_tool_finder(tool_finder)
    
    def create_tool_finder(self, model: Model = DEFAULT_TOOL_FINDER_MODEL) -> None:
        """
        Create and set a new tool finder instance.
        
        Args:
            model: The model to use for tool finding
        """
        self._tool_manager.create_tool_finder(model)
    
    def enable_auto_tool_finding(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic tool finding for each request.
        
        Args:
            enabled: Whether to enable automatic tool finding
        """
        self._tool_manager.enable_auto_tool_finding(enabled)
    
    def find_tools(self, user_prompt: str) -> List[Tool]:
        """
        Find relevant tools for a user prompt without making a main AI request.
        
        Args:
            user_prompt: The user's prompt
            
        Returns:
            List of Tool enum values that are relevant
        """
        return self._tool_manager.find_tools(user_prompt, self.questions[-3:])

    def _prepare_request(self, user_prompt: str) -> tuple[List[Dict[str, Any]], str]:
        """
        Prepare the request by finding tools and building messages.
        
        Args:
            user_prompt: The user's original request
            
        Returns:
            Tuple of (messages list, enhanced prompt)
        """
        # Build conversation history
        messages = []
        messages.extend(self._build_conversation_history())
        
        # Find relevant tools if auto-finding is enabled
        relevant_tools = []
        if self._tool_manager._auto_find_tools:
            relevant_tools = self._tool_manager.find_tools(user_prompt, self.questions[-3:])
        
        # Enhance the prompt with tool information if tools were found
        enhanced_prompt = user_prompt
        if relevant_tools:
            enhanced_prompt = ToolPromptBuilder.build_enhanced_prompt(user_prompt, relevant_tools)
            self._logger.debug(f"Enhanced prompt with tools: {enhanced_prompt}")
        
        # Add the current message
        messages.append(self._build_messages(enhanced_prompt, Role.USER))
        
        return messages, enhanced_prompt

    def _handle_tool_calls(self, messages: List[Dict[str, Any]], response: Any) -> Any:
        """
        Handle any tool calls in the response.
        
        Args:
            messages: The current conversation history
            response: The AI response
            
        Returns:
            The final response after handling tool calls
        """
        return self._tool_manager.handle_tool_calls(messages, response, self.ai)

    def _update_conversation_history(self, user_prompt: str, response: Any, thoughts: str) -> None:
        """
        Update the conversation history with the latest interaction.
        
        Args:
            user_prompt: The user's original prompt
            response: The AI response
            thoughts: The extracted thoughts
        """
        self.thoughts.append(thoughts)
        self.questions.append(user_prompt)  # Store original prompt
        self.responses.append(response.content if hasattr(response, 'content') else response)
    
    def request(self, user_prompt: str) -> str:
        """
        Request the AI response with optional tool finding.
        Overrides the base class method to add tool finding capabilities.
        
        Args:
            user_prompt: The user's original request
            
        Returns:
            AI response string
        """
        self._logger.debug(f"Enhanced AI requesting response for: {user_prompt}")
        
        # Prepare the request
        messages, enhanced_prompt = self._prepare_request(user_prompt)
        
        # Get response from AI
        response = self.ai.request(messages)
        
        # Handle any tool calls
        response = self._handle_tool_calls(messages, response)
        
        # Extract and process the response content
        thoughts = Parser.extract_text(response.content, "<think>", "</think>")
        self._update_conversation_history(user_prompt, response, thoughts)
        
        # If this is a tool response, return it directly
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.content
        elif hasattr(response, 'content'):
            return response.content
        return response
    
    def stream(self, user_prompt: str) -> str:
        """
        Stream the AI response with optional tool finding.
        Overrides the base class method to add tool finding capabilities.
        
        Args:
            user_prompt: The user's original request
            
        Returns:
            Streamed AI response
        """
        self._logger.debug(f"Enhanced AI streaming response for: {user_prompt}")
        
        # Prepare the request
        messages, enhanced_prompt = self._prepare_request(user_prompt)
        
        # Get streamed response from AI
        response = self.ai.stream(messages)
        
        if not response:
            raise AI_Processing_Error("No response from AI")
        
        # Check for tool calls in the response (this is tricky with streaming)
        # Consider implementing a post-stream tool execution step
        tool_result = self._tool_manager._tool_executor.execute_from_response(response)
        if tool_result.success:
            # Follow up with a non-streaming request to handle the tool
            messages = self.ai.add_tool_message(
                messages, 
                tool_result.tool_name, 
                tool_result.result
            )
            follow_up_response = self.ai.request(messages)
            response += f"\n\n[Tool executed: {tool_result.tool_name}]\n{follow_up_response.content}"
        
        # Extract and process the response content
        thoughts = Parser.extract_text(response, "<think>", "</think>")
        self._update_conversation_history(user_prompt, response, thoughts)
        
        return response
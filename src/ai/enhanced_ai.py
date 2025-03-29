from typing import List, Dict, Any, Optional, Union
import src.ai.ai_config as config
from src.ai.model_selector import UseCase
from src.ai.errors import AI_Processing_Error
from src.Logger import Logger, NullLogger
from src.ai.tools.tools_list import Tool
from src.Parser import Parser
from src.ai.tools.tool_executor import ToolExecutor
from src.ai.tools.tool_call_parser import ToolCallParser
from src.ai.tools.tool_prompt_builder import ToolPromptBuilder
import json

# Import the base class and ToolFinder
from src.ai.base_ai import AIBase, Role
from src.ai.tools.tool_finder import ToolFinder


class AI(AIBase):
    """
    Enhanced AI class that inherits from AIBase and adds tool finding capabilities.
    """
    
    def __init__(self, 
                 model_param: Union[config.Model, Dict[str, Any], UseCase], 
                 quality: Optional[config.Quality] = None,
                 speed: Optional[config.Speed] = None,
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
        
        # Add tool finder properties
        self._tool_finder = None
        self._auto_find_tools = False
        
        # Initialize tool-related components
        self._tool_executor = ToolExecutor(self._logger)
        self._tool_call_parser = ToolCallParser(self._logger)
    
    def set_tool_finder(self, tool_finder: ToolFinder) -> None:
        """
        Set a tool finder for this AI instance.
        
        Args:
            tool_finder: A ToolFinder instance to use
        """
        self._tool_finder = tool_finder
        self._logger.info("ToolFinder has been set")
    
    def create_tool_finder(self, model: config.Model = config.DEFAULT_TOOL_FINDER_MODEL) -> None:
        """
        Create and set a new tool finder instance.
        
        Args:
            model: The model to use for tool finding
        """
        self._tool_finder = ToolFinder(
            model=model,
            logger=self._logger
        )
        self._logger.info(f"Created and set ToolFinder using {model.name}")
    
    def enable_auto_tool_finding(self, enabled: bool = True) -> None:
        """
        Enable or disable automatic tool finding for each request.
        
        Args:
            enabled: Whether to enable automatic tool finding
        """
        if self._tool_finder is None:
            self.create_tool_finder()
                        
        self._auto_find_tools = enabled
        self._logger.info(f"Auto tool finding {'enabled' if enabled else 'disabled'}")
    
    def find_tools(self, user_prompt: str) -> List[Tool]:
        """
        Find relevant tools for a user prompt without making a main AI request.
        
        Args:
            user_prompt: The user's prompt
            
        Returns:
            List of Tool enum values that are relevant
        """
        if self._tool_finder is None:
            self.create_tool_finder()
            
        return self._tool_finder.find_tools(user_prompt)
    
    def _get_tool_message_role(self) -> str:
        """
        Get the appropriate role for tool messages based on the provider.
        
        Returns:
            str: The role to use for tool messages
        """
        if self.model.provider_class_name == "Ollama":
            return "tool"
        elif self.model.provider_class_name == "ClaudeAI":
            return "assistant"
        elif self.model.provider_class_name == "ChatGPT":
            return "function"
        else:
            return "assistant"  # Default fallback

    def _format_tool_message(self, name: str, content: str) -> Dict[str, str]:
        """
        Format a tool message according to the provider's requirements.
        
        Args:
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Dict[str, str]: The formatted message
        """
        role = self._get_tool_message_role()
        message = {
            "role": role,
            "content": str(content)
        }
        
        # Add name field only for providers that support it
        if role in ["function", "tool"]:
            message["name"] = name
            
        return message

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
        
        # Check if tool finding is enabled
        relevant_tools = []
        if self._auto_find_tools and self._tool_finder:
            relevant_tools = self._tool_finder.find_tools(user_prompt, self.questions[-3:])
        
        # Build conversation history
        messages = []
        messages.extend(self._build_conversation_history())
        
        # Enhance the prompt with tool information if tools were found
        enhanced_prompt = user_prompt
        if relevant_tools:
            enhanced_prompt = ToolPromptBuilder.build_enhanced_prompt(user_prompt, relevant_tools)
            self._logger.debug(f"Enhanced prompt with tools: {enhanced_prompt}")
        
        # Continue with the standard request flow but use enhanced prompt
        messages.append(self._build_messages(enhanced_prompt, Role.USER))
        response = self.ai.request(messages)
        
        # Handle tool calls if present
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = self._tool_executor.execute(tool_call.name, tool_call.arguments)
                # Let the AI module handle the tool message formatting
                messages = self.ai.add_tool_message(messages, tool_call.name, tool_result)
                # Get AI's response to the tool result
                follow_up_response = self.ai.request(messages)
                response = follow_up_response
        else:
            # Check if there's a tool call in the response text
            tool_call_data = self._tool_call_parser.parse_tool_call(response.content)
            if tool_call_data:
                tool_result = self._tool_executor.execute(
                    tool_call_data["name"],
                    json.dumps(tool_call_data["arguments"])
                )
                # Let the AI module handle the tool message formatting
                messages = self.ai.add_tool_message(messages, tool_call_data["name"], tool_result)
                # Get AI's response to the tool result
                follow_up_response = self.ai.request(messages)
                response = follow_up_response
        
        # Extract and process the response content
        thoughts = Parser.extract_text(response.content, "<think>", "</think>")
        self.thoughts.append(thoughts)
        self.questions.append(user_prompt)  # Store original prompt
        self.responses.append(response.content)
        
        # If this is a tool response, return it directly
        if hasattr(response, 'tool_calls') and response.tool_calls:
            return response.content
        elif tool_call_data:
            return tool_result
        
        return response.content
    
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
        
        # Check if tool finding is enabled
        relevant_tools = []
        if self._auto_find_tools and self._tool_finder:
            relevant_tools = self._tool_finder.find_tools(user_prompt, self.questions[-3:])
        
        # Build conversation history
        messages = []
        messages.extend(self._build_conversation_history())
        
        # Enhance the prompt with tool information if tools were found
        enhanced_prompt = user_prompt
        if relevant_tools:
            enhanced_prompt = ToolPromptBuilder.build_enhanced_prompt(user_prompt, relevant_tools)
            self._logger.debug(f"Enhanced prompt with tools: {enhanced_prompt}")
        
        # Continue with the standard stream flow but use enhanced prompt
        messages.append(self._build_messages(enhanced_prompt, Role.USER))
        response = self.ai.stream(messages)
        
        if not response:
            raise AI_Processing_Error("No response from AI")
            
        thoughts = Parser.extract_text(response, "<think>", "</think>")
        self.thoughts.append(thoughts)
        self.questions.append(user_prompt)  # Store original prompt
        self.responses.append(response)
        
        return response
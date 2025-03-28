from typing import List, Dict, Any, Optional, Set
from src.ai.tools.tools_list import Tool
from src.ai.tools.tools_registry import ToolsRegistry
from src.Prompter import Prompter
import src.ai.AIConfig as config
from src.Logger import Logger, NullLogger
import json

# Import from existing AI modules
from src.ai.modules.Anthropic import ClaudeAI
from src.ai.modules.OpenAI import ChatGPT
from src.ai.modules.Ollama import Ollama
from src.ai.modules.Google import Gemini
from enum import Enum

class Role(Enum):
    USER = "user"
    AI = "ai"

    def __str__(self):
        return self.value


class ToolFinder:
    """
    ToolFinder class that analyzes user requests and identifies appropriate tools.
    """
    
    def __init__(self, 
                 model=config.DEFAULT_TOOL_FINDER_MODEL, 
                 logger: Optional[Logger] = None):
        """
        Initialize the ToolFinder.
        
        Args:
            model: The model to use for tool finding
            logger: Logger instance to use
        """
        self._logger = logger if logger is not None else NullLogger()
        self._model = model


        # Create specialized system prompt for tool finding
        self._system_prompt = """
        You are an assistant who can find the proper tool to answer the question from user, 
        based on the user's question and provided list of tools.
        """

        # Set up the provider
        self._provider_class_name = model.provider_class_name
        provider_class = globals().get(self._provider_class_name)
        
        if not provider_class:
            self._logger.error(f"Provider class not found: {self._provider_class_name}")
            raise Exception(f"Provider class not found: {self._provider_class_name}")
            
        # Initialize the provider
        self._logger.debug(f"Initializing ToolFinder with model: {model.name}")
        self._logger.debug(f"System prompt: {self._system_prompt}")
        
        self.ai = provider_class(self._model, self._system_prompt, self._logger)
        
    def find_tools(self, user_prompt: str, conversation_history: List[str] = None) -> List[Tool]:
        """
        Analyze the user's request with optional conversation history.
        
        Args:
            user_prompt: The user's original request
            conversation_history: List of recent conversation messages (optional)
            
        Returns:
            List of Tool enum values that are relevant
        """
        user_prompt_with_history = user_prompt
        
        # Add conversation context if provided
        if conversation_history and len(conversation_history) > 0:
            context = "\n".join([f"Previous message: {msg}" for msg in conversation_history])
            user_prompt_with_history = f"Recent conversation:\n{context}\n\nCurrent request: {user_prompt}"

        self._logger.debug(f"Finding tools for request: {user_prompt}")
        
        # Create a specialized prompt for tool finding
        finder_prompt = f"""
        Search for the tool that can answer the user's question.
        Return the tool name as a string ONLY if you found the proper tool.
        Return None if no suitable tool is found.
        Analyze this user request:

        USER REQUEST: "{user_prompt_with_history}"
        
        and identify the most appropriate tools from the list of tools:

        LIST OF TOOLS: {Tool.get_tools_descriptions()}

        Return a JSON array with the list of tool names in this format:
        {{
            "tools": ["TOOL_NAME_1", "TOOL_NAME_2", "TOOL_NAME_3"]
        }}

        where TOOL_NAME is the name of the tool from the provided list of tools.
        
        Only include tools that are directly relevant to fulfilling this specific request.
        Do not include any explanations, just the JSON response.
        """
        
        # Build message for the AI provider
        content_key = "content"
        content_value = finder_prompt
        
        if self._model.provider_class_name == "Gemini":
            content_key = "parts"
            content_value = [{"text": finder_prompt}]
            
        messages = [{"role": "user", content_key: content_value}]
        
        # Get response from the model
        response = self.ai.request(messages)
        
        if not response:
            self._logger.warning("No response from ToolFinder")
            return []

        # Extract tool names from the response
        try:
            # Try to parse the response content as JSON
            try:
                result = json.loads(response.content)
                tool_names = result.get("tools", [])
            except json.JSONDecodeError:
                # If not valid JSON, try to extract tool names directly from content
                self._logger.warning("JSON not found in tool finder response, using fallback extraction")
                tool_names = []
                for tool in Tool:
                    if tool.name in response.content:
                        tool_names.append(tool.name)
            
            # Convert names to Tool enum values
            selected_tools = []
            for name in tool_names:
                try:
                    tool = Tool[name]
                    selected_tools.append(tool)
                except KeyError:
                    self._logger.warning(f"Unknown tool name: {name}")
            
            self._logger.info(f"Found {len(selected_tools)} relevant tools: {', '.join([t.name for t in selected_tools])}")
            return selected_tools
            
        except Exception as e:
            self._logger.error(f"Error extracting tools from response: {str(e)}")
            return []
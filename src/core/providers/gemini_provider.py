"""
Gemini provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config.config_manager import ConfigManager
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.models import ToolCall
from .base_provider import BaseProvider
import google.generativeai as genai


class GeminiProvider(BaseProvider):
    """Provider implementation for Google's Gemini AI."""
    
    def __init__(self, 
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            model_id: The model identifier
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        super().__init__(model_id, config_manager, logger)
        
        # Get provider configuration
        provider_config = self._config_manager.get_provider_config("gemini")
        api_key = provider_config.api_key
        
        if not api_key:
            raise AICredentialsError("No Gemini API key found")
        
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name=model_id)
            self._logger.info(f"Successfully initialized Gemini model {model_id}")
        except Exception as e:
            raise AICredentialsError(f"Failed to initialize Gemini client: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to Gemini format.
        Gemini uses a chat session with user/model roles.
        """
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map roles to Gemini format
            if role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})
            elif role == "system":
                # Prepend system message to user's first message
                # Find the first user message
                for i, m in enumerate(gemini_messages):
                    if m["role"] == "user":
                        gemini_messages[i]["parts"][0] = f"System: {content}\n\n{m['parts'][0]}"
                        break
                else:
                    # If no user message, add as a user message
                    gemini_messages.append({"role": "user", "parts": [f"System: {content}"]})
            # Tool messages are not directly supported by Gemini
        
        return gemini_messages
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Gemini API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Either a string response (when no tools are needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        try:
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                chat = self._model.start_chat()
                response = chat.send_message(messages)
            else:
                # Convert messages to Gemini format
                gemini_messages = self._format_messages(messages)
                
                # Create chat session
                chat = self._model.start_chat(history=gemini_messages[:-1])
                
                # Send the last message to get the response
                last_message = gemini_messages[-1]
                response = chat.send_message(last_message["parts"][0])
            
            # Extract content
            content = response.text
            
            # Gemini doesn't support tool calls in the same way as OpenAI/Anthropic
            # We could implement function calling through a workaround if needed
            # For now, just return the content
            return content
            
        except Exception as e:
            self._logger.error(f"Gemini request failed: {str(e)}")
            raise AIRequestError(
                f"Failed to make Gemini request: {str(e)}",
                provider="gemini",
                original_error=e
            )
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the Gemini API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Streamed response as a string
        """
        try:
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                chat = self._model.start_chat()
                response = chat.send_message(messages, stream=True)
            else:
                # Convert messages to Gemini format
                gemini_messages = self._format_messages(messages)
                
                # Create chat session
                chat = self._model.start_chat(history=gemini_messages[:-1])
                
                # Send the last message to get the response
                last_message = gemini_messages[-1]
                response = chat.send_message(last_message["parts"][0], stream=True)
            
            # Collect content chunks
            content = ""
            for chunk in response:
                if chunk.text:
                    content += chunk.text
            
            return content
            
        except Exception as e:
            self._logger.error(f"Gemini streaming failed: {str(e)}")
            raise AIRequestError(
                f"Failed to stream Gemini response: {str(e)}",
                provider="gemini",
                original_error=e
            ) 
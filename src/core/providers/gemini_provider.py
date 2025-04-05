"""
Gemini provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.tool_call import ToolCall
from .base_provider import BaseProvider
import google.generativeai as genai
import json


class GeminiProvider(BaseProvider, ToolCapableProviderInterface):
    """Provider implementation for Google's Gemini AI."""
    
    # Add property for tool support
    supports_tools = True
    
    def __init__(self, 
                 model_id: str,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Gemini provider.
        
        Args:
            model_id: The model identifier
            logger: Logger instance
        """
        super().__init__(model_id, logger)
        
        # Get provider configuration
        api_key = self.config.get_api_key("gemini")
        
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
                # Handle tool calls if present
                if "tool_calls" in msg:
                    self._logger.debug(f"Found tool_calls in assistant message: {msg['tool_calls']}")
                    parts = [content]
                    gemini_messages.append({"role": "model", "parts": parts})
                else:
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
            elif role == "tool":
                # Add tool response as a user message (workaround for Gemini)
                tool_name = msg.get("name", "unknown_tool")
                gemini_messages.append({
                    "role": "user", 
                    "parts": [f"Tool Result ({tool_name}): {content}"]
                })
        
        return gemini_messages
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Gemini API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Either a string response (when no tools were needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        try:
            # Check for tools
            tools = options.pop("tools", None)
            
            # For Gemini models, we can't directly pass tools to the API
            # Instead, we'll inject a prompt with the tool descriptions and usage instructions
            if tools and isinstance(messages, list):
                # Extract the last user message
                last_user_msg = None
                for msg in reversed(messages):
                    if msg["role"] == "user":
                        last_user_msg = msg
                        break
                
                if last_user_msg:
                    # Create a tool description
                    tool_desc = "You have access to the following tools:\n\n"
                    for tool in tools:
                        tool_desc += f"- {tool['name']}: {tool['description']}\n"
                        if 'parameters' in tool and 'properties' in tool['parameters']:
                            tool_desc += "  Parameters:\n"
                            for param_name, param_details in tool['parameters']['properties'].items():
                                param_desc = param_details.get('description', 'No description')
                                param_type = param_details.get('type', 'string')
                                tool_desc += f"    - {param_name} ({param_type}): {param_desc}\n"
                    
                    tool_desc += "\nIf you need to use a tool, respond ONLY with JSON in this exact format:\n"
                    tool_desc += '{"tool": "<tool_name>", "parameters": {"param1": "value1", "param2": "value2"}}\n'
                    tool_desc += "\nDO NOT include any explanation, code blocks, or text before or after the JSON. ONLY output the raw JSON if you need to use a tool."
                    
                    # Append to the last user message
                    last_user_msg["content"] = f"{last_user_msg['content']}\n\n{tool_desc}"
                    self._logger.debug(f"Enhanced user message with tool instructions")
            
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
            
            # Try to identify tool calls in the text response
            if tools and "{" in content and "}" in content:
                try:
                    # First try to parse the entire response as JSON
                    content_cleaned = content.strip()
                    # Remove any markdown code block markers
                    if content_cleaned.startswith("```") and content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[3:-3].strip()
                    if content_cleaned.startswith("```json") and content_cleaned.endswith("```"):
                        content_cleaned = content_cleaned[7:-3].strip()
                    
                    # Try to parse cleaned content as JSON
                    try:
                        data = json.loads(content_cleaned)
                        if "tool" in data and "parameters" in data:
                            tool_name = data["tool"]
                            params = data["parameters"]
                            
                            # Create a tool call
                            tool_call = ToolCall(
                                name=tool_name,
                                arguments=params,  # Pass as dict, not JSON string
                                id=f"tool-{tool_name}"
                            )
                            
                            self._logger.info(f"Detected tool call in Gemini response: {tool_name}")
                            self._logger.debug(f"Tool parameters: {params}")
                            # Return with the tool call
                            return {
                                "content": content,
                                "tool_calls": [tool_call]
                            }
                    except json.JSONDecodeError:
                        # If that fails, try to extract JSON using a better regex
                        import re
                        # This regex handles nested braces properly
                        json_pattern = r'({(?:[^{}]|(?R))*})'
                        # Fallback to simpler pattern if the recursive one isn't supported
                        try:
                            json_matches = re.findall(json_pattern, content, re.DOTALL)
                        except re.error:
                            # Use non-recursive pattern as fallback
                            json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
                            json_matches = re.findall(json_pattern, content, re.DOTALL)
                        
                        for json_str in json_matches:
                            try:
                                data = json.loads(json_str)
                                if "tool" in data and "parameters" in data:
                                    tool_name = data["tool"]
                                    params = data["parameters"]
                                    
                                    # Create a tool call
                                    tool_call = ToolCall(
                                        name=tool_name,
                                        arguments=params,  # Pass as dict, not JSON string
                                        id=f"tool-{tool_name}"
                                    )
                                    
                                    self._logger.info(f"Detected tool call in Gemini response: {tool_name}")
                                    self._logger.debug(f"Tool parameters: {params}")
                                    # Return with the tool call
                                    return {
                                        "content": content,
                                        "tool_calls": [tool_call]
                                    }
                            except Exception as json_err:
                                self._logger.debug(f"Failed to parse JSON chunk: {json_err}")
                                continue
                except Exception as e:
                    self._logger.warning(f"Failed to parse potential tool calls: {str(e)}")
            
            # No tool calls detected, return just the content
            return self.standardize_response(content)
            
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
    
    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content)
        })
        return messages 
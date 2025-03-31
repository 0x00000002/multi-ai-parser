"""
AI implementation with tool-calling capabilities.
Extends the base AI with the ability to call tools to handle complex tasks.
"""
from typing import Dict, List, Any, Optional, Union, Callable
import json
import inspect
from .base_ai import AIBase
from ..exceptions import AIProcessingError, AIToolError
from ..tools.models import ToolCall
from ..prompts import PromptManager
from ..conversation.response_parser import ResponseParser


class AI(AIBase):
    """
    AI implementation with tool-calling capabilities.
    Can register and call tools based on the model's output.
    """
    
    def __init__(self, 
                 model=None, 
                 system_prompt=None, 
                 config_manager=None,
                 logger=None,
                 request_id=None,
                 prompt_manager=None,
                 auto_find_tools=False):
        """
        Initialize the tool-enabled AI.
        
        Args:
            model: The model to use (Model enum or string ID)
            system_prompt: Custom system prompt (or None for default)
            config_manager: Configuration manager instance
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            prompt_manager: PromptManager instance for templated prompts
            auto_find_tools: Whether to automatically find tools in the current scope
        """
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            config_manager=config_manager,
            logger=logger,
            request_id=request_id,
            prompt_manager=prompt_manager
        )
        
        self._tools = {}
        self._tool_schemas = {}
        
        # Check if the provider supports tool calling
        self._supports_tools = hasattr(self._provider, "supports_tools") and self._provider.supports_tools
        if not self._supports_tools:
            self._logger.warning(f"Provider {self._model_config.provider} does not fully support tools")
        
        # Automatically find tools if requested
        if auto_find_tools:
            self._find_tools()
    
    def register_tool(self, 
                     tool_name: str, 
                     tool_function: Callable, 
                     description: str = None, 
                     parameters_schema: Dict = None) -> None:
        """
        Register a tool that can be called by the AI.
        
        Args:
            tool_name: Name of the tool
            tool_function: Function to call
            description: Tool description
            parameters_schema: JSON schema for parameters (or None to infer)
        """
        self._tools[tool_name] = tool_function
        
        # If no parameters schema provided, try to infer from function signature
        if parameters_schema is None:
            parameters_schema = self._infer_schema(tool_function)
        
        # Use function docstring as description if not provided
        if description is None and tool_function.__doc__:
            description = tool_function.__doc__.strip()
        
        self._tool_schemas[tool_name] = {
            "name": tool_name,
            "description": description or f"Call {tool_name} function",
            "parameters": parameters_schema
        }
        
        self._logger.info(f"Registered tool: {tool_name}")
    
    def _infer_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Infer JSON schema from function signature.
        
        Args:
            func: Function to inspect
            
        Returns:
            JSON schema for function parameters
        """
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for name, param in sig.parameters.items():
            # Skip self for methods
            if name == "self":
                continue
                
            properties[name] = {"type": "string"}
            
            # Handle type annotations
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    properties[name]["type"] = "integer"
                elif param.annotation == float:
                    properties[name]["type"] = "number"
                elif param.annotation == bool:
                    properties[name]["type"] = "boolean"
                elif param.annotation == list:
                    properties[name]["type"] = "array"
                elif param.annotation == dict:
                    properties[name]["type"] = "object"
            
            # Add to required list if no default value
            if param.default == inspect.Parameter.empty:
                required.append(name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    def _find_tools(self) -> None:
        """Find tools in the current scope."""
        # This is a placeholder - would need to implement tool discovery
        self._logger.info("Tool discovery not yet implemented")
    
    def _call_tool(self, tool_call: ToolCall) -> str:
        """
        Call a registered tool.
        
        Args:
            tool_call: Tool call object
            
        Returns:
            Tool response as string
            
        Raises:
            AIToolError: If tool execution fails
        """
        tool_name = tool_call.name
        
        if tool_name not in self._tools:
            raise AIToolError(f"Tool not found: {tool_name}")
        
        try:
            # Parse arguments - handle both string and dict formats
            if isinstance(tool_call.arguments, str):
                try:
                    args = json.loads(tool_call.arguments)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as a string
                    self._logger.warning(f"Invalid JSON arguments: {tool_call.arguments}, trying to parse as is")
                    args = tool_call.arguments
            else:
                # Already a dict
                args = tool_call.arguments
            
            # Call the tool function
            self._logger.info(f"Calling tool: {tool_name} with args: {args}")
            result = self._tools[tool_name](**args)
            
            # Convert result to string if needed
            if not isinstance(result, str):
                result = json.dumps(result)
                
            return result
            
        except Exception as e:
            self._logger.error(f"Tool execution failed: {str(e)}")
            raise AIToolError(f"Failed to execute tool {tool_name}: {str(e)}")
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a tool-enabled request to the AI model.
        Handles tool calling and follow-up requests automatically.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's final response, after any tool calls are processed
            
        Raises:
            AIProcessingError: If the request fails
        """
        self._logger.info(f"Processing tool-enabled request: {prompt[:50]}...")
        
        # Set up tool definitions if supported
        if self._supports_tools and self._tool_schemas:
            options["tools"] = list(self._tool_schemas.values())
        
        try:
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Make initial request
            response = self._provider.request(self._conversation_manager.get_messages(), **options)
            
            # If the response is a string, it means no tools were needed
            if isinstance(response, str):
                # Process the string response to handle thoughts
                # parser = ResponseParser(logger=self._logger)
                # parsed = parser.parse_response(
                #     response,
                #     extract_thoughts=True,
                #     show_thinking=self._config_manager.show_thinking
                # )
                
                # Get the processed content
                processed_content = response
                
                # Add assistant message with potential thoughts
                self._conversation_manager.add_message(
                    role="assistant",
                    content=processed_content,
                    extract_thoughts=False  # Already extracted
                )
                
                # Update conversation history - Removed as messages are added individually
                # self._conversation_manager.add_interaction(
                #     prompt, 
                #     processed_content,
                #     extract_thoughts=False  # Already extracted
                # )
                
                return processed_content
            
            # Handle tool calls if present
            if response.get('tool_calls'):
                # Add assistant message with tool calls
                self._conversation_manager.add_message(
                    role="assistant",
                    content=response.get('content', ''),
                    tool_calls=response.get('tool_calls', [])
                )
                
                # Process each tool call
                for tool_call in response.get('tool_calls', []):
                    try:
                        # Call the tool
                        tool_result = self._call_tool(tool_call)
                        
                        # Add tool result to conversation
                        self._conversation_manager.add_message(
                            role="tool",
                            name=tool_call.name,
                            content=tool_result
                        )
                    except AIToolError as e:
                        # Add error message as tool result
                        self._conversation_manager.add_message(
                            role="tool",
                            name=tool_call.name,
                            content=f"Error: {str(e)}"
                        )
                
                # Make follow-up request with tool results
                follow_up = self._provider.request(self._conversation_manager.get_messages(), **options)
                
                # If the follow-up is a string (no more tools)
                if isinstance(follow_up, str):
                    final_content = follow_up
                else:
                    # Get content from follow-up response
                    final_content = follow_up.get('content', '')
                
                # Add final assistant message
                self._conversation_manager.add_message(
                    role="assistant",
                    content=final_content
                )
                
                # Update conversation history - Removed as messages are added individually
                # self._conversation_manager.add_interaction(prompt, final_content)
                
                return final_content
            
            # Fallback to handling standard response
            # Removed redundant parsing here, add_message handles it
            # parser = ResponseParser(logger=self._logger)
            raw_content = response.get('content', '')
            # parsed = parser.parse_response(
            #     raw_content,
            #     extract_thoughts=True,
            #     show_thinking=self._config_manager.show_thinking
            # )
            # final_content = parsed["content"]

            # Add assistant message (letting manager handle internal storage and parsing)
            self._conversation_manager.add_message(
                role="assistant",
                content=raw_content, # Pass raw content for internal parsing
                extract_thoughts=True,
                show_thinking=self._config_manager.show_thinking
            )

            # Update conversation history - Removed as messages are added individually
            # self._conversation_manager.add_interaction(
            #     prompt,
            #     raw_content,
            #     extract_thoughts=True,
            #     show_thinking=self._config_manager.show_thinking
            # )

            # Need to return the parsed content, which add_message doesn't return
            # Let's get the last message added which contains the parsed content
            last_message = self._conversation_manager.get_last_message()
            final_content = last_message['content'] if last_message else ''

            return final_content
            
        except Exception as e:
            self._logger.error(f"Tool-enabled request failed: {str(e)}")
            raise AIProcessingError(f"Failed to process tool-enabled request: {str(e)}")
    
    def request_with_template(self, 
                             template_id: str, 
                             variables: Optional[Dict[str, Any]] = None,
                             user_id: Optional[str] = None,
                             **options) -> str:
        """
        Make a tool-enabled request using a prompt template.
        
        Args:
            template_id: ID of the template to use
            variables: Variables for the template
            user_id: User ID for A/B testing
            options: Additional options for the request
            
        Returns:
            The model's response as a string
            
        Raises:
            ValueError: If prompt_manager is not set or template not found
        """
        if not self._prompt_manager:
            raise ValueError("No prompt manager set. Use set_prompt_manager() first.")
        
        # Start timing for metrics
        import time
        start_time = time.time()
        
        # Get rendered prompt from template
        rendered_prompt, usage_id = self._prompt_manager.render_prompt(
            template_id=template_id,
            variables=variables,
            user_id=user_id,
            context={"model": self._model_config.model_id, "tools": list(self._tool_schemas.keys())}
        )
        
        self._logger.info(f"Using template {template_id} with variables: {variables}")
        
        # Set up tool definitions if supported
        if self._supports_tools and self._tool_schemas:
            options["tools"] = list(self._tool_schemas.values())
        
        # Make the request with the rendered prompt
        response = self.request(rendered_prompt, **options)
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        
        # Record metrics if we have a usage ID
        if usage_id:
            metrics = {
                "latency": latency,
                "model": self._model_config.model_id,
                "tools_enabled": self._supports_tools,
                "tool_count": len(self._tool_schemas)
            }
            
            # Record performance
            self._prompt_manager.record_prompt_performance(
                usage_id=usage_id,
                metrics=metrics
            )
        
        return response
    
    def get_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered tools and their schemas.
        
        Returns:
            Dictionary of tool schemas
        """
        return self._tool_schemas
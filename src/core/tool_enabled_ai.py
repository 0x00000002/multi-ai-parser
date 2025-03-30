"""
Tool-enabled AI implementation.
Extends the base AI with tool execution capabilities.
"""
from typing import Dict, List, Any, Optional, Union, Set
from .interfaces import ToolCapableProviderInterface, LoggerInterface
from .base_ai import AIBase
from ..config.config_manager import ConfigManager
from ..exceptions import AISetupError, AIProcessingError, AIToolError
from ..tools.tool_manager import ToolManager
from ..tools.tool_registry import ToolRegistry
from ..tools.tool_finder import ToolFinder
from ..tools.models import ToolResult
from ..utils.logger import LoggerFactory
from typing import Dict, List, Any, Optional, Union
from .interfaces import ToolCapableProviderInterface, LoggerInterface
from .base_ai import AIBase
from ..config.config_manager import ConfigManager
from ..exceptions import AISetupError, AIProcessingError, AIToolError
from ..tools.tool_manager import ToolManager
from ..tools.tool_registry import ToolRegistry
from ..tools.tool_finder import ToolFinder
from ..utils.logger import LoggerFactory


class ToolEnabledAI(AIBase):
    """
    Tool-enabled AI that can use tools to enhance responses.
    """
    
    def __init__(self, 
                 model_id: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 config_manager: Optional[ConfigManager] = None,
                 logger: Optional[LoggerInterface] = None,
                 tool_manager: Optional[ToolManager] = None,
                 auto_find_tools: bool = False,
                 tool_finder_model_id: Optional[str] = None,
                 request_id: Optional[str] = None):
        """
        Initialize the tool-enabled AI.
        
        Args:
            model_id: The model to use (or None for default)
            system_prompt: Custom system prompt (or None for default)
            config_manager: Configuration manager instance
            logger: Logger instance
            tool_manager: Tool manager instance
            auto_find_tools: Whether to automatically find relevant tools
            tool_finder_model_id: Model ID to use for tool finding
            request_id: Unique identifier for tracking this session
            
        Raises:
            AISetupError: If initialization fails
        """
        # Initialize base AI
        super().__init__(
            model_id=model_id,
            system_prompt=system_prompt,
            config_manager=config_manager,
            logger=logger,
            request_id=request_id
        )
        
        # Check if the provider supports tools
        if not isinstance(self._provider, ToolCapableProviderInterface):
            self._logger.warning(f"Provider {self._model_config.provider} does not fully support tools")
        
        # Set up tool manager
        self._tool_manager = tool_manager or ToolManager(
            logger=self._logger,
            config_manager=self._config_manager
        )
        
        # Configure tool finder if auto-finding is enabled
        if auto_find_tools:
            self.enable_auto_tool_finding(
                enabled=True,
                tool_finder_model_id=tool_finder_model_id
            )
    
    def enable_auto_tool_finding(self, enabled: bool = True, 
                                tool_finder_model_id: Optional[str] = None) -> None:
        """
        Enable or disable automatic tool finding.
        
        Args:
            enabled: Whether to enable automatic tool finding
            tool_finder_model_id: Model ID to use for tool finding
        """
        self._tool_manager.enable_auto_tool_finding(
            enabled=enabled,
            tool_finder_model_id=tool_finder_model_id
        )
        self._logger.info(f"Auto tool finding {'enabled' if enabled else 'disabled'}")
    
    def register_tool(self, tool_name: str, tool_function: callable, 
                     description: str, parameters_schema: Dict[str, Any]) -> None:
        """
        Register a custom tool.
        
        Args:
            tool_name: Unique name for the tool
            tool_function: The function implementing the tool
            description: Description of what the tool does
            parameters_schema: JSON schema for the tool parameters
        """
        self._tool_manager.register_tool(
            tool_name=tool_name,
            tool_function=tool_function,
            description=description,
            parameters_schema=parameters_schema
        )
        self._logger.info(f"Tool registered: {tool_name}")
    
    def find_tools(self, prompt: str) -> Set[str]:
        """
        Find relevant tools for a given prompt.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Set of tool names that are relevant to the prompt
        """
        return self._tool_manager.find_tools(prompt)
    
    def execute_tool(self, tool_name: str, **args) -> ToolResult:
        """
        Execute a specific tool directly.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            AIToolError: If tool execution fails
        """
        try:
            return self._tool_manager.execute_tool(tool_name, **args)
        except Exception as e:
            self._logger.error(f"Tool execution failed: {str(e)}")
            raise AIToolError(f"Failed to execute tool {tool_name}: {str(e)}")
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a request with tool capabilities.
        Overrides the base implementation to add tool handling.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response as a string
        """
        self._logger.info(f"Processing tool-enabled request: {prompt[:50]}...")
        
        try:
            # Find relevant tools if auto-finding is enabled
            enhanced_prompt = prompt
            if self._tool_manager.auto_find_tools:
                tool_names = self._tool_manager.find_tools(prompt)
                if tool_names:
                    self._logger.info(f"Found relevant tools: {', '.join(tool_names)}")
                    enhanced_prompt = self._tool_manager.enhance_prompt(prompt, tool_names)
            
            # Build messages with conversation history
            messages = self._build_messages(enhanced_prompt)
            
            # Make the request to the provider
            response = self._provider.request(messages, **options)
            
            # Handle tool calls if present
            if self._has_tool_calls(response):
                self._logger.info("Processing tool calls in response")
                response = self._process_tool_calls(messages, response)
            
            # Extract content from response
            content = self._extract_content(response)
            
            # Update conversation history with original prompt and final response
            self._conversation_manager.add_interaction(prompt, content)
            
            return content
            
        except Exception as e:
            self._logger.error(f"Tool-enabled request failed: {str(e)}")
            raise AIProcessingError(f"Request failed: {str(e)}")
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """Check if the response contains tool calls."""
        # This implementation depends on the specific response format
        # A more robust implementation would handle different formats
        return 'tool_calls' in response and response['tool_calls']
    
    def _process_tool_calls(self, messages: List[Dict[str, Any]], 
                           response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process tool calls in the response.
        
        Args:
            messages: The current messages list
            response: The response containing tool calls
            
        Returns:
            Updated response after tool execution
        """
        for tool_call in response.get('tool_calls', []):
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})
            
            try:
                # Execute the tool
                tool_result = self._tool_manager.execute_tool(tool_name, **tool_args)
                
                # Add tool result to messages
                messages = self._provider.add_tool_message(
                    messages=messages,
                    name=tool_name,
                    content=str(tool_result.result)
                )
                
                # Get follow-up response
                response = self._provider.request(messages)
                
                # Check for additional tool calls recursively
                if self._has_tool_calls(response):
                    return self._process_tool_calls(messages, response)
                
            except Exception as e:
                self._logger.error(f"Tool execution failed: {str(e)}")
                # Continue with next tool call or return current response
        
        return response
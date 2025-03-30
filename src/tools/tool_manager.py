"""
Tool manager for coordinating tool registration, discovery, and execution.
"""
from typing import Dict, List, Any, Optional, Set, Union, Callable
from ..core.interfaces import LoggerInterface, ToolStrategy
from ..config.config_manager import ConfigManager
from ..exceptions import AIToolError
from .tool_finder import ToolFinder
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor
from .tool_prompt_builder import ToolPromptBuilder
from .models import ToolResult
from ..utils.logger import LoggerFactory


class ToolManager:
    """
    Centralized tool management system.
    Coordinates tool registration, discovery, and execution.
    """
    
    def __init__(self, 
                 logger: Optional[LoggerInterface] = None,
                 config_manager: Optional[ConfigManager] = None,
                 tool_registry: Optional[ToolRegistry] = None,
                 tool_finder: Optional[ToolFinder] = None,
                 tool_executor: Optional[ToolExecutor] = None):
        """
        Initialize the tool manager.
        
        Args:
            logger: Logger instance
            config_manager: Configuration manager
            tool_registry: Tool registry instance
            tool_finder: Tool finder instance
            tool_executor: Tool executor instance
        """
        self._logger = logger or LoggerFactory.create()
        self._config_manager = config_manager or ConfigManager()
        self._tool_registry = tool_registry or ToolRegistry(self._logger)
        self._tool_executor = tool_executor or ToolExecutor(self._logger)
        self._tool_finder = tool_finder
        self._auto_find_tools = False
    
    def enable_auto_tool_finding(self, enabled: bool = True, 
                                tool_finder_model_id: Optional[str] = None) -> None:
        """
        Enable or disable automatic tool finding.
        
        Args:
            enabled: Whether to enable automatic tool finding
            tool_finder_model_id: Model ID to use for tool finding
        """
        if enabled and not self._tool_finder:
            self._tool_finder = ToolFinder(
                model_id=tool_finder_model_id,
                config_manager=self._config_manager,
                logger=self._logger
            )
        self._auto_find_tools = enabled
        self._logger.info(f"Auto tool finding {'enabled' if enabled else 'disabled'}")
    
    def register_tool(self, tool_name: str, tool_function: Callable, 
                     description: str, parameters_schema: Dict[str, Any]) -> None:
        """
        Register a custom tool.
        
        Args:
            tool_name: Unique name for the tool
            tool_function: The function implementing the tool
            description: Description of what the tool does
            parameters_schema: JSON schema for the tool parameters
            
        Raises:
            AIToolError: If registration fails
        """
        try:
            self._tool_registry.register_tool(
                tool_name=tool_name,
                tool_function=tool_function,
                description=description,
                parameters_schema=parameters_schema
            )
            self._logger.info(f"Tool registered: {tool_name}")
        except Exception as e:
            self._logger.error(f"Tool registration failed: {str(e)}")
            raise AIToolError(f"Failed to register tool {tool_name}: {str(e)}")
    
    @property
    def auto_find_tools(self) -> bool:
        """Get auto tool finding setting."""
        return self._auto_find_tools
    
    def find_tools(self, prompt: str, conversation_history: Optional[List[str]] = None) -> Set[str]:
        """
        Find relevant tools for a given prompt.
        
        Args:
            prompt: The user prompt
            conversation_history: Optional list of recent conversation messages
            
        Returns:
            Set of tool names that are relevant to the prompt
            
        Raises:
            AIToolError: If tool finding fails or is not configured
        """
        if not self._tool_finder:
            raise AIToolError("Tool finder not configured")
        
        try:
            return self._tool_finder.find_tools(prompt, conversation_history)
        except Exception as e:
            self._logger.error(f"Tool finding failed: {str(e)}")
            return set()  # Return empty set on error
    
    def execute_tool(self, tool_name: str, **args) -> ToolResult:
        """
        Execute a specific tool.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            
        Returns:
            Tool execution result
            
        Raises:
            AIToolError: If tool execution fails
        """
        try:
            # Get tool from registry
            tool = self._tool_registry.get_tool(tool_name)
            if not tool:
                raise AIToolError(f"Tool not found: {tool_name}")
            
            # Execute the tool
            return self._tool_executor.execute(tool, **args)
        except Exception as e:
            self._logger.error(f"Tool execution failed: {str(e)}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                message=f"Execution failed: {str(e)}"
            )
    
    def enhance_prompt(self, prompt: str, tool_names: Set[str]) -> str:
        """
        Enhance a prompt with tool information.
        
        Args:
            prompt: The original prompt
            tool_names: Set of tool names to include
            
        Returns:
            Enhanced prompt with tool information
            
        Raises:
            AIToolError: If prompt enhancement fails
        """
        try:
            tools = [self._tool_registry.get_tool(name) for name in tool_names if self._tool_registry.get_tool(name)]
            return ToolPromptBuilder.build_enhanced_prompt(prompt, tools)
        except Exception as e:
            self._logger.error(f"Prompt enhancement failed: {str(e)}")
            raise AIToolError(f"Failed to enhance prompt: {str(e)}")
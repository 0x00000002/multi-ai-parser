"""
Tool registry for managing available tools.
"""
from typing import Dict, List, Any, Optional, Set, Callable, Union
from ..core.interfaces import LoggerInterface, ToolStrategy
from ..exceptions import AIToolError
from .models import ToolDefinition, ToolResult
from ..utils.logger import LoggerFactory


class DefaultToolStrategy(ToolStrategy):
    """Default implementation of the ToolStrategy interface."""
    
    def __init__(self, 
                 function: Callable, 
                 description: str, 
                 parameters_schema: Dict[str, Any]):
        """
        Initialize the tool strategy.
        
        Args:
            function: The function to execute
            description: Tool description
            parameters_schema: JSON schema for parameters
        """
        self._function = function
        self._description = description
        self._parameters_schema = parameters_schema
    
    def execute(self, **args) -> Any:
        """Execute the tool with the provided arguments."""
        return self._function(**args)
    
    def get_description(self) -> str:
        """Get the tool description."""
        return self._description
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the parameters schema."""
        return self._parameters_schema


class ToolRegistry:
    """
    Registry for managing tool definitions and implementations.
    """
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the tool registry.
        
        Args:
            logger: Logger instance
        """
        self._logger = logger or LoggerFactory.create()
        self._tools: Dict[str, ToolStrategy] = {}
        self._tools_metadata: Dict[str, ToolDefinition] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def _register_builtin_tools(self) -> None:
        """Register any built-in tools."""
        # This would be implemented to register default tools
        pass
    
    def register_tool(self, 
                     tool_name: str, 
                     tool_function: Callable, 
                     description: str, 
                     parameters_schema: Dict[str, Any],
                     tool_strategy: Optional[ToolStrategy] = None) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool_name: Unique name for the tool
            tool_function: The function implementing the tool
            description: Description of what the tool does
            parameters_schema: JSON schema for the tool parameters
            tool_strategy: Optional custom tool strategy implementation
            
        Raises:
            AIToolError: If registration fails (e.g., name already exists)
        """
        if tool_name in self._tools:
            raise AIToolError(f"Tool '{tool_name}' already registered")
        
        try:
            # Create tool metadata
            tool_definition = ToolDefinition(
                name=tool_name,
                description=description,
                parameters=parameters_schema
            )
            
            # Create tool implementation
            strategy = tool_strategy or DefaultToolStrategy(
                function=tool_function,
                description=description,
                parameters_schema=parameters_schema
            )
            
            # Store in registry
            self._tools[tool_name] = strategy
            self._tools_metadata[tool_name] = tool_definition
            
            self._logger.info(f"Tool registered: {tool_name}")
            
        except Exception as e:
            self._logger.error(f"Tool registration failed: {str(e)}")
            raise AIToolError(f"Failed to register tool {tool_name}: {str(e)}")
    
    def get_tool(self, tool_name: str) -> Optional[ToolStrategy]:
        """
        Get a tool implementation by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool implementation or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """
        Get tool metadata by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool definition or None if not found
        """
        return self._tools_metadata.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, ToolStrategy]:
        """
        Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to implementations
        """
        return self._tools.copy()
    
    def get_all_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """
        Get all tool definitions.
        
        Returns:
            Dictionary mapping tool names to definitions
        """
        return self._tools_metadata.copy()
    
    def format_tools_for_provider(self, 
                                 provider_name: str, 
                                 tool_names: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tool definitions for a specific provider.
        
        Args:
            provider_name: Name of the provider
            tool_names: Optional set of specific tools to format
            
        Returns:
            List of tool definitions formatted for the provider
        """
        # This implementation would handle differences in tool formats between providers
        # For now, just return a generic format
        
        formatted_tools = []
        
        # Filter tools if specific ones requested
        tools_to_format = {
            name: defn for name, defn in self._tools_metadata.items()
            if tool_names is None or name in tool_names
        }
        
        for tool_name, tool_def in tools_to_format.items():
            formatted_tools.append({
                "name": tool_name,
                "description": tool_def.description,
                "parameters": tool_def.parameters
            })
            
        return formatted_tools
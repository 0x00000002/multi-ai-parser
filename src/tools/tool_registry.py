"""
Tool registry for managing available tools.
"""
from typing import Dict, List, Any, Optional, Set, Callable, Union
from .interfaces import ToolStrategy
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIToolError
from .models import ToolDefinition, ToolResult
from ..config.config_manager import Provider  # Import Provider enum


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
                parameters_schema=parameters_schema,
                function=tool_function
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
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool exists in the registry.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool exists, False otherwise
        """
        return tool_name in self._tools
    
    def get_tool(self, tool_name: str) -> Optional[ToolStrategy]:
        """
        Get a tool implementation by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool implementation or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        Get the description of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool description or None if not found
        """
        tool = self._tools.get(tool_name)
        return tool.get_description() if tool else None
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the parameters schema of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool parameters schema or None if not found
        """
        tool = self._tools.get(tool_name)
        return tool.get_schema() if tool else None
    
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
        Format tool definitions for a specific provider's API.

        Args:
            provider_name: Name of the provider (e.g., "ANTHROPIC", "OPENAI", "GOOGLE").
            tool_names: Optional set of specific tools to format. If None, format all.

        Returns:
            List of tool definitions formatted for the provider.
            Returns an empty list if the provider is unknown or formatting fails.
        """
        formatted_tools = []
        provider = Provider.from_string(provider_name) # Use the enum helper

        if not provider:
            self._logger.warning(f"Unknown provider '{provider_name}' for tool formatting.")
            return []

        # Filter tools if specific ones requested
        tools_to_format = {
            name: defn for name, defn in self._tools_metadata.items()
            if tool_names is None or name in tool_names
        }

        for tool_name, tool_def in tools_to_format.items():
            try:
                # Common structure
                base_tool_format = {
                    "name": tool_name,
                    "description": tool_def.description,
                }

                # Provider-specific parameter schema structure
                if provider in [Provider.OPENAI_GPT_4O, Provider.OPENAI_GPT_4O_MINI, Provider.OPENAI_GPT_4_TURBO]:
                    base_tool_format["parameters"] = tool_def.parameters_schema
                    formatted_tools.append({"type": "function", "function": base_tool_format})
                elif provider in [Provider.ANTHROPIC_CLAUDE_3_5_SONNET, Provider.ANTHROPIC_CLAUDE_3_OPUS, Provider.ANTHROPIC_CLAUDE_3_HAIKU]:
                     # Anthropic uses 'input_schema'
                    base_tool_format["input_schema"] = tool_def.parameters_schema
                    formatted_tools.append(base_tool_format)
                elif provider in [Provider.GOOGLE_GEMINI_1_5_PRO, Provider.GOOGLE_GEMINI_1_5_FLASH, Provider.GOOGLE_GEMINI_2_5_PRO]:
                     # Gemini uses 'parameters'
                    base_tool_format["parameters"] = tool_def.parameters_schema
                    # Wrap in 'function_declaration' for Gemini
                    formatted_tools.append({"function_declaration": base_tool_format})
                else:
                    # Fallback for other providers or default
                    self._logger.warning(f"Using default tool format for provider: {provider.name}")
                    base_tool_format["parameters"] = tool_def.parameters_schema # Corrected attribute access
                    formatted_tools.append(base_tool_format)

            except Exception as e:
                 self._logger.error(f"Failed to format tool '{tool_name}' for provider '{provider_name}': {e}")
                 # Optionally skip this tool or return [] depending on desired robustness

        return formatted_tools 
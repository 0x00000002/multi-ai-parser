"""
Tool Manager Module

This module provides a manager for coordinating tool operations in the Agentic-AI framework.
"""
from typing import Dict, Any, List, Optional, Set, Union, TYPE_CHECKING
import json
import time

from src.utils.logger import LoggerFactory
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_executor import ToolExecutor
from src.tools.models import ToolDefinition, ToolResult, ToolExecutionStatus
from src.exceptions import AIToolError
from src.config.unified_config import UnifiedConfig

if TYPE_CHECKING:
    from src.agents.tool_finder_agent import ToolFinderAgent
    from src.agents import AgentFactory


class ToolManager:
    """
    Manager for coordinating tool operations in the Agentic-AI framework.
    
    This class coordinates tool registration, discovery, and execution.
    It works with the ToolRegistry to maintain tool definitions and usage statistics,
    and with the ToolFinderAgent to find relevant tools for user requests.
    """
    
    def __init__(self, unified_config=None, logger=None, tool_registry=None, tool_executor=None, agent_factory=None):
        """
        Initialize the tool manager.
        
        Args:
            unified_config: Optional UnifiedConfig instance
            logger: Optional logger instance
            tool_registry: Optional tool registry
            tool_executor: Optional tool executor
            agent_factory: Optional agent factory for creating the ToolFinderAgent
        """
        self.logger = logger or LoggerFactory.create("tool_manager")
        self.config = unified_config or UnifiedConfig.get_instance()
        
        # Load tool configuration
        self.tool_config = self.config.get_tool_config()
        
        # Initialize tool registry and executor with config
        self.tool_registry = tool_registry or ToolRegistry(logger=self.logger)
        
        # Configure tool executor with settings from config
        executor_config = self.tool_config.get("execution", {})
        self.tool_executor = tool_executor or ToolExecutor(
            logger=self.logger,
            timeout=executor_config.get("timeout", 30),
            max_retries=executor_config.get("max_retries", 3)
        )
        
        self.agent_factory = agent_factory
        
        # Tool finder agent
        self.tool_finder_agent = None
        
        # Initialize stats configuration
        self._init_stats_config()
        
        self.logger.info("Tool manager initialized")
        
    def _init_stats_config(self):
        """Initialize statistics configuration from the config file."""
        stats_config = self.tool_config.get("stats", {})
        self.stats_storage_path = stats_config.get("storage_path", "data/tool_stats.json")
        
        # If stats tracking is enabled and a storage path is specified, try to load
        if stats_config.get("track_usage", True) and self.stats_storage_path:
            try:
                self.load_usage_stats(self.stats_storage_path)
            except Exception as e:
                self.logger.warning(f"Failed to load tool usage stats: {str(e)}")
                
    def register_tool(self, tool_name: str, tool_definition: ToolDefinition) -> None:
        """
        Register a tool with the tool registry.
        
        Args:
            tool_name: Name of the tool
            tool_definition: Tool definition object
        """
        self.tool_registry.register_tool(tool_name, tool_definition)
        
        # If tool finder agent exists, update its available tools
        if self.tool_finder_agent:
            self.logger.debug(f"Updating tool finder agent with new tool: {tool_name}")
    
    def enable_agent_based_tool_finding(self, ai_instance) -> None:
        """
        Enable agent-based tool finding using the ToolFinderAgent.
        
        Args:
            ai_instance: AI instance for the ToolFinderAgent
        """
        # Check if tool finder is enabled in the configuration
        finder_config = self.tool_config.get("finder_agent", {})
        if not finder_config.get("enabled", True):
            self.logger.info("Agent-based tool finding is disabled in configuration")
            return
            
        if not self.agent_factory:
            self.logger.warning("Agent factory not provided, cannot create ToolFinderAgent")
            return
            
        try:
            # Import here to avoid circular dependency
            from src.agents import AgentFactory
            
            # Create the ToolFinderAgent with configuration
            self.tool_finder_agent = self.agent_factory.create_agent(
                "tool_finder",
                ai_instance=ai_instance,
                tool_registry=self.tool_registry,
                max_recommendations=finder_config.get("max_recommendations", 5),
                use_history=finder_config.get("use_history", True)
            )
            
            self.logger.info("Agent-based tool finding enabled")
        except Exception as e:
            self.logger.error(f"Failed to create ToolFinderAgent: {str(e)}")
    
    def find_tools(self, prompt: str, conversation_history: Optional[List[str]] = None) -> List[str]:
        """
        Find relevant tools for a given prompt.
        
        Args:
            prompt: User prompt to analyze
            conversation_history: Optional conversation history
            
        Returns:
            List of relevant tool names
        """
        if not self.tool_finder_agent:
            self.logger.warning("Tool finder agent not enabled, using registry recommendations")
            # Get the max recommendations from config
            finder_config = self.tool_config.get("finder_agent", {})
            max_tools = finder_config.get("max_recommendations", 5)
            return self.tool_registry.get_recommended_tools(prompt, max_tools=max_tools)
            
        try:
            # Create a request object for the agent
            request = type('Request', (), {
                'prompt': prompt,
                'conversation_history': conversation_history or []
            })
            
            # Process the request with the tool finder agent
            response = self.tool_finder_agent.process_request(request)
            
            if response.status == "success":
                return response.selected_tools
            else:
                self.logger.warning(f"Tool finder agent returned error: {response.content}")
                return []
        except Exception as e:
            self.logger.error(f"Error finding tools: {str(e)}")
            return []
    
    def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters for the tool
            
        Returns:
            ToolResult object with execution results
        """
        try:
            # Get the tool definition
            tool_definition = self.tool_registry.get_tool(tool_name)
            
            if not tool_definition:
                return ToolResult(
                    status=ToolExecutionStatus.ERROR,
                    error=f"Tool not found: {tool_name}",
                    result=None
                )
                
            # Execute the tool using configs from tool_config if applicable
            tool_specific_config = self.config.get_tool_config(tool_name)
            execution_params = {**kwargs}
            
            # Add any tool-specific configuration parameters
            if tool_specific_config:
                for param, value in tool_specific_config.items():
                    if param not in execution_params:
                        execution_params[param] = value
            
            # Track start time for metrics
            start_time = time.time()
            
            # Extract request_id if present for metrics tracking
            request_id = kwargs.get("request_id")
            
            # Execute the tool
            result = self.tool_executor.execute(tool_definition, **execution_params)
            
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Update usage stats if tracking is enabled
            stats_config = self.tool_config.get("stats", {})
            if stats_config.get("track_usage", True):
                success = result.status == ToolExecutionStatus.SUCCESS
                self.tool_registry.update_usage_stats(
                    tool_name, 
                    success,
                    request_id=request_id,
                    duration_ms=execution_time_ms
                )
            
            return result
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error=str(e),
                result=None
            )
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary with tool information or None if not found
        """
        tool_definition = self.tool_registry.get_tool(tool_name)
        
        if not tool_definition:
            return None
            
        # Get usage stats
        usage_stats = self.tool_registry.get_usage_stats(tool_name)
        
        # Get any additional tool configuration
        tool_config = self.config.get_tool_config(tool_name)
        
        return {
            "name": tool_name,
            "description": tool_definition.description,
            "parameters": tool_definition.parameters,
            "usage_stats": usage_stats,
            "config": tool_config
        }
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered tools.
        
        Returns:
            Dictionary mapping tool names to tool information
        """
        tools = {}
        
        for tool_name in self.tool_registry.get_tool_names():
            tools[tool_name] = self.get_tool_info(tool_name)
            
        return tools
    
    def format_tools_for_model(self, model_id: str, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tools for use with a specific model.
        
        Args:
            model_id: Model identifier
            tool_names: Optional list of tool names to format
            
        Returns:
            List of formatted tools for the model's provider
        """
        # Get the model configuration
        model_config = self.config.get_model_config(model_id)
        if not model_config or "provider" not in model_config:
            self.logger.warning(f"Unable to determine provider for model {model_id}, using default tool format")
            return []
            
        # Get the provider from the model configuration
        provider_name = model_config["provider"].upper()
            
        # Convert list to set if provided
        tool_names_set = set(tool_names) if tool_names else None
        
        # Format tools for the provider
        return self.tool_registry.format_tools_for_provider(provider_name, tool_names_set)
    
    def save_usage_stats(self, file_path: str = None) -> None:
        """
        Save usage statistics to a file.
        
        Args:
            file_path: Path to save the statistics to, uses config path if None
        """
        # Use the configured path if none provided
        if file_path is None:
            stats_config = self.tool_config.get("stats", {})
            file_path = stats_config.get("storage_path", "data/tool_stats.json")
            
        self.tool_registry.save_stats(file_path)
    
    def load_usage_stats(self, file_path: str = None) -> None:
        """
        Load usage statistics from a file.
        
        Args:
            file_path: Path to load the statistics from, uses config path if None
        """
        # Use the configured path if none provided
        if file_path is None:
            stats_config = self.tool_config.get("stats", {})
            file_path = stats_config.get("storage_path", "data/tool_stats.json")
            
        self.tool_registry.load_stats(file_path) 
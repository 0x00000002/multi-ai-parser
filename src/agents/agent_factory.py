"""
Agent Factory Module

This module provides a factory for creating agent instances in the Agentic-AI framework.
"""
from typing import Dict, Any, Optional, Type, List
from .interfaces import AgentInterface
from .agent_registry import AgentRegistry
from .base_agent import BaseAgent
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig
from .interfaces import AgentFactoryInterface
from .tool_finder_agent import ToolFinderAgent
from ..tools.tool_registry import ToolRegistry


class AgentFactory(AgentFactoryInterface):
    """
    Factory for creating agent instances in the Agentic-AI framework.
    
    This class creates instances of agents based on their type and configuration.
    It uses the AgentRegistry to look up agent classes and creates instances
    with appropriate parameters.
    """
    
    def __init__(self, 
                 registry: AgentRegistry,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the agent factory.
        
        Args:
            registry: An AgentRegistry instance
            unified_config: UnifiedConfig instance (optional)
            logger: Logger instance (optional)
        """
        self.registry = registry
        self.config = unified_config or UnifiedConfig.get_instance()
        self.logger = logger or LoggerFactory.create(name="agent_factory")
        
        self.logger.info("Agent factory initialized")
    
    def create(self, agent_type: str, **kwargs) -> Optional[BaseAgent]:
        """
        Create an agent instance of the specified type.
        
        Args:
            agent_type: The type of agent to create
            **kwargs: Additional parameters to pass to the agent constructor
            
        Returns:
            An instance of BaseAgent or None if creation fails
        """
        try:
            # Look up the agent class
            agent_class = self.registry.get_agent_class(agent_type)
            
            if not agent_class:
                self.logger.error(f"Agent type not found: {agent_type}")
                return None
            
            # Get model from kwargs or use default
            model = kwargs.pop('model', None)
            
            # Initialize AI instance if needed
            ai_instance = kwargs.get('ai_instance')
            if not ai_instance and agent_type != "orchestrator":
                # Only create AI instance if not provided and not orchestrator
                from ..core.tool_enabled_ai import AI
                
                # Get the appropriate model and system prompt
                agent_config = self.config.get_agent_config(agent_type) or {}
                default_model = agent_config.get("default_model")
                system_prompt = kwargs.get('system_prompt') or agent_config.get("system_prompt")
                
                # Use the provided model if available, otherwise use default
                selected_model = model or default_model
                
                ai_instance = AI(
                    model=selected_model,
                    system_prompt=system_prompt,
                    logger=self.logger
                )
                
                kwargs['ai_instance'] = ai_instance
            
            # Create the agent instance
            # Remove agent_id from kwargs if it matches the agent_type for orchestrator
            # to avoid duplicate agent_id parameter
            agent_creation_kwargs = kwargs.copy()
            if agent_type == "orchestrator" and 'agent_id' not in agent_creation_kwargs:
                # Orchestrator sets its own agent_id in __init__, don't pass it
                agent_instance = agent_class(
                    unified_config=self.config,
                    logger=self.logger,
                    **agent_creation_kwargs
                )
            else:
                # For other agents, pass the agent_type as agent_id
                agent_instance = agent_class(
                    unified_config=self.config,
                    logger=self.logger,
                    agent_id=agent_type,
                    **agent_creation_kwargs
                )
            
            self.logger.info(f"Created agent: {agent_type}")
            return agent_instance
            
        except Exception as e:
            self.logger.error(f"Error creating agent {agent_type}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_orchestrator(self, tool_registry: Optional[ToolRegistry] = None, **kwargs) -> Optional[BaseAgent]:
        """
        Create an orchestrator agent instance.
        
        Args:
            tool_registry: Tool registry instance (optional)
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            An Orchestrator agent instance or None if creation fails
        """
        try:
            # Create necessary subcomponents
            from .request_analyzer import RequestAnalyzer
            from .response_aggregator import ResponseAggregator
            from ..core.model_selector import ModelSelector
            
            # Create RequestAnalyzer
            request_analyzer = kwargs.get('request_analyzer')
            if not request_analyzer:
                request_analyzer = RequestAnalyzer(
                    unified_config=self.config,
                    logger=self.logger
                )
                kwargs['request_analyzer'] = request_analyzer
            
            # Create ResponseAggregator
            response_aggregator = kwargs.get('response_aggregator')
            if not response_aggregator:
                response_aggregator = ResponseAggregator(
                    unified_config=self.config,
                    logger=self.logger
                )
                kwargs['response_aggregator'] = response_aggregator
            
            # Create ModelSelector
            model_selector = kwargs.get('model_selector')
            if not model_selector:
                model_selector = ModelSelector(
                    unified_config=self.config
                )
                kwargs['model_selector'] = model_selector
            
            # Create ToolFinderAgent if a tool registry is provided
            tool_finder_agent = kwargs.get('tool_finder_agent')
            if not tool_finder_agent and tool_registry:
                # Check if AI is provided for tool finder
                tool_finder_ai = kwargs.get('tool_finder_ai')
                if not tool_finder_ai:
                    from ..core.tool_enabled_ai import AI
                    tool_finder_ai = AI(
                        unified_config=self.config,
                        logger=self.logger
                    )
                
                # Create the tool finder agent
                tool_finder_agent = ToolFinderAgent(
                    ai_instance=tool_finder_ai,
                    tool_registry=tool_registry,
                    unified_config=self.config,
                    logger=self.logger
                )
                kwargs['tool_finder_agent'] = tool_finder_agent
            
            # Create the orchestrator with factory reference
            kwargs['agent_factory'] = self
            
            return self.create("orchestrator", **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error creating orchestrator: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_tool_finder(self, tool_registry: ToolRegistry, **kwargs) -> Optional[ToolFinderAgent]:
        """
        Create a tool finder agent instance.
        
        Args:
            tool_registry: Tool registry instance
            **kwargs: Additional parameters to pass to the constructor
            
        Returns:
            A ToolFinderAgent instance or None if creation fails
        """
        try:
            # Add tool registry to kwargs
            kwargs['tool_registry'] = tool_registry
            
            # Check if AI is provided
            ai_instance = kwargs.get('ai_instance')
            if not ai_instance:
                from ..core.tool_enabled_ai import AI
                
                # Get configuration values
                agent_config = self.config.get_agent_config("tool_finder") or {}
                default_model = agent_config.get("default_model")
                system_prompt = kwargs.get('system_prompt') or agent_config.get("system_prompt")
                
                # Use model from kwargs if provided
                model = kwargs.pop('model', default_model)
                
                ai_instance = AI(
                    model=model,
                    system_prompt=system_prompt,
                    logger=self.logger
                )
                
                kwargs['ai_instance'] = ai_instance
            
            # Create the agent
            return self.create("tool_finder", **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error creating tool finder agent: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
"""
Agent Registry Module

This module provides a registry for managing agents in the Agentic-AI framework.
"""
from typing import Dict, Type, Optional, Any, List

from src.agents.interfaces import AgentInterface, AgentRegistryInterface
from src.utils.logger import LoggerFactory


class AgentRegistry(AgentRegistryInterface):
    """
    Registry for managing agents in the Agentic-AI framework.
    
    This class maintains a registry of available agent types and provides
    methods for registering and retrieving agent classes.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the agent registry.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or LoggerFactory.create("agent_registry")
        self._agents: Dict[str, Type[AgentInterface]] = {}
        
        # Register agents using the agent_registrar
        self._register_agents()
        
        self.logger.info("Agent registry initialized")
    
    def _register_agents(self):
        """
        Register agents using the agent_registrar to avoid circular imports.
        """
        from src.agents.agent_registrar import register_core_agents, register_extension_agents
        
        # Register core agents
        register_core_agents(self, self.logger)
        
        # Register any extension agents
        register_extension_agents(self, self.logger)
    
    def register(self, agent_type: str, agent_class: Type[AgentInterface]) -> None:
        """
        Register an agent class.
        
        Args:
            agent_type: Type identifier for the agent
            agent_class: Agent class to register
        """
        if not issubclass(agent_class, AgentInterface):
            self.logger.warning(f"Agent class {agent_class.__name__} does not implement AgentInterface")
            return
            
        self._agents[agent_type] = agent_class
            
        self.logger.info(f"Registered agent type: {agent_type}")
    
    def get_agent_class(self, agent_type: str) -> Optional[Type[AgentInterface]]:
        """
        Get an agent class by type.
        
        Args:
            agent_type: Type identifier for the agent
            
        Returns:
            Agent class if found, None otherwise
        """
        return self._agents.get(agent_type)
    
    def get_agent_types(self) -> List[str]:
        """
        Get all registered agent types.
        
        Returns:
            List of agent type identifiers
        """
        return list(self._agents.keys())
    
    def has_agent_type(self, agent_type: str) -> bool:
        """
        Check if an agent type is registered.
        
        Args:
            agent_type: Type identifier for the agent
            
        Returns:
            True if the agent type is registered, False otherwise
        """
        return agent_type in self._agents
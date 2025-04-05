"""
Interfaces for agent implementations in the multi-agent architecture.
"""
from typing import Dict, List, Any, Optional, Union, Protocol, TypeVar
from typing_extensions import runtime_checkable
from enum import Enum


class AgentResponseStatus(str, Enum):
    """Status codes for agent responses."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    CANCELLED = "cancelled"


class AgentResponse:
    """Standard response format for agent operations."""
    
    def __init__(self, 
                 content: Any,
                 status: AgentResponseStatus = AgentResponseStatus.SUCCESS,
                 error: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent response.
        
        Args:
            content: Response content
            status: Response status
            error: Optional error message
            metadata: Optional metadata dictionary
        """
        self.content = content
        self.status = status
        self.error = error
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata
        }


# Type variable for request and response
T = TypeVar('T')

@runtime_checkable
class AgentInterface(Protocol):
    """Interface for agent implementation."""
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request and return a response.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        ...
    
    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            request: The request object
            
        Returns:
            Confidence score (0.0-1.0) indicating ability to handle
        """
        ...

@runtime_checkable
class AsyncAgentInterface(Protocol):
    """Interface for asynchronous agent implementation."""
    
    async def async_process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request asynchronously and return a response.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        ...
    
    async def async_can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine asynchronously if this agent can handle the request.
        
        Args:
            request: The request object
            
        Returns:
            Confidence score (0.0-1.0) indicating ability to handle
        """
        ...

@runtime_checkable
class AgentRegistryInterface(Protocol):
    """Interface for agent registry."""
    
    def register(self, name: str, agent_class: Any) -> None:
        """
        Register an agent class.
        
        Args:
            name: Name to register the agent under
            agent_class: The agent class to register
        """
        ...
    
    def get_agent_class(self, name: str) -> Optional[Any]:
        """
        Get an agent class by name.
        
        Args:
            name: Name of the agent class to retrieve
            
        Returns:
            The agent class if found, None otherwise
        """
        ...
    
    def get_all_agents(self) -> List[str]:
        """
        Get names of all registered agents.
        
        Returns:
            List of registered agent names
        """
        ...

@runtime_checkable
class AgentFactoryInterface(Protocol):
    """Interface for agent factory."""
    
    def create(self, agent_type: str, **kwargs) -> AgentInterface:
        """
        Create an agent instance.
        
        Args:
            agent_type: Type of agent to create
            kwargs: Additional arguments for agent initialization
            
        Returns:
            Initialized agent instance
        """
        ...
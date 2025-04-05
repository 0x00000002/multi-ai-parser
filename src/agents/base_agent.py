"""
Base agent implementation for the multi-agent architecture.
"""
from typing import Dict, Any, Optional, TYPE_CHECKING
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig

if TYPE_CHECKING:
    from ..core.base_ai import AIBase
    from ..tools.tool_manager import ToolManager


class BaseAgent:
    """Base implementation for all agents."""
    
    def __init__(self, 
                 ai_instance: Optional['AIBase'] = None,
                 tool_manager: Optional['ToolManager'] = None,
                 unified_config: Optional[UnifiedConfig] = None,
                 logger: Optional[LoggerInterface] = None,
                 agent_id: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            ai_instance: AI instance for processing
            tool_manager: Tool manager for handling tools
            unified_config: UnifiedConfig instance
            logger: Logger instance
            agent_id: Agent identifier
        """
        self.ai_instance = ai_instance
        self.tool_manager = tool_manager
        self.config = unified_config or UnifiedConfig.get_instance()
        self.logger = logger or LoggerFactory.create(name=f"agent.{agent_id}")
        self.agent_id = agent_id or "base_agent"
        
        # Get agent-specific configuration
        self.agent_config = self.config.get_agent_config(self.agent_id) or {}
        
        self.logger.info(f"Initialized {self.agent_id} agent")
        
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request with this agent.
        
        Args:
            request: The request object containing prompt and metadata
            
        Returns:
            Response object with content and metadata
        """
        self.logger.info(f"Processing request with {self.agent_id} agent")
        
        # Check for model override in the request
        model_override = request.get("model")
        system_prompt_override = request.get("system_prompt")
        
        # Apply model override if specified
        if model_override and self.ai_instance:
            original_model = self.ai_instance.get_model_info().get("model_id")
            if model_override != original_model:
                self.logger.info(f"Overriding model: {original_model} -> {model_override}")
                # Create a new AI instance with the overridden model
                self.ai_instance = self.ai_instance.__class__(
                    model=model_override,
                    logger=self.ai_instance._logger
                )
                
        # Apply system prompt override if specified
        if system_prompt_override and self.ai_instance:
            self.logger.info("Using system prompt from request")
            self.ai_instance.set_system_prompt(system_prompt_override)
            
        # Extract prompt from request
        if isinstance(request, dict) and "prompt" in request:
            prompt = request["prompt"]
        else:
            prompt = str(request)
            
        # Process with AI instance
        if self.ai_instance:
            try:
                response = self.ai_instance.request(prompt)
                
                # Restore original model after processing if we changed it
                if model_override and original_model:
                    self.ai_instance = self.ai_instance.__class__(
                        model=original_model,
                        logger=self.ai_instance._logger
                    )
                    
                return {
                    "content": response,
                    "agent_id": self.agent_id,
                    "status": "success"
                }
            except Exception as e:
                self.logger.error(f"Error processing request: {str(e)}")
                return {
                    "content": f"Error: {str(e)}",
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": str(e)
                }
        else:
            self.logger.warning("No AI instance available for processing")
            return {
                "content": "Error: No AI instance available for processing",
                "agent_id": self.agent_id,
                "status": "error"
            }
    
    def can_handle(self, request: Dict[str, Any]) -> float:
        """
        Determine if this agent can handle the request.
        
        Args:
            request: The request object
            
        Returns:
            Confidence score (0.0-1.0) indicating ability to handle
        """
        # Base implementation returns low confidence
        # Specialized agents should override this
        return 0.1
    
    def _enrich_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a response with additional metadata.
        
        Args:
            response: The response to enrich
            
        Returns:
            Enriched response with metadata
        """
        # Add agent metadata if not present
        if "agent_id" not in response:
            response["agent_id"] = self.agent_id
        
        # Add status if not present
        if "status" not in response:
            response["status"] = "success"
        
        return response
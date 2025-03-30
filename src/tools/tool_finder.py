"""
Tool finder for identifying relevant tools for a given prompt.
"""
from typing import List, Set, Optional, Dict, Any
from .interfaces import ToolStrategy
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.config_manager import ConfigManager
from ..exceptions import AIToolError


class ToolFinder:
    """Finder for identifying relevant tools for a given prompt."""
    
    def __init__(self,
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the tool finder.
        
        Args:
            model_id: The model ID to use for tool finding
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self._model_id = model_id
        self._config_manager = config_manager
        self._logger = logger or LoggerFactory.create()
        self._available_tools = {}  # Will be populated by ToolManager
    
    def set_available_tools(self, tools: Dict[str, ToolStrategy]) -> None:
        """
        Set the available tools for finding.
        
        Args:
            tools: Dictionary of available tools by name
        """
        self._available_tools = tools
    
    def find_tools(self, prompt: str, conversation_history: Optional[List[str]] = None) -> Set[str]:
        """
        Find relevant tools for a given prompt.
        
        Args:
            prompt: The user prompt
            conversation_history: Optional list of recent conversation messages
            
        Returns:
            Set of tool names that are relevant
            
        Raises:
            AIToolError: If tool finding fails
        """
        try:
            relevant_tools = set()
            
            # Get all tool descriptions
            tool_descriptions = {
                name: tool.get_description()
                for name, tool in self._available_tools.items()
            }
            
            # Simple keyword-based matching
            prompt_lower = prompt.lower()
            for name, description in tool_descriptions.items():
                # Check if any keywords from the description appear in the prompt
                keywords = set(description.lower().split())
                if any(keyword in prompt_lower for keyword in keywords):
                    relevant_tools.add(name)
            
            self._logger.debug(f"Found relevant tools: {relevant_tools}")
            return relevant_tools
            
        except Exception as e:
            self._logger.error(f"Tool finding failed: {str(e)}")
            raise AIToolError(f"Failed to find tools: {str(e)}") 
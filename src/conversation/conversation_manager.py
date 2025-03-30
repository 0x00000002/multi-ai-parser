"""
Manages conversation history and state.
"""
from typing import Dict, List, Any, Optional
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import ConversationError
from dataclasses import dataclass


@dataclass
class Message:
    """Represents a message in a conversation."""
    role: str
    content: str
    name: Optional[str] = None


class ConversationManager:
    """Manages conversation history and state."""
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the conversation manager.
        
        Args:
            logger: Logger instance for logging operations
        """
        self._logger = logger or LoggerFactory.create()
        self._messages: List[Message] = []
        self._metadata: Dict[str, Any] = {}
        self._context: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, name: Optional[str] = None, **kwargs) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The message content
            name: Optional name for the message (e.g., tool name)
            **kwargs: Additional message metadata
        """
        message = Message(role=role, content=content, name=name)
        self._messages.append(message)
        self._logger.debug(f"Added {role} message to conversation")
    
    def add_interaction(self, user_message: str, assistant_message: str) -> None:
        """
        Add a user-assistant interaction to the conversation.
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
        """
        self.add_message(role="user", content=user_message)
        self.add_message(role="assistant", content=assistant_message)
        self._logger.debug("Added user-assistant interaction to conversation")
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the conversation.
        
        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "name": msg.name
            }
            for msg in self._messages
            if msg.name is not None or msg.role != "tool"
        ]
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """
        Get the last message in the conversation.
        
        Returns:
            The last message dictionary or None if no messages
        """
        if not self._messages:
            return None
        last_msg = self._messages[-1]
        return {
            "role": last_msg.role,
            "content": last_msg.content,
            "name": last_msg.name
        }
    
    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        self._messages.clear()
        self._logger.debug("Cleared conversation messages")
    
    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set conversation metadata.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value
        self._logger.debug(f"Set conversation metadata: {key}")
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get conversation metadata.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self._metadata.get(key, default)
    
    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all conversation metadata.
        
        Returns:
            Dictionary of all metadata
        """
        return self._metadata.copy()
    
    def clear_metadata(self) -> None:
        """Clear all conversation metadata."""
        self._metadata.clear()
        self._logger.debug("Cleared conversation metadata")
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context value for the conversation.
        
        Args:
            key: Context key
            value: Context value
        """
        self._context[key] = value
        self._logger.debug(f"Set context {key}")
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context value from the conversation.
        
        Args:
            key: Context key
            default: Default value if key not found
            
        Returns:
            Context value or default
        """
        return self._context.get(key, default)
    
    def reset(self) -> None:
        """Reset the conversation state."""
        self.clear_messages()
        self.clear_metadata()
        self._context.clear()
        self._logger.debug("Reset conversation state") 
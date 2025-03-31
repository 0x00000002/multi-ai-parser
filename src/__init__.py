"""
AI Framework package.

Re-exports key components for easier access.
Example:
    from src import AI, ConfigManager, PromptManager
"""

# Core components
from .core import AI, AIBase, AsyncAI
# from .core import ModelSelector # Removed - Causes ImportError
from .core.interfaces import AIInterface, ProviderInterface, AsyncAIInterface, AsyncProviderInterface

# Configuration
from .config import ConfigManager
# Assuming Model enum will be fixed in src/config/models.py
# from .config.models import Model
from .config.config_manager import Quality, Speed, Privacy

# Conversation management
from .conversation import ConversationManager, Message

# Prompt management
from .prompts import PromptManager, PromptTemplate

# Tool management
from .tools import ToolManager
from .tools.models import ToolDefinition, ToolResult, ToolCall

# Utilities
from .utils import LoggerFactory, LoggerInterface

# Exceptions
from .exceptions import AIError, AIConfigError, AIProviderError, AIToolError, ConversationError

__all__ = [
    # Core
    "AI",
    "AIBase",
    "AsyncAI",
    # "ModelSelector", # Removed - Causes ImportError
    "AIInterface",
    "ProviderInterface",
    "AsyncAIInterface",
    "AsyncProviderInterface",

    # Config
    "ConfigManager",
    # "Model", # Uncomment when src/config/models.py is fixed
    "UseCase",
    "Quality",
    "Speed",
    "Privacy",

    # Conversation
    "ConversationManager",
    "Message",

    # Prompts
    "PromptManager",
    "PromptTemplate",

    # Tools
    "ToolManager",
    "ToolDefinition",
    "ToolResult",
    "ToolCall",

    # Utils
    "LoggerFactory",
    "LoggerInterface",

    # Exceptions
    "AIError",
    "AIConfigError",
    "AIProviderError",
    "AIToolError",
    "ConversationError",
]

# The following were commented out in the original file, keeping for reference if needed
# from src.ai.AI import AI
# from src.ModelSelector import ModelSelector, UseCase, Quality, Speed
# from src.website import Website

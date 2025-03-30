"""
Model definitions for the AI system.
"""
from enum import Enum


class Model(str, Enum):
    """Available AI models in the system."""
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    O3_MINI = "o3-mini"
    GPT_4O_MINI = "gpt-4o-mini"
    GEMMA_3 = "gemma3"
    DEEPSEEK_32B = "deepseek-32b" 
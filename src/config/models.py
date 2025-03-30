"""
Models module for the AI framework.
This file is dynamically updated when the configuration is loaded.
"""
from enum import Enum

class Model(str, Enum):
    """Available AI models loaded from configuration."""
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    O3_MINI = "o3-mini"
    GPT_4O_MINI = "gpt-4o-mini"
    GEMMA3 = "gemma3"
    DEEPSEEK_32B = "deepseek-32b"
    QWQ = "qwq"
    DEEPSEEK_7B = "deepseek-7b"
    GEMINI_2_5_PRO = "gemini-2-5-pro"
    GEMINI_1_5_PRO = "gemini-1-5-pro"

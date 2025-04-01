"""
Models module for the AI framework.
This file should be manually updated to reflect the models defined in the configuration.
"""
from enum import Enum

class Model(str, Enum):
    """Available AI models loaded from configuration."""
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3-5-haiku"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    O3_MINI = "o3-mini"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GEMINI_2_5_PRO = "gemini-2-5-pro"
    GEMINI_1_5_PRO = "gemini-1-5-pro"
    GEMMA3_27B = "gemma3-27b"
    GEMMA3_12B = "gemma3-12b"
    DEEPSEEK_32B = "deepseek-32b"
    DEEPSEEK_7B = "deepseek-7b"
    QWQ = "qwq"
    PHI4 = "phi4"
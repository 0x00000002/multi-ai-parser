from typing import Dict, List, ClassVar  
from src.ai.ai_config import Provider

from src.ai.tools.tools_types import OpenAI_Function, Anthropic_Function, Google_Function, Ollama_Function
from src.ai.tools.tools_list import Tool

class ToolsRegistry:
    TOOLS: ClassVar[Dict[Provider, type]] = {
        Provider.OPENAI: OpenAI_Function,
        Provider.ANTHROPIC: Anthropic_Function,
        Provider.GOOGLE: Google_Function,
        Provider.OLLAMA: Ollama_Function
    }

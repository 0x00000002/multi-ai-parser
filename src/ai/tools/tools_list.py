from enum import Enum
from src.ai.tools.tool_name import tool_name, function_name 

class Tool(Enum):
    SOME_TOOL = (tool_name, function_name)
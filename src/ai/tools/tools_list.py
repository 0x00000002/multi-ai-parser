from enum import Enum
from src.ai.tools.tiicket_price import ticket_price_tool, get_ticket_price
from typing import Optional, Tuple, Callable, Dict, List
from src.ai.tools.models import Function

class Tool(Enum):
    TICKET_ORACLE = (ticket_price_tool, get_ticket_price)

    @classmethod
    def get_tools(cls) -> List[Dict[str, str]]:
        return [tool.value[0].dict() for tool in cls]

    @classmethod
    def get_tools_descriptions(cls) -> Dict[str, str]:
        tools_dict = {}
        for tool in Tool:
            tools_dict[tool.name] = tool.value[0].description     
        return tools_dict
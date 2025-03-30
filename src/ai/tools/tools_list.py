from enum import Enum
from src.ai.tools.tiicket_price import ticket_price_tool, get_ticket_price
from typing import Dict, List

class Tool(Enum):
    TICKET_ORACLE = (ticket_price_tool, get_ticket_price)

    @classmethod
    def list_tools(cls) -> Dict[str, str]:
        tools_dict = {}
        for tool in Tool:
            tools_dict[tool.name] = tool.value[0].description     
        return tools_dict
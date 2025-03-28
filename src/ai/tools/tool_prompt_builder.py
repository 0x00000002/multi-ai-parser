from typing import Dict, List
from src.ai.tools.tools_list import Tool

class ToolPromptBuilder:
    """Handles building prompts that include tool information."""
    
    @staticmethod
    def build_tools_section(tools: List[Tool]) -> str:
        """
        Build the tools section of the prompt.
        
        Args:
            tools: List of available tools
            
        Returns:
            Formatted string containing tool information and usage instructions
        """
        tools_info = {}
        for tool in tools:
            tools_info[tool.value[0].name] = tool.value[0].description
        
        tools_section = "\n\nAvailable tools for this request:\n"
        tools_section += "\n".join([f"- {name}: {desc}" for name, desc in tools_info.items()])
        tools_section += "\n\nIMPORTANT: You MUST use the available tools to answer the user's question. Do not ask for additional information."
        tools_section += "\n\nTo use a tool, format your response like this:\n"
        tools_section += "TOOL_CALL: {\"name\": \"tool_name\", \"arguments\": {\"param1\": \"value1\"}}\n"
        tools_section += "Your response text here\n\n"
        tools_section += "Example:\n"
        tools_section += "TOOL_CALL: {\"name\": \"get_ticket_price\", \"arguments\": {\"destination_city\": \"New York\"}}\n"
        tools_section += "\nCRITICAL: When you receive a tool's response, ONLY use that exact response. Do not add any additional information, explanations, or make up prices/values. Just return the tool's response as-is."
        
        return tools_section
    
    @staticmethod
    def build_enhanced_prompt(user_prompt: str, tools: List[Tool]) -> str:
        """
        Build an enhanced prompt that includes tool information.
        
        Args:
            user_prompt: The original user prompt
            tools: List of available tools
            
        Returns:
            Enhanced prompt with tool information
        """
        tools_section = ToolPromptBuilder.build_tools_section(tools)
        return f"{user_prompt}\n{tools_section}" 
"""
Tool prompt builder for enhancing prompts with tool information.
"""
from typing import List, Set
from .interfaces import ToolStrategy


class ToolPromptBuilder:
    """Builder for enhancing prompts with tool information."""
    
    @staticmethod
    def build_enhanced_prompt(prompt: str, tools: List[ToolStrategy]) -> str:
        """
        Build an enhanced prompt with tool information.
        
        Args:
            prompt: The original prompt
            tools: List of tools to include
            
        Returns:
            Enhanced prompt with tool information
        """
        if not tools:
            return prompt
            
        tool_descriptions = []
        for tool in tools:
            description = tool.get_description()
            schema = tool.get_schema()
            tool_descriptions.append(f"- {tool.get_name()}: {description}")
            tool_descriptions.append(f"  Parameters: {schema}")
        
        enhanced_prompt = f"{prompt}\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
        return enhanced_prompt 
"""
Tool prompt builder for enhancing prompts with tool information.
"""
from typing import List, Set, Tuple
from .interfaces import ToolStrategy


class ToolPromptBuilder:
    """Builder for enhancing prompts with tool information."""
    
    @staticmethod
    def build_enhanced_prompt(prompt: str, tools_with_names: List[Tuple[str, ToolStrategy]]) -> str:
        """
        Build an enhanced prompt with tool information.
        
        Args:
            prompt: The original prompt
            tools_with_names: List of tuples, each containing (tool_name, tool_strategy)
            
        Returns:
            Enhanced prompt with tool information
        """
        if not tools_with_names:
            return prompt
            
        tool_descriptions = []
        for tool_name, tool in tools_with_names:
            description = tool.get_description()
            schema = tool.get_schema()
            tool_descriptions.append(f"- {tool_name}: {description}")
            tool_descriptions.append(f"  Parameters: {schema}")
        
        enhanced_prompt = f"{prompt}\n\nAvailable tools:\n" + "\n".join(tool_descriptions)
        return enhanced_prompt 
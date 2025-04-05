"""
Tool Prompt Builder.
Enhances prompts with tool information.
"""
from typing import List, Dict, Any, Optional
from ..prompts.prompt_template import PromptTemplate
from ..utils.logger import LoggerInterface, LoggerFactory


class ToolPromptBuilder:
    """
    Enhances prompts with information about available tools.
    """
    
    def __init__(self, logger: Optional[LoggerInterface] = None, prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the tool prompt builder.
        
        Args:
            logger: Logger instance
            prompt_template: PromptTemplate service for generating prompts
        """
        self._logger = logger or LoggerFactory.create(name="tool_prompt_builder")
        self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
    
    def enhance_prompt_with_tools(self, prompt: str, tool_descriptions: List[str]) -> str:
        """
        Enhance a prompt with information about available tools.
        
        Args:
            prompt: The original prompt
            tool_descriptions: List of tool descriptions
            
        Returns:
            Enhanced prompt with tool information
        """
        if not tool_descriptions:
            return prompt
            
        try:
            # Prepare template variables
            variables = {
                "prompt": prompt,
                "tool_descriptions": "\n".join(tool_descriptions)
            }
            
            # Use template to generate enhanced prompt
            try:
                enhanced_prompt, usage_id = self._prompt_template.render_prompt(
                    template_id="tool_enhancement",
                    variables=variables
                )
                
                # Record metrics
                self._prompt_template.record_prompt_performance(
                    usage_id=usage_id,
                    metrics={
                        "num_tools": len(tool_descriptions),
                        "original_prompt_length": len(prompt),
                        "enhanced_prompt_length": len(enhanced_prompt)
                    }
                )
                
                return enhanced_prompt
                
            except ValueError:
                # Fallback if template not found
                self._logger.warning("Template 'tool_enhancement' not found, using fallback")
                return self._fallback_enhance_prompt(prompt, tool_descriptions)
        except Exception as e:
            self._logger.error(f"Error enhancing prompt: {str(e)}")
            return self._fallback_enhance_prompt(prompt, tool_descriptions)
    
    def _fallback_enhance_prompt(self, prompt: str, tool_descriptions: List[str]) -> str:
        """
        Fallback method to enhance a prompt without using templates.
        
        Args:
            prompt: The original prompt
            tool_descriptions: List of tool descriptions
            
        Returns:
            Enhanced prompt with tool information
        """
        return f"{prompt}\n\nAvailable tools:\n" + "\n".join(tool_descriptions) 
"""
Handles parsing of AI responses, including extraction of thoughts and content.
"""
import re
from typing import Optional, Dict, Any
from ..exceptions import AIResponseError
from ..utils.logger import LoggerInterface


class ResponseParser:
    """Handles parsing of AI responses and thoughts extraction."""
    
    def __init__(self, logger: Optional[LoggerInterface] = None):
        """
        Initialize the response parser.
        
        Args:
            logger: Logger instance for logging operations
        """
        self._logger = logger
    
    def parse_response(self, 
                      response: str, 
                      extract_thoughts: bool = True,
                      show_thinking: bool = False,
                      thought_start: Optional[str] = None,
                      thought_end: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse an AI response, optionally extracting thoughts.
        
        Args:
            response: The raw AI response
            extract_thoughts: Whether to extract thoughts from the response
            show_thinking: Whether to include thoughts in the response content
            thought_start: Custom start pattern for thoughts
            thought_end: Custom end pattern for thoughts
            
        Returns:
            Dictionary containing parsed response with optional thoughts
        """
        try:
            # Define the thought patterns
            start_tag = thought_start or "<think>"
            end_tag = thought_end or "</think>"
            
            # Create the regex pattern for the full thought block (including tags)
            thought_block_pattern = f"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
            
            # Initialize result
            result = {}
            
            # Search for thought blocks
            thought_match = re.search(thought_block_pattern, response, re.DOTALL)
            
            if thought_match and extract_thoughts:
                # Extract the thought content (without tags)
                thoughts = thought_match.group(1).strip()
                result["thoughts"] = thoughts
                
                # Remove the entire thought block (including tags) from the response
                clean_content = re.sub(thought_block_pattern, "", response, flags=re.DOTALL).strip()
                
                # If clean_content is empty, something went wrong, use the whole response
                if not clean_content:
                    # Try to get content after the last thought tag
                    content_after_thoughts = response.split(end_tag)[-1].strip()
                    clean_content = content_after_thoughts if content_after_thoughts else response
                
                # Decide final content based on show_thinking flag
                if show_thinking:
                    result["content"] = response  # Keep original with thoughts
                else:
                    result["content"] = clean_content  # Use clean version without thoughts
            else:
                # No thoughts found or extraction disabled
                result["content"] = response
                
            return result
            
        except Exception as e:
            if self._logger:
                self._logger.error(f"Error parsing response: {str(e)}")
            raise AIResponseError(f"Failed to parse AI response: {str(e)}")

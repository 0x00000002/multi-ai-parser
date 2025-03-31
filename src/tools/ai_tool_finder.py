"""
AI-powered tool finder that uses an LLM to select relevant tools.
"""
import json
from typing import List, Optional, Set, Dict, Any
from ..utils.logger import LoggerInterface
from ..config.config_manager import ConfigManager
from ..config.config_manager import ModelConfig  # Import ModelConfig
from ..exceptions import AIToolError
from .models import ToolDefinition
from ..exceptions import AISetupError, AIProcessingError


class AIToolFinder:
    """
    Uses an AI model to find relevant tools based on user prompts and tool descriptions.
    """

    DEFAULT_SYSTEM_PROMPT = """
You are an expert assistant responsible for selecting the most relevant tools to fulfill a user's request.
Based on the user's request and the provided list of available tools (with their descriptions), identify which tools are necessary.

Respond ONLY with a JSON array containing the names of the selected tools.
The format MUST be:
{
    "tools": ["TOOL_NAME_1", "TOOL_NAME_2", ...]
}

If no tools are relevant or necessary for the user's request, return an empty array:
{
    "tools": []
}

Do not include any explanations, commentary, or text outside the JSON structure in your response.
"""

    def __init__(self,
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: LoggerInterface):
        """
        Initialize the AIToolFinder.

        Args:
            model_id: The ID of the AI model to use for tool finding.
            config_manager: Configuration manager instance.
            logger: Logger instance.
        """
        self._logger = logger
        self._config_manager = config_manager
        self._model_id = model_id
        self._available_tools: Dict[str, ToolDefinition] = {}

        try:
            # Get model configuration
            model_config = self._config_manager.get_model_config(self._model_id)
            if not model_config:
                 raise AIToolError(f"Model configuration not found for ID: {self._model_id}")

            self._provider_name = model_config.provider # Store provider name
            
            # Move the import inside the method to avoid circular imports
            from ..core.provider_factory import ProviderFactory
            
            # Initialize the AI provider for tool finding
            self._finder_ai = ProviderFactory.create(
                model_id=self._model_id,
                config_manager=self._config_manager,
                logger=self._logger
            )
            self._logger.info(f"AIToolFinder initialized with model: {self._model_id}")

        except Exception as e:
            self._logger.error(f"Failed to initialize AIToolFinder: {str(e)}")
            raise AIToolError(f"AIToolFinder initialization failed: {str(e)}") from e

    def set_available_tools(self, tools: Dict[str, ToolDefinition]) -> None:
        """
        Set the list of tools available for selection.

        Args:
            tools: A dictionary mapping tool names to their ToolDefinition.
        """
        self._available_tools = tools
        self._logger.debug(f"AIToolFinder received {len(tools)} available tools.")

    def _format_tools_for_prompt(self) -> str:
        """Formats the available tools into a string list for the prompt."""
        if not self._available_tools:
            return "No tools available."

        tool_list_str = "AVAILABLE TOOLS:\n"
        for name, definition in self._available_tools.items():
            tool_list_str += f"- {name}: {definition.description}\n"

        return tool_list_str.strip()

    def find_tools(self, user_prompt: str, conversation_history: Optional[List[str]] = None) -> Set[str]:
        """
        Analyzes the user's prompt and returns a set of relevant tool names.

        Args:
            user_prompt: The user's request.
            conversation_history: Optional list of recent conversation messages.

        Returns:
            A set containing the names of the selected tools. Returns an empty set on failure.
        
        Raises:
            AIToolError: If the underlying AI call fails.
        """
        if not self._available_tools:
            self._logger.warning("AIToolFinder called but no tools are available.")
            return set()

        # Format tools for the prompt
        available_tools_str = self._format_tools_for_prompt()

        # Construct the prompt for the tool finding AI
        finder_prompt_parts = []
        if conversation_history:
            history_str = "\n".join(conversation_history)
            finder_prompt_parts.append(f"CONVERSATION HISTORY:\n{history_str}\n")

        # Add user request, ensure quotes within the prompt are handled
        # Using simple concatenation to avoid f-string complexity with quotes
        finder_prompt_parts.append('USER REQUEST: "' + user_prompt.replace('"', '\\"') + '"\n') 

        # Add available tools section
        finder_prompt_parts.append(available_tools_str)
        
        finder_prompt_content = "\n".join(finder_prompt_parts)

        self._logger.debug(f"AIToolFinder prompt content:\n------\n{finder_prompt_content}\n------")

        try:
            # Use the request method of the initialized provider
            # The provider should handle message formatting internally based on its type
            response_content = self._finder_ai.request(finder_prompt_content)

            if not response_content:
                self._logger.warning("AIToolFinder received no response content from the AI.")
                return set()

            # Parse the JSON response
            try:
                # Attempt to clean potential markdown code fences
                cleaned_response = response_content.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:] # Remove ```json
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3] # Remove trailing ```
                elif cleaned_response.startswith("```"):
                     cleaned_response = cleaned_response[3:] # Remove ```
                     if cleaned_response.endswith("```"):
                         cleaned_response = cleaned_response[:-3] # Remove trailing ```
                
                # Ensure cleaned response is stripped of extra whitespace before parsing
                cleaned_response = cleaned_response.strip()

                # Check if the cleaned response is empty before attempting to parse
                if not cleaned_response:
                    self._logger.warning("AIToolFinder response content is empty after cleaning markdown fences.")
                    return set()

                result = json.loads(cleaned_response)
                selected_tool_names = result.get("tools", [])

                if not isinstance(selected_tool_names, list):
                     self._logger.warning(f"AIToolFinder received invalid format for tools list: {selected_tool_names}")
                     return set()
                
                # Validate tool names against available tools
                valid_tools = set()
                for name in selected_tool_names:
                    if name in self._available_tools:
                        valid_tools.add(name)
                    else:
                         self._logger.warning(f"AIToolFinder identified an unknown tool: '{name}'")

                self._logger.info(f"AIToolFinder selected tools: {valid_tools if valid_tools else 'None'}")
                return valid_tools

            except json.JSONDecodeError as json_err:
                self._logger.error(f"AIToolFinder failed to parse JSON response: {json_err}")
                self._logger.debug(f"Raw response content received: >>>{response_content}<<<")
                return set() # Return empty set if JSON parsing fails
            except Exception as parse_err:
                 self._logger.error(f"AIToolFinder error processing response: {parse_err}")
                 return set()


        except Exception as e:
            self._logger.error(f"AIToolFinder request failed: {str(e)}")
            # Propagate the error for ToolManager to potentially handle
            raise AIToolError(f"AIToolFinder failed to get response from AI: {str(e)}") from e

        return set() # Should not be reached, but safety return 
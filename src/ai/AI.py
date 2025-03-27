from typing import List, Dict, Any, Optional, Union
import src.ai.AIConfig as config
from src.ai.ModelSelector import ModelSelector, UseCase
from src.Parser import Parser
from src.ai.tools.tools_registry import ToolsRegistry
from enum import Enum
from src.ai.Errors import AI_API_Key_Error, AI_Processing_Error, AI_Streaming_Error, AI_Setup_Error
from src.Logger import Logger, NullLogger
# Import all provider classes to make them available in globals()
from src.ai.modules.Anthropic import ClaudeAI
from src.ai.modules.OpenAI import ChatGPT
from src.ai.modules.Ollama import Ollama
from src.ai.modules.Google import Gemini


class Role(Enum):
    USER = "user"
    AI = "ai"

    def __str__(self):
        return self.value


class AI:
    def __init__(self, 
                 model_param: Union[config.Model, Dict[str, Any], UseCase], 
                 quality: Optional[config.Quality] = None,
                 speed: Optional[config.Speed] = None,
                 use_local: bool = False,
                 system_prompt: str = "",
                 logger: Optional[Logger] = None):
        """
        Initialize the AI with flexible model selection.
        
        Args:
            model_param: Can be one of:
                - A specific Model instance
                - A dictionary with model parameters
                - A UseCase enum value (with optional quality/speed params)
            quality: Quality level from AIConfig.Quality
            speed: Speed preference from AIConfig.Speed
            use_local: Whether to use local models
            system_prompt: System prompt to use
            logger: Logger instance to use (defaults to NullLogger)
        """
        self._system_prompt = system_prompt
        self.thoughts = []
        self.questions = []
        self.responses = []
        self._logger = logger if logger is not None else NullLogger()
        
        # Determine the model based on input params
        if isinstance(model_param, config.Model):
            self._logger.debug(f"Direct Model instance provided: {model_param.name}")
            # Direct Model instance provided
            self._model = model_param
        elif isinstance(model_param, dict):
            # Dictionary of criteria provided
            self._logger.debug(f"Dictionary of criteria provided: {model_param}")
            params = model_param.copy()
            
            self._model = config.find_model(
                params.get('privacy', config.Privacy.EXTERNAL),
                params.get('quality', config.Quality.MEDIUM),
                params.get('speed', config.Speed.STANDARD)
            )
        elif isinstance(model_param, UseCase):
            # UseCase enum provided, use ModelSelector to get params
            self._logger.debug(f"UseCase enum provided: {model_param}")
            self._model = ModelSelector.get_model_params(
                use_case=model_param,
                quality=quality or None,
                speed=speed or None,
                use_local=use_local
            )
            
            # If system_prompt wasn't specified, get the default for this use case
            if not system_prompt:
                self._system_prompt = ModelSelector.get_system_prompt(model_param)
                self._logger.debug(f"System prompt: {self._system_prompt}")
        else:
            self._logger.error("model_param must be a Model, dictionary, or UseCase enum")
            raise AI_Setup_Error("model_param must be a Model, dictionary, or UseCase enum")
            
        self._logger.debug(f"Provider class name: {self._model.provider_class_name}")
        self._logger.debug(f"System prompt: {self._system_prompt}")
        self._logger.debug(f"Model: {self._model.name} ({self._model.model_id})")
        self._provider_class_name = self._model.provider_class_name
        
        # Get the appropriate provider class from globals using the class name
        provider_class = globals().get(self._provider_class_name)
        if not provider_class:
            raise AI_Setup_Error(f"Provider class not found: {self._provider_class_name}")
            
        # Log the selected model
        self._logger.debug(f"Selected model: {self._model.name} ({self._model.model_id})")
        self._logger.debug(f"Model properties: Privacy={self._model.privacy.name}, " 
            f"Quality={self._model.quality.name}, Speed={self._model.speed.name}")

        # Initialize the provider with the model and system prompt
        self.ai = provider_class(self._model, self._system_prompt, self._logger)
        
    @classmethod
    def for_use_case(cls, use_case: UseCase, quality: config.Quality = config.Quality.MEDIUM, 
                     speed: config.Speed = config.Speed.STANDARD, use_local: bool = False, 
                     custom_system_prompt: Optional[str] = None, logger: Optional[Logger] = None):
        """
        Factory method to create an AI instance for a specific use case.
        
        Args:
            use_case: The specific use case
            quality: Quality level from AIConfig.Quality
            speed: Speed preference from AIConfig.Speed
            use_local: Whether to use local models
            custom_system_prompt: Optional custom system prompt (overrides default)
            logger: Logger instance to use
            
        Returns:
            An AI instance configured for the specified use case
        """
        system_prompt = custom_system_prompt or ModelSelector.get_system_prompt(use_case)
        
        return cls(
            model_param=use_case,
            quality=quality,
            speed=speed,
            use_local=use_local,
            system_prompt=system_prompt,
            logger=logger
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value: str):
        if not isinstance(value, str):
            self._logger.error("System prompt must be a string")
            raise AI_Setup_Error("System prompt must be a string")
        self._system_prompt = value
        self._logger.debug(f"System prompt set to: {value}")

    @property
    def model(self) -> config.Model:
        return self._model
    
    @model.setter
    def model(self, value: config.Model):
        self._model = value
        self._provider_class_name = value.provider_class_name
        
        # Get provider class and initialize
        provider_class = globals().get(self._provider_class_name)
        if not provider_class:
            error_msg = f"Provider class not found: {self._provider_class_name}"
            self._logger.error(error_msg) 
            raise AI_Setup_Error(error_msg)
        
        # Initialize provider
        self.ai = provider_class(self._model, self._system_prompt, self._logger)        
        self._logger.info(f"Model changed to {self._model.name} ({self._model.model_id})")

    @property
    def results(self) -> str:
        """Property that returns the response for compatibility with existing code."""
        return self.response        
    
    @property
    def logger(self) -> Logger:
        return self._logger
        
    
    @logger.setter
    def logger(self, value: Logger):
        self._logger = value
        self._logger.info(f"Logger set for {self._model.name} AI")
        # Update the provider's logger
        if hasattr(self, 'ai'):
            self.ai.logger = value

    def _build_conversation_history(self) -> List[Dict[str, Any]]:
        """Build conversation history with context if available."""
        messages = []
        
        # Add previous conversation as context if available
        if self.questions:
            # Split context into past interactions
            for i in range(len(self.questions)):
                messages.append(self._build_messages(self.questions[i], Role.USER))
                messages.append(self._build_messages(self.responses[i], Role.AI))
        self._logger.debug(f"Conversation history: {messages}")
        return messages

    def _build_messages(self, user_prompt: str, role: Role) -> List[Dict[str, Any]]:
        """Build messages for the AI provider."""
        content_key = "content"
        ai_role = "assistant"
        content_value = user_prompt
        
        if (self.model.provider_class_name == "Gemini"):
            content_key = "parts"
            ai_role = "model"
            content_value = [{"text": user_prompt}]

        role_value = ai_role if role == Role.AI else role.value
        messages = {"role": role_value, content_key: content_value}
        self._logger.debug(f"Message built: {messages}")
        return messages

    def stream(self, user_prompt: str) -> str:
        """Stream the AI response back."""
        self._logger.debug(f"Streaming response for: {user_prompt}")
        messages = []           
        messages.extend(self._build_conversation_history())
        messages.append(self._build_messages(user_prompt, Role.USER))
        response = self.ai.stream(messages)
        thoughts = Parser.extract_text(response, "<think>", "</think>")

        self.thoughts.append(thoughts)
        self.questions.append(user_prompt)
        self.responses.append(response)
    
        return response
    
    def request(self, user_prompt: str) -> str:
        """Request the AI response."""
        self._logger.debug(f"Requesting response for: {user_prompt}")
        messages = []           
        messages.extend(self._build_conversation_history())
        messages.append(self._build_messages(user_prompt, Role.USER))
        response = self.ai.request(messages)
        thoughts = Parser.extract_text(response, "<think>", "</think>")
        self.thoughts.append(thoughts)
        self.questions.append(user_prompt)
        self.responses.append(response)
        return response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.questions = []
        self.responses = []
        self.thoughts = []
        self._logger.info("Conversation history has been reset")


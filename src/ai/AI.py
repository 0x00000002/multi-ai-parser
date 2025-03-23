from typing import List, Dict, Any, Optional, Type, Union
import logging
import src.ai.AIConfig as config
from src.ai.ModelSelector import ModelSelector, UseCase
from src.Parser import Parser
from enum import Enum

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

class AI_Processing_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI_Streaming_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI:
    def __init__(self, 
                 model_param: Union[config.Model, Dict[str, Any], UseCase], 
                 quality: Optional[config.Quality] = None,
                 speed: Optional[config.Speed] = None,
                 use_local: bool = False,
                 system_prompt: str = ""):
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
        """
        self._system_prompt = system_prompt
        self.thoughts = []
        self.questions = []
        self.responses = []

        
        # Determine the model based on input params
        if isinstance(model_param, config.Model):
            # Direct Model instance provided
            self._model = model_param
        elif isinstance(model_param, dict):
            # Dictionary of criteria provided
            params = model_param.copy()
            
            self._model = config.find_model(
                params.get('privacy', config.Privacy.EXTERNAL),
                params.get('quality', config.Quality.MEDIUM),
                params.get('speed', config.Speed.STANDARD)
            )
        elif isinstance(model_param, UseCase):
            # UseCase enum provided, use ModelSelector to get params
            params = ModelSelector.get_model_params(
                use_case=model_param,
                quality=quality or config.Quality.MEDIUM,
                speed=speed or config.Speed.STANDARD,
                use_local=use_local
            )
            
            # If system_prompt wasn't specified, get the default for this use case
            if not system_prompt:
                self._system_prompt = ModelSelector.get_system_prompt(model_param)
                
            self._model = config.find_model(
                params['privacy'],
                params['quality'],
                params['speed']
            )
        else:
            raise ValueError("model_param must be a Model, dictionary, or UseCase enum")
            
        self._provider_class_name = self._model.provider_class_name
        
        # Get the appropriate provider class from globals using the class name
        provider_class = globals().get(self._provider_class_name)
        if not provider_class:
            raise ValueError(f"Provider class not found: {self._provider_class_name}")
            
        # Initialize the provider with the model and system prompt
        self.ai = provider_class(self._model, self._system_prompt)
        
        # Log the selected model
        logging.info(f"Selected model: {self._model.name} ({self._model.model_id})")
        logging.info(f"Model properties: Privacy={self._model.privacy.name}, " 
                     f"Quality={self._model.quality.name}, Speed={self._model.speed.name}")

    @classmethod
    def for_use_case(cls, use_case: UseCase, quality: config.Quality = config.Quality.MEDIUM, 
                     speed: config.Speed = config.Speed.STANDARD, use_local: bool = False, 
                     custom_system_prompt: Optional[str] = None):
        """
        Factory method to create an AI instance for a specific use case.
        
        Args:
            use_case: The specific use case
            quality: Quality level from AIConfig.Quality
            speed: Speed preference from AIConfig.Speed
            use_local: Whether to use local models
            custom_system_prompt: Optional custom system prompt (overrides default)
            
        Returns:
            An AI instance configured for the specified use case
        """
        system_prompt = custom_system_prompt or ModelSelector.get_system_prompt(use_case)
        
        return cls(
            model_param=use_case,
            quality=quality,
            speed=speed,
            use_local=use_local,
            system_prompt=system_prompt
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value: str):
        if not isinstance(value, str):
            raise ValueError("System prompt must be a string")
        self._system_prompt = value

    @property
    def model(self) -> config.Model:
        return self._model
    
    @model.setter
    def model(self, value: config.Model):
        self._model = value
        self._provider_class_name = value.provider_class_name
        
        # Reinitialize the AI provider when model changes
        provider_class = globals().get(self._provider_class_name)
        if not provider_class:
            raise ValueError(f"Provider class not found: {self._provider_class_name}")
        self.ai = provider_class(self._model, self._system_prompt)

    @property
    def results(self) -> str:
        """Property that returns the response for compatibility with existing code."""
        return self.response        
    
    def _build_conversation_history(self) -> List[Dict[str, Any]]:
        """Build conversation history with context if available."""
        messages = []
        
        # Add previous conversation as context if available
        if self.questions:
            # Split context into past interactions
            for i in range(len(self.questions)):
                messages.append(self._build_messages(self.questions[i], Role.USER))
                messages.append(self._build_messages(self.responses[i], Role.AI))
        return messages

    def _build_messages(self, user_prompt: str, role: Role) -> List[Dict[str, Any]]:
        """Build messages for the AI provider."""
        is_gemini = self.model.provider_class_name == "Gemini"
        content_key = "parts" if is_gemini else "content"
        ai_role = "model" if is_gemini else "assistant"
        role_value = ai_role if role == Role.AI else role.value
        content_value = [{"text": user_prompt}] if is_gemini else user_prompt
        return {"role": role_value, content_key: content_value}

    def stream(self, user_prompt: str) -> str:
        """Stream the AI response back."""
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
        # logger.info("Conversation history has been reset")
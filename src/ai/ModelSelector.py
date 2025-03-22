from enum import Enum, auto
from typing import Dict, Any, Optional
import src.ai.AIConfig as config

class UseCase(Enum):
    """Common use cases for AI models across the application."""
    TRANSLATION = auto()
    SUMMARIZATION = auto()
    CODING = auto()
    CHAT = auto()
    CONTENT_GENERATION = auto()
    DATA_ANALYSIS = auto()

class ModelSelector:
    """
    Central model selection logic that can be shared across different components.
    Maps business requirements to model characteristics.
    """
    
    @staticmethod
    def get_model_params(
        use_case: UseCase,
        quality: config.Quality = config.Quality.MEDIUM,
        speed: config.Speed = config.Speed.STANDARD,
        use_local: bool = False
    ) -> Dict[str, Any]:
        """
        Get model parameters based on use case and requirements.
        
        Args:
            use_case: The specific use case (e.g., translation, summarization)
            quality: Desired quality level from AIConfig.Quality
            speed: Speed preference from AIConfig.Speed
            use_local: Whether to use local models
            
        Returns:
            Dictionary with model selection parameters
        """
        # Base parameters based on privacy preference
        params = {
            'privacy': config.Privacy.LOCAL if use_local else config.Privacy.EXTERNAL,
            'quality': quality,
            'speed': speed
        }

        
        # Specific adjustments for different use cases
        if use_case == UseCase.TRANSLATION:
            # Translation might need higher quality for accuracy
            if quality == config.Quality.HIGH:
                # Ensure we use a model great at language tasks
                # (Here we could boost specific aspects if needed)
                pass
                
        elif use_case == UseCase.CODING:
            # For coding, we might prioritize capability over speed
            if quality != config.Quality.LOW and params['speed'] == config.Speed.FAST:
                # Coding requires precision, so we might override speed for quality
                params['speed'] = config.Speed.STANDARD
        
        # Add more use-case specific adjustments as needed
        
        return params

    @staticmethod
    def get_system_prompt(use_case: UseCase) -> str:
        """
        Get an appropriate system prompt for the given use case.
        
        Args:
            use_case: The specific use case
            
        Returns:
            A system prompt tailored to the use case
        """
        prompts = {
            UseCase.TRANSLATION: "You are an expert translator. Translate the text accurately while preserving meaning, tone, and cultural nuances.",
            UseCase.SUMMARIZATION: "You are an expert at summarizing content. Create concise, informative summaries that capture the key points.",
            UseCase.CODING: "You are an expert programmer. Provide clean, efficient, and well-documented code.",
            UseCase.CHAT: "You are a helpful, friendly assistant. Provide accurate and informative responses.",
            UseCase.CONTENT_GENERATION: "You are a creative content creator. Generate engaging, original content.",
            UseCase.DATA_ANALYSIS: "You are a data analysis expert. Analyze data thoroughly and provide insightful interpretations."
        }
        
        return prompts.get(use_case, "You are a helpful assistant.")
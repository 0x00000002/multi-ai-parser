from enum import Enum, auto
from typing import Dict, Any, Optional
import src.ai.AIConfig as config

class UseCase(Enum):
    """Common use cases for AI models across the application."""
    TRANSLATION = auto()
    SUMMARIZATION = auto()
    CODING = auto()
    SOLIDITY_CODING = auto()
    CHAT = auto()
    CONTENT_GENERATION = auto()
    DATA_ANALYSIS = auto()
    WEB_ANALYSIS = auto()
    IMAGE_GENERATION = auto()

class ModelSelector:
    """
    Central model selection logic that can be shared across different components.
    Maps business requirements to model characteristics.
    """
    
    @staticmethod
    def get_model_params(
        use_case: UseCase,
        quality: Optional[config.Quality] = None,
        speed: Optional[config.Speed] = None,
        use_local: Optional[bool] = False
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
        use_case_params = {
            UseCase.TRANSLATION: {
                'quality': config.Quality.LOW,
                'speed': config.Speed.STANDARD
            },
            UseCase.CODING: {
                'quality': config.Quality.MEDIUM,
                'speed': config.Speed.STANDARD
            },
            UseCase.SOLIDITY_CODING: {
                'quality': config.Quality.HIGH,
                'speed': config.Speed.STANDARD
            },
            UseCase.CHAT: {
                'quality': config.Quality.MEDIUM,
                'speed': config.Speed.STANDARD
            },
            UseCase.CONTENT_GENERATION: {
                'quality': config.Quality.MEDIUM,
                'speed': config.Speed.STANDARD
            },
            UseCase.DATA_ANALYSIS: {
                'quality': config.Quality.HIGH,
                'speed': config.Speed.SLOW
            },  
            UseCase.WEB_ANALYSIS: {
                'quality': config.Quality.MEDIUM,
                'speed': config.Speed.STANDARD
            },
            UseCase.IMAGE_GENERATION: {
                'quality': config.Quality.HIGH,   
                'speed': config.Speed.SLOW
            }
        }

        # Base parameters based on privacy preference
        params = {
            'privacy': config.Privacy.LOCAL if use_local else config.Privacy.EXTERNAL,
            'quality': quality if quality else use_case_params[use_case]['quality'],
            'speed': speed if speed else use_case_params[use_case]['speed']
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
            UseCase.SOLIDITY_CODING: "You are an expert Solidity programmer. Provide safe, clean, gas-efficient, and well-documented Solidity code.",
            UseCase.CHAT: "You are a helpful, friendly assistant. Provide accurate and informative responses.",
            UseCase.CONTENT_GENERATION: "You are a creative content creator. Generate engaging, original content.",
            UseCase.DATA_ANALYSIS: "You are a data analysis expert. Analyze data thoroughly and provide insightful interpretations.",
            UseCase.WEB_ANALYSIS: "You are an expert web pages analyst. Analyze web pages thoroughly and provide insightful interpretations.",
            UseCase.IMAGE_GENERATION: "You are an expert image generator. Generate high-quality, realistic images based on text descriptions.",
        }
        
        return prompts.get(use_case, "You are a helpful assistant.")
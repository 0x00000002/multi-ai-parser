"""
Model selection logic that works with our configuration system.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
from ..config.config_manager import ConfigManager, ModelConfig, Model
from ..exceptions import AISetupError
from src.config.models import Model as ModelEnum


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
    
    @classmethod
    def from_string(cls, name: str) -> 'UseCase':
        """Convert string to UseCase enum."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown use case: {name}")


class ModelSelector:
    """
    Smart model selection based on use cases and requirements.
    Uses configuration from config.yml for model definitions.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the model selector.
        
        Args:
            config_manager: Configuration manager instance
        """
        self._config_manager = config_manager
        # Build a mapping between config model_ids and Model enum values
        self._model_id_to_enum = self._build_model_mapping()
    
    def _build_model_mapping(self) -> Dict[str, ModelEnum]:
        """
        Build a mapping from config model_ids to Model enum values.
        
        This approach decouples the naming conventions in the config 
        from the enum values by creating a direct lookup table.
        
        Returns:
            Dictionary mapping model_id strings to Model enum values
        """
        mapping = {}
        config_models = self._config_manager.get_all_models()
        
        # First try to find exact matches
        for model_key, model_config in config_models.items():
            model_id = model_config.model_id
            
            # Try direct match first
            for enum_model in ModelEnum:
                if enum_model.value == model_id:
                    mapping[model_id] = enum_model
                    break
        
        # For any unmatched model_ids, use the config key as a fallback to find a match
        for model_key, model_config in config_models.items():
            model_id = model_config.model_id
            if model_id not in mapping:
                for enum_model in ModelEnum:
                    if enum_model.name.lower() == model_key.replace('-', '_').lower():
                        mapping[model_id] = enum_model
                        break
        
        return mapping
    
    def select_model(self,
                    use_case: UseCase,
                    quality: Optional[str] = None,
                    speed: Optional[str] = None,
                    privacy: Optional[str] = None,
                    max_cost: Optional[float] = None,
                    estimated_tokens: Optional[Tuple[int, int]] = None) -> Model:
        """
        Select the most appropriate model based on use case and requirements.
        
        Args:
            use_case: The specific use case
            quality: Desired quality level (HIGH, MEDIUM, LOW)
            speed: Speed preference (FAST, STANDARD, SLOW)
            privacy: Privacy preference (LOCAL, EXTERNAL)
            max_cost: Maximum cost per request
            estimated_tokens: Tuple of (input_tokens, output_tokens) for cost estimation
            
        Returns:
            The selected model enum
            
        Raises:
            AISetupError: If no suitable model is found
        """
        # Get base parameters from use case configuration
        use_case_name = use_case.name.lower()
        params = self._config_manager.get_use_case_config(use_case_name)
        # Override with explicit parameters if provided
        if quality:
            params['quality'] = quality
        if speed:
            params['speed'] = speed
        if privacy:
            params['privacy'] = privacy
        
        # Get all available models
        models = self._config_manager.get_all_models()
        # Filter models based on requirements
        candidates = self._filter_models(models, params)
        
        if not candidates:
            raise AISetupError(
                f"No suitable model found for use case {use_case} "
                f"with quality={params['quality']}, "
                f"speed={params['speed']}, "
                f"privacy={params.get('privacy', 'EXTERNAL')}"
            )
        
        
        # Apply cost constraints if specified
        if max_cost is not None and estimated_tokens is not None:
            candidates = self._apply_cost_constraints(
                candidates, 
                max_cost, 
                estimated_tokens
            )
            
            if not candidates:
                raise AISetupError(
                    f"No models found within cost limit of {max_cost} "
                    f"for estimated tokens {estimated_tokens}"
                )
        
        # Select the best matching model
        model = self._select_best_model(candidates, params)
        
        # Get the model_id from the selected candidate
        model_id = candidates[0].model_id
        
        # Look up the model_id in our mapping
        if model_id in self._model_id_to_enum:
            return self._model_id_to_enum[model_id]
            
        # If the model_id isn't in our mapping, try to find a match in the enum values directly
        try:
            return Model(model_id)
        except ValueError:
            # If we can't find a direct match, raise a clear error
            raise AISetupError(
                f"No matching Model enum found for model_id '{model_id}'. "
                f"Please update the Model enum in src/config/models.py to include this model ID exactly."
            )
    
    def _filter_models(self, 
                      models: Dict[str, ModelConfig], 
                      params: Dict[str, str]) -> List[ModelConfig]:
        """Filter models based on requirements."""
        candidates = []
        
        for model_id, model_config in models.items():
            # Check if model matches quality and speed requirements
            if (model_config.quality == params['quality'] and 
                model_config.speed == params['speed']):
                
                # Check privacy if specified
                if 'privacy' in params:
                    if model_config.privacy == params['privacy']:
                        candidates.append(model_config)
                else:
                    candidates.append(model_config)
        
        return candidates
    
    def _apply_cost_constraints(self,
                              candidates: List[ModelConfig],
                              max_cost: float,
                              estimated_tokens: Tuple[int, int]) -> List[ModelConfig]:
        """Filter models based on cost constraints."""
        input_tokens, output_tokens = estimated_tokens
        affordable_models = []
        
        for model in candidates:
            # Get cost configuration from model
            cost_config = getattr(model, 'cost', {})
            if not cost_config:
                continue
                
            # Calculate estimated cost
            input_cost = cost_config['input_tokens'] * input_tokens
            output_cost = cost_config['output_tokens'] * output_tokens
            total_cost = max(input_cost + output_cost, cost_config['minimum_cost'])
            
            if total_cost <= max_cost:
                affordable_models.append(model)
        
        return affordable_models
    
    def _select_best_model(self, 
                          candidates: List[ModelConfig], 
                          params: Dict[str, str]) -> str:
        """Select the best model from candidates."""
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0].model_id
        
        # Sort by quality and speed preferences
        # Higher quality and faster speed are preferred
        quality_weights = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        speed_weights = {'FAST': 3, 'STANDARD': 2, 'SLOW': 1}
        
        candidates.sort(
            key=lambda x: (
                quality_weights[x.quality],
                speed_weights[x.speed]
            ),
            reverse=True
        )
        return candidates[0]
        # return candidates[0].model_id
    
    def get_system_prompt(self, use_case: UseCase) -> str:
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
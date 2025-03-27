import os
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv; load_dotenv()


class Provider(Enum):
    OPENAI = "ChatGPT"
    ANTHROPIC = "ClaudeAI"
    OLLAMA = "Ollama"
    GOOGLE = "Gemini"

class Privacy(Enum):
    LOCAL = auto()
    EXTERNAL = auto()

class Quality(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()

class Speed(Enum):
    FAST = auto()
    STANDARD = auto()
    SLOW = auto()

class Model(Enum):
    # OpenAI models
    CHATGPT_O3_MINI = ("gpt-o3-mini", Provider.OPENAI.value, Privacy.EXTERNAL, Quality.HIGH, Speed.FAST)
    CHATGPT_4O_MINI = ("gpt-4o-mini", Provider.OPENAI.value, Privacy.EXTERNAL, Quality.MEDIUM, Speed.FAST)

    # Anthropic models
    CLAUDE_SONNET_3_7 = ("claude-3-7-sonnet-20250219", Provider.ANTHROPIC.value, Privacy.EXTERNAL, Quality.HIGH, Speed.FAST)
    CLAUDE_SONNET_3_5 = ("claude-3-5-sonnet-latest", Provider.ANTHROPIC.value, Privacy.EXTERNAL, Quality.MEDIUM, Speed.FAST)
    CLAUDE_HAIKU_3_5 = ("claude-3-5-haiku-latest", Provider.ANTHROPIC.value, Privacy.EXTERNAL, Quality.LOW, Speed.FAST)
    
    # Ollama models
    OLLAMA_GEMMA3 = ("gemma3:27b", Provider.OLLAMA.value, Privacy.LOCAL, Quality.MEDIUM, Speed.STANDARD)
    OLLAMA_DEEPSEEK_R1_32B = ("deepseek-r1:32b", Provider.OLLAMA.value, Privacy.LOCAL, Quality.MEDIUM, Speed.SLOW)
    OLLAMA_QWQ = ("qwq", Provider.OLLAMA.value, Privacy.LOCAL, Quality.LOW, Speed.STANDARD)
    OLLAMA_DEEPSEEK_R1_7B = ("deepseek-r1:7b", Provider.OLLAMA.value, Privacy.LOCAL, Quality.LOW, Speed.STANDARD)
    
    # Google models
    GEMINI_1_5_PRO = ("gemini-1.5-pro", Provider.GOOGLE.value, Privacy.EXTERNAL, Quality.MEDIUM, Speed.FAST)
    GEMINI_2_FLASH = ("gemini-2-flash", Provider.GOOGLE.value, Privacy.EXTERNAL, Quality.HIGH, Speed.FAST)
    GEMINI_2_FLASH_LITE = ("gemini-2-flash-lite", Provider.GOOGLE.value, Privacy.EXTERNAL, Quality.MEDIUM, Speed.FAST)

    def __init__(self, model_id: str, provider_class: str, privacy: Privacy, quality: Quality, speed: Speed):
        self.model_id = model_id
        self.provider_class_name = provider_class
        self.privacy = privacy
        self.quality = quality
        self.speed = speed

    @property
    def provider_class_name(self) -> str:
        return self._provider_class_name
        
    @provider_class_name.setter
    def provider_class_name(self, value: str):
        self._provider_class_name = value
    
    @property
    def api_key(self) -> Optional[str]:
        provider_to_env = {
            "ChatGPT": "OPENAI_API_KEY",
            "ClaudeAI": "ANTHROPIC_API_KEY",
            "Ollama": "OLLAMA_API_KEY",
            "Gemini": "GOOGLE_API_KEY"
        }
        env_var = provider_to_env.get(self.provider_class_name)
        return os.getenv(env_var) if env_var else None
        
    def __str__(self):
        return self.model_id

def find_model(privacy: Privacy, quality: Quality, speed: Speed) -> Model:
    """
    Find the most suitable model that matches the specified criteria.
    
    Args:
        privacy: Privacy level preference
        quality: Quality level preference
        speed: Speed preference
        
    Returns:
        The best matching Model
    
    Raises:
        ValueError: If no matching model is found
    """
    candidates = []
    
    # Find all models that match the exact criteria
    for model in Model:
        if (model.privacy == privacy and 
            model.quality == quality and 
            model.speed == speed):
            candidates.append(model)
    
    # If we found exact matches, return the first one
    if candidates:
        return candidates[0]
    
    # If no exact matches, try to find the closest match
    # Start relaxing criteria in order of priority (adjust as needed)
    # Here we prioritize: 1. Privacy, 2. Quality, 3. Speed
    all_models = list(Model)
    
    # First, ensure privacy is matched
    privacy_matches = [m for m in all_models if m.privacy == privacy]
    if not privacy_matches:
        raise ValueError(f"No models available with privacy setting {privacy}")
    
    # Among privacy matches, find closest quality match
    # We consider one level up or one level down in quality
    quality_value = quality.value
    quality_matches = [
        m for m in privacy_matches if 
        abs(m.quality.value - quality_value) <= 1
    ]
    
    if not quality_matches:
        # If no close quality matches, just use all privacy matches
        quality_matches = privacy_matches
    
    # Sort by how close they are to the requested quality (exact matches first)
    quality_matches.sort(key=lambda m: abs(m.quality.value - quality_value))
    
    # Among quality matches, find speed matches or closest alternatives
    speed_matches = [m for m in quality_matches if m.speed == speed]
    
    if speed_matches:
        return speed_matches[0]
    else:
        # Return the best quality match regardless of speed
        return quality_matches[0]

# Convenience accessors for model selection
PRIVACY = Privacy
QUALITY = Quality
SPEED = Speed
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import src.ai.AIConfig as config

import logging
import anthropic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AI_Processing_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI_Streaming_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI_API_Key_Error(Exception):
    """Custom exception for API key related errors."""
    pass

class ClaudeParams(BaseModel):
    model: str
    max_tokens: int = 1024
    system: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    system_instruction: Optional[str] = None
    max_output_tokens: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None

class ClaudeAI:    
    def __init__(self, model: config.Model = config.Model.CLAUDE_SONNET_3_5, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.response = ""
        self.model = model

        # Validation check - ensure this is an Anthropic model
        if model.provider_class_name != "ClaudeAI":
            raise ValueError(f"Model {model.name} is not compatible with ClaudeAI. It uses {model.provider_class_name}.")
        
        api_key = model.api_key
        if not api_key or not api_key.startswith("sk-ant-api"):
            raise AI_API_Key_Error("No valid Anthropic API key found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Successfully initialized client for {model.name} model")

    def _get_params(self, messages, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model.model_id,  # Use model_id instead of the model enum
            "system": self.system_prompt,
            "messages": messages,
            **{k: v for k, v in optional_params.items() if k in ClaudeParams.__annotations__ and v is not None}
        }
        
        # Create and validate with Pydantic model directly
        validated_params = ClaudeParams(**params)        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)

    def stream(self, messages, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(messages, optional_params)
        try:
            response = ""
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response += text                
            return response
        except Exception as e:
            logger.error(f"ClaudeAI {self.model.model_id}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"ClaudeAI {self.model.model_id}: Error during streaming: {str(e)}")

    def request(self, messages, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(messages, optional_params)
        
        try:
            response = self.client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"ClaudeAI {self.model.model_id}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"ClaudeAI {self.model.model_id}: Error during request: {str(e)}")
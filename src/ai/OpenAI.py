from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

import src.ai.AIConfig as config
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OpenAIParams(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_completion_tokens: Optional[int] = 1024
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    stream: Optional[bool] = None

class ChatGPT:    
    def __init__(self, model: config.Model = config.Model.CHATGPT_4O_MINI, system_prompt: str = ""):
        self.system_prompt = system_prompt
        self.response = ""
        self.model = model

        # Validation check - ensure this is an Anthropic model
        if model.provider_class_name != "ChatGPT":
            raise ValueError(f"Model {model.name} is not an OpenAI model. It belongs to {model.provider_class_name}.")
        
        api_key = model.api_key
        if not api_key or not api_key.startswith("sk-proj-"):
            raise AI_API_Key_Error("No valid OpenAI API key found")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Successfully initialized client for {model.name} model")


    def _get_params(self, messages, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model.model_id,
            "messages": messages,
            **{k: v for k, v in optional_params.items() if k in OpenAIParams.__fields__ and v is not None}
        }
        
        # Create and validate with Pydantic model directly
        validated_params = OpenAIParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)


    def stream(self, messages, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(messages, optional_params)
        params["stream"] = True
        
        try:
            response = ""
            for chunk in self.client.chat.completions.create(**params):
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    response += text
            
            return response
        except Exception as e:
            logger.error(f"OpenAI{self.model}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"OpenAI{self.model}: Error during streaming: {str(e)}")

    def request(self, messages, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(messages, optional_params)
        
        try:
            response = self.client.chat.completions.create(**params)    
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI{self.model}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"OpenAI{self.model}: Error during request: {str(e)}")

    @property
    def results(self):
        """Property that returns the response for compatibility with existing code."""
        return self.response



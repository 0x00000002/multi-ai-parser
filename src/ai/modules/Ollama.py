#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.Logger import Logger

import ollama
import src.ai.AIConfig as config


# In[2]:
class OllamaParams(BaseModel):
    model: str
    stream: Optional[bool] = None
    messages: List[Dict[str, str]]

class Ollama:    
    def __init__(self, model: config.Model = config.Model.OLLAMA_GEMMA3, system_prompt: str = "", logger: Optional[Logger] = None):
        self.system_prompt = system_prompt
        self.response = ""
        self.model = model
        self.logger = logger

        # Validation check - ensure this is an Anthropic model
        if model.provider_class_name != "Ollama":
            raise ValueError(f"Model {model.name} is not compatible with Ollama. It uses {model.provider_class_name}.")
        
        # Ollama typically runs locally, so no API key needed
        # Just check if the service is available
        try:
            self.client = ollama.Client()
            # Test connection
            self.client.list()
            if self.logger:
                self.logger.info(f"Successfully initialized client for provider Ollama")
        except Exception as e:
            raise AI_API_Key_Error(f"Could not connect to Ollama service : {str(e)}")
        
        if self.logger:
            self.logger.info(f"Successfully initialized client for {model.name} model")


    def _get_params(self, messages, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model.model_id,
            "messages": messages,
            **{k: v for k, v in optional_params.items() if k in OllamaParams.__annotations__ and v is not None}
        }

        # Create and validate with Pydantic model directly
        validated_params = OllamaParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)

    def stream(self, messages, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(messages, optional_params)
        
        try:
            response = ""
            for chunk in self.client.chat(**params):
                # Since chunk is a tuple like ('message', Message(role='assistant', content='...'))
                if chunk[0] == 'message':  # Check if it's a message chunk
                    text = chunk[1].content  # Access the content directly from Message object
                    if text:
                        print(text, end="", flush=True)
                        response += text   
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ollama{self.model}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"Ollama {self.model}: Error during streaming: {str(e)}")

    def request(self, messages, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(messages, optional_params)
        
        try:
            res = self.client.chat(**params)
            response = res.get("message", {}).get("content", "")
            return response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Ollama{self.model}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"Ollama {self.model}: Error during request: {str(e)}")


#!/usr/bin/env python
# coding: utf-8

import os
from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model(Enum):
    O3_MINI = 1

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
    def __init__(self, system_prompt, model: Model = Model.O3_MINI):
        self.openai_models = {
            Model.O3_MINI: "o3-mini"
        }

        self.system_prompt = system_prompt
        self.response = ""
        self.context = ""
        self.model = self.openai_models[model]

        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise AI_API_Key_Error("No OpenAI API key found")
        
        self.api_key = api_key
        self.client = OpenAI()
        logger.info(f"Successfully initialized client for provider OpenAI")

    def _get_params(self, user_prompt, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Prepare messages with system prompt and user message
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Construct params dict with only non-None values
        params = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            **{k: v for k, v in optional_params.items() if k in OpenAIParams.__fields__ and v is not None}
        }
        
        # Create and validate with Pydantic model directly
        validated_params = OpenAIParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)

    def _save_response(self, response, user_prompt):
        """Save the response and user prompt for later reference."""
        if isinstance(response, str):
            self.response = response
        else:
            # For OpenAI completion objects
            self.response = response.choices[0].message.content if hasattr(response, 'choices') else ""
        
        self.context = user_prompt
        return self.response

    def stream(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(user_prompt, optional_params)
        params["stream"] = True
        
        try:
            response = ""
            for chunk in self.client.chat.completions.create(**params):
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    response += text
            
            self._save_response(response, user_prompt)      
            return self.response
        except Exception as e:
            logger.error(f"OpenAI{self.model}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"OpenAI{self.model}: Error during streaming: {str(e)}")

    def request(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            response = self.client.chat.completions.create(**params)
            self._save_response(response, user_prompt)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI{self.model}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"OpenAI{self.model}: Error during request: {str(e)}")

    @property
    def results(self):
        """Property that returns the response for compatibility with existing code."""
        return self.response




# Example usage
if __name__ == "__main__":
    ai = ChatGPT("You are an expert in Solidity smart contract development.")
    response = ai.request("Explain gas optimization in Solidity")
    print("\nFull response:", response)


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In[2]:
class Model(Enum):
    DEEPSEEK_SMALL = 1
    DEEPSEEK_LARGE = 2
    QWQ = 3
    GEMMA3 = 4

class OllamaParams(BaseModel):
    model: str
    stream: Optional[bool] = None
    messages: List[Dict[str, str]]

class Ollama:    
    def __init__(self, system_prompt, model: Model = Model.GEMMA3):
        self.ollama_models = {
            Model.DEEPSEEK_SMALL: "deepseek-r1:7b",
            Model.DEEPSEEK_LARGE: "deepseek-r1:32b",
            Model.QWQ: "qwq:latest",
            Model.GEMMA3: "gemma3:27b",
        }

        self.system_prompt = system_prompt
        self.response = ""
        self.context = ""
        self.model = self.ollama_models[model]

        # Ollama typically runs locally, so no API key needed
        # Just check if the service is available
        try:
            self.client = ollama.Client()
            # Test connection
            self.client.list()
            logger.info(f"Successfully initialized client for provider Ollama")
        except Exception as e:
            raise AI_API_Key_Error(f"Could not connect to Ollama service : {str(e)}")

    def _get_params(self, user_prompt, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}, {"role": "system", "content": self.system_prompt}],
        }
        
        # Create and validate with Pydantic model directly
        validated_params = OllamaParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)

    def _save_response(self, response, user_prompt):
        """Save the response and user prompt for later reference."""
        self.response = response        
        self.context = user_prompt
        return self.response

    def stream(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            response = ""
            for chunk in self.client.chat(**params):
                # Since chunk is a tuple like ('message', Message(role='assistant', content='...'))
                if chunk[0] == 'message':  # Check if it's a message chunk
                    text = chunk[1].content  # Access the content directly from Message object
                    if text:
                        print(text, end="", flush=True)
                        response += text   
            self._save_response(response, user_prompt)      
            return self.response
        except Exception as e:
            logger.error(f"Ollama{self.model}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"Ollama{self.model}: Error during streaming: {str(e)}")

    def request(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(user_prompt, optional_params)
        print(params)
        
        try:
            res = self.client.chat(**params)
            response = res.get("message", {}).get("content", "")
            self._save_response(response, user_prompt)
            return self.response
        except Exception as e:
            logger.error(f"Ollama{self.model}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"Ollama{self.model}: Error during request: {str(e)}")


    @property
    def results(self):
        """Property that returns the response for compatibility with existing code."""
        return self.response




# In[100]:

# ai = Ollama("you are an expert in Solidity. Answer user question with all your knowledge in smart contracts development.")
# ai.request("tell story", "Tell any story for 50 words")


# # In[101]:

# # print(ai.response + "\n\n----------------\n\n" + ai.thoughts)
# ai.stream("summarize story", "summarize the story in 12 words")


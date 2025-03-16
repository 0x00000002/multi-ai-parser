#!/usr/bin/env python
# coding: utf-8

import os
from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import anthropic


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Model(Enum):
    SONNET_3_7 = 1
    SONNET_3_5 = 2
    HAIKU_3_5 = 3

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
    def __init__(self, system_prompt, model: Model = Model.SONNET_3_7):
        self.claude_models = {
        Model.SONNET_3_7: "claude-3-7-sonnet-20250219",
        Model.SONNET_3_5: "claude-3-5-sonnet",
        Model.HAIKU_3_5: "claude-3-5-haiku"
        }

        self.system_prompt = system_prompt
        self.thoughts = ""
        self.response = ""
        self.context = ""
        self.model = self.claude_models[model]

        load_dotenv(override=True)
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key or not api_key.startswith("sk-ant-api"):
            raise AI_API_Key_Error("No Anthropic API key found")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Successfully initialized client for provider Anthropic")


    def _get_params(self, user_prompt, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model,
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            **{k: v for k, v in optional_params.items() if k in ClaudeParams.__fields__ and v is not None}
        }
        
        # Create and validate with Pydantic model directly
        validated_params = ClaudeParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True) if hasattr(validated_params, 'model_dump') else validated_params.dict(exclude_none=True)


    def _save_response(self, response, user_prompt):
        """Save the response and user prompt for later reference."""
        if isinstance(response, str):
            self.response = response
        else:
            # For anthropic.Message objects
            self.response = response.content[0].text if response.content else ""
        
        self.context = user_prompt
        return self.response

    def stream(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Stream the AI response back."""
        params = self._get_params(user_prompt, optional_params)
        try:
            response = ""
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response += text                
            self._save_response(response, user_prompt)      
            return self.response
        except Exception as e:
            logger.error(f"ClaudeAI{self.model}: Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"ClaudeAI{self.model}: Error during streaming: {str(e)}")

    def request(self, user_prompt, optional_params: Dict[str, Any] = {}):
        """Make a non-streaming request to the AI model."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            response = self.client.messages.create(**params)
            self._save_response(response, user_prompt)
            return response.content[0].text
        except Exception as e:
            logger.error(f"ClaudeAI{self.model}: Error during request: {str(e)}")
            raise AI_Processing_Error(f"ClaudeAI{self.model}: Error during request: {str(e)}")





# In[10]:

claude = ClaudeAI("you are an expert in Solidity. Answer user question with all your knowledge in smart contracts development.")
# ai.stream("tell story", "Tell any story for 50 words")


# In[11]:

# print(ai.response + "\n\n----------------\n\n" + ai.thoughts)
# claude.stream("summarize story", "summarize the story in 12 words")


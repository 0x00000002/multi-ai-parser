#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from src.ai.Errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import src.ai.AIConfig as config
from src.Logger import Logger

from google import genai


# In[2]:

class GeminiConfig(BaseModel):
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Controls randomness in the output. Values closer to 0 make the output more focused and deterministic"
    )
    max_output_tokens: int = Field(
        default=1024,
        gt=0,
        description="The maximum number of tokens to generate in the response"
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The cumulative probability cutoff for token selection"
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=0,
        description="The number of highest probability tokens to consider for generation"
    )
    candidate_count: Optional[int] = Field(
        default=1,
        gt=0,
        description="Number of candidate responses to generate"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="List of strings that will stop generation when encountered"
    )

class GeminiParams(BaseModel):
    model: str = Field(
        description="The name of the Gemini model to use"
    )
    contents: List[Dict[str, Any]] = Field(
        description="The input contents for the model"
    )
    system_instruction: Optional[str] = Field(
        default=None,
        description="Optional system-level instruction to guide the model's behavior"
    )
    safety_settings: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Safety settings to control content filtering"
    )
    config: Optional[GeminiConfig] = Field(
        default_factory=GeminiConfig,
        description="Configuration parameters for model generation"
    )

    class Config:
        arbitrary_types_allowed = True

class Gemini:
    def __init__(self, model: config.Model = config.Model.GEMINI_1_5_PRO, system_prompt: str = "", logger: Optional[Logger] = None):
        self.system_prompt = system_prompt
        self.response = ""
        self.model = model
        self.logger = logger

        # Validation check - ensure this is an Anthropic model
        if model.provider_class_name != "Gemini":
            raise ValueError(f"Model {model.name} is not an Google model. It belongs to {model.provider_class_name}.")
        
        load_dotenv(override=True)
        api_key = model.api_key
        if not api_key or not api_key.startswith("AIza"):
            raise AI_API_Key_Error("No valid Google API key found")
        
        # Configure the Gemini API
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise AI_API_Key_Error(f"Failed to configure Google Gemini API: {str(e)}")
        
        if self.logger:
            self.logger.info(f"Successfully initialized client for {model.name} model")
        

    def _build_conversation_history(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Build conversation history with context if available."""
        contents = []
        
        # Add previous conversation as context if available
        if self.context:
            # Split context into past interactions
            interactions = self.context.split("\n\nThe response: \n\n")
            for i in range(0, len(interactions)-1, 2):
                if i+1 < len(interactions):
                    contents.append({"role": "user", "parts": [{"text": interactions[i]}]})
                    contents.append({"role": "model", "parts": [{"text": interactions[i+1]}]})
        
        # Add current user prompt
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        
        return contents

    def _get_params(self, contents: List[Dict[str, Any]], optional_params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get the parameters for the request.
        
        Args:
            user_prompt: The user's input text
            optional_params: Additional parameters to customize the request
        
        Returns:
            Dict containing the formatted parameters for the Gemini API request
        """
        # Build conversation with history if available
        
        
        # Initialize the config with default values
        config = GeminiConfig().dict()
        
        # Update config with any custom generation parameters
        if 'config' in optional_params:
            user_config = optional_params.pop('config', {})
            # Only update with valid config parameters
            valid_config_params = set(GeminiConfig.__fields__.keys())
            config.update({
                k: v for k, v in user_config.items() 
                if k in valid_config_params and v is not None
            })
        
        # Construct base params
        params = {
            "contents": contents,
            "model": self.model.model_id,
            "config": config,
        }
        
        # Add system instruction if provided
        if "system_instruction" in optional_params:
            params["system_instruction"] = optional_params.pop("system_instruction")
        
        # Add safety settings if provided
        if "safety_settings" in optional_params:
            params["safety_settings"] = optional_params.pop("safety_settings")
        
        # Add any remaining valid parameters
        valid_params = set(GeminiParams.__fields__.keys()) - {"model", "contents"}
        params.update({
            k: v for k, v in optional_params.items() 
            if k in valid_params and v is not None
        })
        
        return params
        

    def stream(self, user_prompt: str, optional_params: Dict[str, Any] = {}) -> str:
        """Stream the AI response back."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            # Start the streaming response
            response = ""
            for chunk in self.client.models.generate_content_stream(**params):
                if hasattr(chunk, 'text'):
                    text = chunk.text
                    print(text, end="", flush=True)
                    response += text
                elif hasattr(chunk, 'parts') and len(chunk.parts) > 0:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end="", flush=True)
                            response += part.text
            return response
        except Exception as e:
            error_msg = f"Error streaming from Gemini {self.model}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            
            if "rate limit" in str(e).lower():
                raise AI_Streaming_Error(f"Rate limit exceeded. Please try again later: {str(e)}")
            elif "safety" in str(e).lower():
                raise AI_Streaming_Error(f"Content blocked by safety filters: {str(e)}")
            else:
                raise AI_Streaming_Error(error_msg)

    def request(self, user_prompt: str, optional_params: Dict[str, Any] = {}) -> str:
        """Make a non-streaming request to the AI model."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            # Generate the content
            res = self.client.models.generate_content(**params)
            return res.text
        except Exception as e:
            error_msg = f"Error requesting from Gemini {self.model}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            
            if "rate limit" in str(e).lower():
                raise AI_Processing_Error(f"Rate limit exceeded. Please try again later: {str(e)}")
            elif "safety" in str(e).lower():
                raise AI_Processing_Error(f"Content blocked by safety filters: {str(e)}")
            else:
                raise AI_Processing_Error(error_msg)



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
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In[2]:
class Model(Enum):
    GEMINI_PRO = 1
    GEMINI_FLASH = 2
    GEMINI_VISION = 3

class GeminiParams(BaseModel):
    model: str
    contents: List[Dict[str, Any]]
    system_instruction: Optional[str] = None
    safety_settings: Optional[List[Dict[str, Any]]] = None 
    generation_config: Optional[Dict[str, Any]] = None

class Gemini:
    def __init__(self, system_prompt, model: Model = Model.GEMINI_FLASH):
        self.gemini_models = {
            Model.GEMINI_PRO: "gemini-1.5-pro",
            Model.GEMINI_FLASH: "gemini-1.5-flash",
            Model.GEMINI_VISION: "gemini-1.5-pro-vision",
        }

        self.system_prompt = system_prompt
        self.response = ""
        self.context = ""
        self.model_name = self.gemini_models[model]
        self._model_instance = None

        # Initialize API
        self._initialize_api()
        
        # Initialize the generation config with reasonable defaults
        self.default_generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        logger.info(f"Successfully initialized client for Google Gemini with model: {self.model_name}")
    
    def _initialize_api(self) -> None:
        """Initialize the Google Generative AI API."""
        load_dotenv(override=True)
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise AI_API_Key_Error("No Google API key found in environment variables")
        
        # Check for a valid API key format
        if not api_key.startswith("AIza"):
            logger.warning("The Google API key may be invalid. It should start with 'AIza'")
        
        # Configure the Gemini API
        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            raise AI_API_Key_Error(f"Failed to configure Google Gemini API: {str(e)}")

    def _get_model_instance(self) -> Any:
        """Get or create a model instance with the system instruction."""
        if not self._model_instance:
            try:
                self._model_instance = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.system_prompt
                )
                logger.debug(f"Created new model instance for {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to create model instance: {str(e)}")
                raise AI_Processing_Error(f"Failed to create model instance: {str(e)}")
        
        return self._model_instance

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

    def _get_params(self, user_prompt: str, optional_params: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Get the parameters for the request."""
        # Build conversation with history if available
        if optional_params.get('use_conversation_history', True) and self.context:
            contents = self._build_conversation_history(user_prompt)
        else:
            # Simple user message with no history
            contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
        
        # Prepare generation config, merging defaults with any provided values
        generation_config = self.default_generation_config.copy()
        if 'generation_config' in optional_params:
            generation_config.update(optional_params.pop('generation_config', {}))
            
        # Construct params dict with only non-None values
        params = {
            "contents": contents,
            "generation_config": generation_config,
        }
        
        # Add any additional valid parameters
        valid_params = set(GeminiParams.__fields__.keys()) - {"model", "contents", "system_instruction"}
        params.update({k: v for k, v in optional_params.items() if k in valid_params and v is not None})
        
        return params

    def _save_response(self, response: Any, user_prompt: str) -> str:
        """Save the response and user prompt for later reference."""
        # Extract text from different response types
        if isinstance(response, str):
            response_text = response
        elif hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'parts') and len(response.parts) > 0:
            response_text = ''.join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            logger.warning(f"Unexpected response format: {type(response)}")
            response_text = str(response)
        
        # Store the response
        self.response = response_text
        
        # Update context for conversation history
        if self.context:
            self.context += f"\n\nThe response: \n\n{response_text}\n\n{user_prompt}"
        else:
            self.context = f"{user_prompt}\n\nThe response: \n\n{response_text}"
        
        return self.response

    def stream(self, user_prompt: str, optional_params: Dict[str, Any] = {}) -> str:
        """Stream the AI response back."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            # Get the model instance with system instruction already set
            model = self._get_model_instance()
            
            # Start the streaming response
            response = ""
            for chunk in model.generate_content(**params, stream=True):
                if hasattr(chunk, 'text'):
                    text = chunk.text
                    print(text, end="", flush=True)
                    response += text
                elif hasattr(chunk, 'parts') and len(chunk.parts) > 0:
                    for part in chunk.parts:
                        if hasattr(part, 'text') and part.text:
                            print(part.text, end="", flush=True)
                            response += part.text
            
            self._save_response(response, user_prompt)      
            return self.response
        except Exception as e:
            error_msg = f"Error streaming from Gemini {self.model_name}: {str(e)}"
            logger.error(error_msg)
            
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
            # Get the model instance with system instruction already set
            model = self._get_model_instance()
            
            # Generate the content
            response = model.generate_content(**params)
            self._save_response(response, user_prompt)
            return self.response
        except Exception as e:
            error_msg = f"Error requesting from Gemini {self.model_name}: {str(e)}"
            logger.error(error_msg)
            
            if "rate limit" in str(e).lower():
                raise AI_Processing_Error(f"Rate limit exceeded. Please try again later: {str(e)}")
            elif "safety" in str(e).lower():
                raise AI_Processing_Error(f"Content blocked by safety filters: {str(e)}")
            else:
                raise AI_Processing_Error(error_msg)

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.context = ""
        logger.info("Conversation history has been reset")

    @property
    def results(self) -> str:
        """Property that returns the response for compatibility with existing code."""
        return self.response




# In[10]:

ai = Gemini("you are an expert in Solidity. Answer user question with all your knowledge in smart contracts development.")
ai.stream("Tell any story for 50 words")


# In[11]:

# print(ai.response + "\n\n----------------\n\n" + ai.thoughts)
ai.stream("summarize the story in 12 words")


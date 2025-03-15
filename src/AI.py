#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display
from enum import Enum
import json
from pydantic import BaseModel, TypeAdapter
from typing import List, Dict, Any, Optional
import logging
from anthropic import Anthropic
from openai import OpenAI
import ollama
import google.generativeai as genai

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
ollama_api_key = os.getenv('OLLAMA_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

if not openai_api_key or not anthropic_api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not openai_api_key.startswith("sk-proj-") or not anthropic_api_key.startswith("sk-ant-api"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif openai_api_key.strip() != openai_api_key or anthropic_api_key.strip() != anthropic_api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
elif not google_api_key.startswith("AIza"):
    print("An API key was found, but it doesn't start AIza; please check you're using the right key - see troubleshooting notebook")



# In[2]:

class Provider(Enum):
    OPENAI = 1
    ANTHROPIC = 2
    OLLAMA = 3
    GOOGLE = 4

class AIModel(Enum):
    CHATGPT = 1
    CLAUDE = 2
    DEEPSEEK_SMALL = 3
    DEEPSEEK_LARGE = 4
    QWQ = 5
    PHI = 6
    GEMMA3_LOCAL = 7
    GEMINI_2_FLASH = 8

class AIConfig(BaseModel):
    provider: Provider
    model: str
    canThink: bool
    api_key: str

class AIParams(BaseModel):
    model: str
    messages: List[Dict[str, str]]

class AIParamsWithStream(AIParams):
    stream: bool = False

class OLLAMAParams(AIParamsWithStream):
    pass

class OpenAIParams(AIParamsWithStream):
    pass

class OpenAIParamsT(AIParamsWithStream):
    temperature: Optional[float] = None # 0.0 - 1.0, doesn't supported by o3 series

class AnthropicParams(AIParams):
    max_tokens: int = 1024
    system: str

class GoogleParams(BaseModel):
    model: str
    system_instruction: str
    max_output_tokens: int = 400
    top_k: int = 2
    top_p: float = 0.5
    temperature: float = 0.5
    response_mime_type: str = 'application/json'
    stop_sequences: List[str] = ['\n']
    seed: int = 42
    # safety_settings: List[Dict[str, int]] = [{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 3}]

class AI_Processing_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI_Streaming_Error(Exception):
    """Custom exception for AI-related errors."""
    pass

class AI:    
    # Define AI_CONFIG at the class level instead of within __init__
    AI_CONFIG = {
        AIModel.CHATGPT: AIConfig(
            provider=Provider.OPENAI,
            model="o3-mini",
            canThink=False,
            api_key=openai_api_key
        ),
        AIModel.CLAUDE: AIConfig(
            provider=Provider.ANTHROPIC,
            model="claude-3-7-sonnet-20250219", 
            canThink=False,
            api_key=anthropic_api_key
        ),
        AIModel.DEEPSEEK_SMALL: AIConfig(
            provider=Provider.OLLAMA,
            model="deepseek-r1:7b",
            canThink=True,
            api_key=ollama_api_key
        ),
        AIModel.DEEPSEEK_LARGE: AIConfig(
            provider=Provider.OLLAMA,
            model="deepseek-r1:32b",
            canThink=True,
            api_key=ollama_api_key
        ),
        AIModel.QWQ: AIConfig(
            provider=Provider.OLLAMA,
            model="qwq:latest",
            canThink=True,
            api_key=ollama_api_key
        ),
        AIModel.PHI: AIConfig(
            provider=Provider.OLLAMA,
            model="phi4:latest",
            canThink=False,
            api_key=ollama_api_key
        ),
        AIModel.GEMMA3_LOCAL: AIConfig(
            provider=Provider.GOOGLE,
            model="gemma3:27b",
            canThink=False,
            api_key=google_api_key,
        ),
        AIModel.GEMINI_2_FLASH: AIConfig(
            provider=Provider.GOOGLE,
            model="gemini-2.5-flash",
            canThink=False,
            api_key=google_api_key
        )
    }
    
    def __init__(self, ai_model, system_prompt):
        # Remove AI_CONFIG dictionary from here since it's now a class variable
        # Get the dictionary from the enum value
        self.config = AI.AI_CONFIG[ai_model]
        self.system_prompt = system_prompt
        self.thoughts = ""
        self.response = ""
        self.context = ""
        self._stream_handlers = {
            Provider.ANTHROPIC: self._stream_anthropic,
            Provider.OPENAI: self._stream_openai,
            # Provider.GOOGLE: self._stream_google,
            Provider.OLLAMA: self._stream_ollama
        }
        self._request_handlers = {
            Provider.ANTHROPIC: self._request_anthropic,
            Provider.OPENAI: self._request_openai,
            # Provider.GOOGLE: self._request_google,
            Provider.OLLAMA: self._request_ollama
        }
        # Initialize the appropriate client based on the provider
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the client based on the provider."""
        match self.config.provider:
            case Provider.OPENAI:
                self.client = OpenAI()
            case Provider.ANTHROPIC:
                self.client = Anthropic(api_key=self.config.api_key)
            case Provider.GOOGLE:
                self.client = genai.Client(api_key=self.config.api_key)
            case Provider.OLLAMA:
                self.client = ollama
            case _:
                raise AI_Processing_Error(f"Unsupported provider: {self.config.provider}")
        logger.info(f"Successfully initialized client for provider {self.config.provider}")

    def _get_prompt(self, user_prompt):
        """Format the prompt by including context and user query."""
        prompt = ""
        if self.context:
            prompt += "This is the previous discussion (not the user's question). Read it carefully and answer the user's question, which will start after latest <<<Q>>> tag. \n\n"
            prompt += self.context 
            prompt += "\n\nNow, here IS the user's question: <<<Q>>> \n\n"
        prompt += user_prompt
        return prompt

    def _get_params(self, user_prompt, stream=False):
        """Create provider-specific parameters for the request.
        Args:
            user_prompt (str): The user's input prompt
            stream (bool, optional): Whether to stream the response. Defaults to False.
        Returns:
            dict: Provider-specific parameters for the API request
        """
        # Create message structure
        user_message = [{"role": "user", "content": self._get_prompt(user_prompt)}]
        system_message = [{"role": "system", "content": self.system_prompt}]

        # Define provider configurations
        PROVIDER_CONFIGS = {
            Provider.ANTHROPIC: {
                'param_class': AnthropicParams,
                'messages': user_message,
            },
            Provider.OPENAI: {
                'param_class': (OpenAIParamsT 
                            if self.config.model not in ["o3-mini", "o3-preview"]
                            else OpenAIParams),
                'messages': system_message + user_message,
                'stream': stream,
            },
            Provider.OLLAMA: {
                'param_class': OLLAMAParams,
                'messages': system_message + user_message,
                'stream': stream,
            },
            Provider.GOOGLE: {
                'param_class': GoogleParams,
                'messages': user_message,
                'stream': stream,
            }
        }

        # Get provider-specific configuration
        provider_config = PROVIDER_CONFIGS[self.config.provider]

        # Build parameters dictionary
        params = {
            "model": self.config.model,
            "messages": provider_config['messages'],
        }

        # Add optional parameters if they exist
        optional_params = ['stream', 'system', 'temperature', 'system_instruction', 'max_output_tokens', 'top_k', 'top_p', 'seed']  
        params.update({
            param: provider_config[param] for param in optional_params if param in provider_config and provider_config[param] is not None
        })

        return params

    def _save_response(self, response, user_prompt):
        """Process and save the AI response."""
        if self.config.canThink == True:
            self.thoughts = Parser.extract_text(response, "<think>", "</think>")
            self.response = Parser.extract_text(response,"</think>", )
        else:
            self.response = response
        self.context += user_prompt + "\n\nThe response: \n\n" + response

    def _request_anthropic(self, params):
        """Handle requests for Anthropic."""
        response = self.client.messages.create(**params)
        return response.content[0].text

    def _request_openai(self, params):
        """Handle requests for OpenAI."""
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content

    def _request_ollama(self, params):
        """Handle requests for Ollama."""
        response = self.client.chat(**params)
        return response.get("message", {}).get("content", "")

    def _stream_anthropic(self, params):
        """Handle streaming for Anthropic."""
        response = ""
        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                response += text                
        return response

    def _stream_openai(self, params):
        """Handle streaming for OpenAI."""
        response = ""
        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text is not None:
                print(text, end="", flush=True)
                response += text
        return response

    def _stream_ollama(self, params):
        """Handle streaming for Ollama."""
        response = ""
        for chunk in self.client.chat(**params):
            text = chunk.get('message', {}).get('content', '')
            if text:
                print(text, end="", flush=True)
                response += text
        return response    

    def stream(self, task, user_prompt):
        """Stream the AI response back."""
        logger.info(f"{self.config.model}'s stream about \"{task}\"")        
        params = self._get_params(user_prompt, stream=True)
        try:
            response = self._stream_handlers[self.config.provider](params)
            self._save_response(response, user_prompt)
        except Exception as e:
            logger.error(f"Error during streaming: {str(e)}")
            raise AI_Streaming_Error(f"Error during streaming: {str(e)}")

    def request(self, task, user_prompt):
        """Make a non-streaming request to the AI model."""
        logger.info(f"Requesting '{self.config.model}' model about '{task}'")
        params = self._get_params(user_prompt, stream=False)
        try:
            response = self._request_handlers[self.config.provider](params)
            self._save_response(response, user_prompt)
        except Exception as e:
            logger.error(f"Error during request: {str(e)}")
            raise AI_Processing_Error(f"Error during request: {str(e)}")


    @property
    def results(self):
        """Property that returns the response for compatibility with existing code."""
        return self.response




# In[10]:

ai = AI(AIModel.CHATGPT, "you are an expert in Solidity. Answer user question with all your knowledge in smart contracts development.")
ai.stream("tell story", "Tell any story for 50 words")


# In[11]:

# print(ai.response + "\n\n----------------\n\n" + ai.thoughts)
ai.stream("summarize story", "summarize the story in 12 words")


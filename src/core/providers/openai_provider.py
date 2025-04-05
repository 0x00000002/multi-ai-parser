"""
OpenAI provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, Tuple, BinaryIO
from ..interfaces import ProviderInterface, MultimediaProviderInterface, ToolCapableProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.tool_call import ToolCall
from .base_provider import BaseProvider
import openai


class OpenAIProvider(BaseProvider, MultimediaProviderInterface, ToolCapableProviderInterface):
    """Provider implementation for OpenAI models with multimedia capabilities."""
    
    # Role mapping for OpenAI API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
    }
    
    def __init__(self, 
                 model_id: str,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_id: The model identifier
            logger: Logger instance
        """
        super().__init__(model_id, logger)
        
        # Initialize model parameters
        self.parameters = self.model_config.get("parameters", {})
        
        self.logger.info(f"Initialized OpenAI provider with model {model_id}")
    
    def _initialize_credentials(self) -> None:
        """Initialize OpenAI API credentials."""
        try:
            # Get API key from configuration
            api_key = self.provider_config.get("api_key")
            if not api_key:
                api_key = self.config.get_api_key("openai")
                
            # Set up OpenAI client
            self.client = openai.OpenAI(api_key=api_key)
            
            # Get base URL if specified (for Azure, etc.)
            base_url = self.provider_config.get("base_url")
            if base_url:
                self.client.base_url = base_url
                
            # Get organization if specified
            org_id = self.provider_config.get("organization")
            if org_id:
                self.client.organization = org_id
                
        except Exception as e:
            raise AICredentialsError(f"Failed to initialize OpenAI credentials: {str(e)}")
    
    def _map_role(self, role: str) -> str:
        """Map standard role to OpenAI role."""
        return self._ROLE_MAP.get(role, "user")
        
    def request(self, messages: List[Dict[str, str]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the OpenAI API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Response content as a string, or a dictionary with content and tool calls
        """
        try:
            # Format messages for OpenAI API
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-OpenAI parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "frequency_penalty", 
                            "presence_penalty", "stop", "tools", "tool_choice", "response_format"]:
                    del params[key]
            
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=formatted_messages,
                **params
            )
            
            # Extract tool calls if present
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                            id=tool_call.id
                        )
                    )
            
            # If tool calls are present, return a dictionary with content and tool calls
            if tool_calls:
                return {
                    "content": response.choices[0].message.content or "",
                    "tool_calls": tool_calls
                }
            
            # Otherwise, return just the content
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making OpenAI request: {str(e)}")
    
    def request_with_tools(self, 
                          prompt: str, 
                          tools: List[Dict[str, Any]],
                          conversation: Optional[List[Tuple[str, str]]] = None,
                          system_prompt: Optional[str] = None,
                          structured_tools: bool = True) -> Dict[str, Any]:
        """
        Make a request that can use tools.
        
        Args:
            prompt: User prompt
            tools: List of tool definitions
            conversation: Optional conversation history as (role, content) tuples
            system_prompt: Optional system prompt
            structured_tools: Whether to use structured tools format
            
        Returns:
            Dictionary with content and tool calls
        """
        try:
            # Build messages
            messages = []
            
            # Add system message
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history
            if conversation:
                for role, content in conversation:
                    messages.append({"role": role, "content": content})
                    
            # Add current prompt
            messages.append({"role": "user", "content": prompt})
            
            # Format tools for OpenAI API
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                }
                formatted_tools.append(formatted_tool)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=self._format_messages(messages),
                tools=formatted_tools,
                tool_choice="auto"
            )
            
            # Process response
            content = response.choices[0].message.content or ""
            
            # Extract tool calls
            tool_calls = []
            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                            id=tool_call.id
                        )
                    )
            
            return {
                "content": content,
                "tool_calls": tool_calls
            }
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error making OpenAI tools request: {str(e)}")
    
    def stream(self, messages: List[Dict[str, str]], **options) -> str:
        """
        Stream a response from the OpenAI API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Format messages for OpenAI API
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = self.parameters.copy()
            params.update(options)
            
            # Remove any non-OpenAI parameters
            for key in list(params.keys()):
                if key not in ["temperature", "max_tokens", "top_p", "frequency_penalty", 
                            "presence_penalty", "stop", "tools", "tool_choice", "response_format"]:
                    del params[key]
            
            # Make the streaming API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=formatted_messages,
                stream=True,
                **params
            )
            
            # Collect the chunks
            chunks = []
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                    
            # Join all chunks
            return "".join(chunks)
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in streaming: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error streaming from OpenAI: {str(e)}")
    
    def analyze_image(self, image_data: Union[str, BinaryIO], prompt: str) -> str:
        """
        Analyze an image with the model.
        
        Args:
            image_data: Image data (file path, URL, or file-like object)
            prompt: Prompt describing what to analyze in the image
            
        Returns:
            Analysis as a string
        """
        try:
            # Create message with image
            if isinstance(image_data, str) and (image_data.startswith("http") or image_data.startswith("https")):
                # Handle URL
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": image_data}
                }
            else:
                # Handle base64 or file-like object
                if isinstance(image_data, str):
                    # Assume it's a file path
                    with open(image_data, "rb") as f:
                        # Encode to base64
                        import base64
                        base64_image = base64.b64encode(f.read()).decode("utf-8")
                else:
                    # File-like object
                    import base64
                    base64_image = base64.b64encode(image_data.read()).decode("utf-8")
                    
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
                
            # Create messages with image
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    image_message
                ]}
            ]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages
            )
            
            return response.choices[0].message.content or ""
            
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in image analysis: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error analyzing image with OpenAI: {str(e)}")
    
    def generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard") -> str:
        """
        Generate an image with DALL-E.
        
        Args:
            prompt: Prompt describing the image to generate
            size: Image size (1024x1024, 512x512, or 256x256)
            quality: Image quality (standard or hd)
            
        Returns:
            URL of the generated image
        """
        try:
            # Check if model is DALL-E
            if "dall-e" not in self.model_id.lower():
                self.logger.warning(f"Attempting to generate image with non-DALL-E model: {self.model_id}")
            
            # Make API call
            response = self.client.images.generate(
                model=self.model_id,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1
            )
            
            # Return image URL
            if response.data and response.data[0].url:
                return response.data[0].url
            else:
                raise AIProviderError("No image URL returned from OpenAI")
                
        except openai.APIError as e:
            raise AIRequestError(f"OpenAI API error in image generation: {str(e)}")
        except Exception as e:
            raise AIProviderError(f"Error generating image with OpenAI: {str(e)}")
    
    def transcribe_audio(self, audio_data: Union[str, BinaryIO]) -> str:
        """
        Transcribe audio using the OpenAI API.
        
        Args:
            audio_data: Audio file path or file-like object
            
        Returns:
            Transcription as a string
        """
        try:
            # Make the transcription request
            if isinstance(audio_data, str):
                # Assume it's a file path
                with open(audio_data, "rb") as f:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f
                    )
            else:
                # Handle file-like object
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_data
                )
            
            return transcript.text
            
        except Exception as e:
            self.logger.error(f"OpenAI transcription failed: {str(e)}")
            raise AIRequestError(f"OpenAI transcription failed: {str(e)}")
    
    def text_to_speech(self, 
                      text: str, 
                      **options) -> Union[bytes, str]:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: Text to convert to speech
            options: Additional options including:
                - voice: Voice to use (default: alloy)
                - model: TTS model to use (default: tts-1)
                - output_path: Path to save audio file (optional)
                - format: Audio format (default: mp3)
            
        Returns:
            Audio data as bytes or path to saved audio file
        """
        try:
            # Set default options
            voice = options.get("voice", "alloy")
            model = options.get("model", "tts-1")
            output_path = options.get("output_path", None)
            
            # Generate speech
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text
            )
            
            # Save to file if output path is provided
            if output_path:
                response.stream_to_file(output_path)
                self.logger.info(f"Speech saved to file: {output_path}")
                return output_path
            
            # Otherwise return audio data
            audio_data = response.content
            self.logger.info("Speech generation successful")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"OpenAI text-to-speech failed: {str(e)}")
            raise AIRequestError(
                f"Failed to generate speech: {str(e)}",
                provider="openai",
                original_error=e
            )
    
    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content),
            "tool_call_id": next((tc.get("id") for msg in reversed(messages) 
                                 if msg.get("role") == "assistant" 
                                 and msg.get("tool_calls") 
                                 for tc in msg["tool_calls"] 
                                 if tc.get("name") == name), "unknown")
        })
        return messages 
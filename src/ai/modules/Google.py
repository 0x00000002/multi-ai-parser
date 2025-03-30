from src.ai.errors import AI_Processing_Error, AI_Streaming_Error, AI_API_Key_Error
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import src.ai.ai_config as config
from src.logger import Logger
from src.ai.tools.models import ToolCallRequest, ToolCall

from google import genai

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def _get_params(self, messages, optional_params: Dict[str, Any] = {}) -> dict:
        """Get the parameters for the request."""
        # Construct params dict with only non-None values
        params = {
            "model": self.model.model_id,
            "contents": messages,
            **{k: v for k, v in optional_params.items() if k in GeminiParams.model_fields and v is not None}
        }
        
        # Create and validate with Pydantic model directly
        validated_params = GeminiParams(**params)
        
        # Convert to dict, excluding None values
        return validated_params.model_dump(exclude_none=True)

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

    def add_tool_message(self, messages: List[Dict[str, str]], name: str, content: str) -> List[Dict[str, str]]:
        """
        Add a tool message to the conversation history in Gemini format.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            List[Dict[str, str]]: Updated conversation history
        """
        messages.append({
            "role": "model",
            "parts": [{"text": str(content)}]
        })
        return messages

    def request(self, user_prompt: str, optional_params: Dict[str, Any] = {}) -> str:
        """Make a non-streaming request to the AI model."""
        params = self._get_params(user_prompt, optional_params)
        
        try:
            # Generate the content
            res = self.client.models.generate_content(**params)
            
            # Convert Gemini response to standardized ToolCallRequest
            tool_calls = []
            candidate = res.candidates[0]
            if hasattr(candidate, 'tool_calls') and candidate.tool_calls:
                for tool_call in candidate.tool_calls:
                    tool_calls.append(ToolCall(
                        name=tool_call.name,
                        arguments=tool_call.arguments
                    ))
            
            return ToolCallRequest(
                tool_calls=tool_calls,
                content=candidate.content.parts[0].text,
                finish_reason=candidate.finish_reason
            )
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



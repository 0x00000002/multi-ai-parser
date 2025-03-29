from pydantic import BaseModel
from typing import Dict, List
from src.ai.tools.models import Function, Parameters
from src.ai.tools.tools_list import Tool

class Formatable(BaseModel):
    @classmethod
    def get_tools(cls, tools: List[Tool]) -> List[Dict]:
        return [cls.from_base(tool.value[0]).model_dump() for tool in tools]
    
    @classmethod
    def from_base(cls, base_function: Function):
        raise NotImplementedError("Subclasses must implement from_base()")

class OpenAI_Function(Formatable):
    type: str = "function"
    function: Function
    
    @classmethod
    def from_base(cls, base_function: Function):
        return cls(function=base_function)

class Ollama_Function(OpenAI_Function):
    pass

class Anthropic_Function(Formatable):
    name: str
    description: str
    input_schema: Parameters

    @classmethod
    def from_base(cls, base_function: Function):
        return cls(
            name=base_function.name, 
            # The description is sent to LLM, 
            # so it's good to provide an example of the request
            # which should trigger the tool call
            description=base_function.description, 
            input_schema=base_function.parameters
        )

class Gemini_Function_Declaration(BaseModel):
    name: str
    description: str
    parameters: Parameters
    response: Dict = {"type": "object", "properties": {}}

class Google_Function(Formatable):
    function_declarations: List[Gemini_Function_Declaration]
    
    @classmethod
    def from_base(cls, base_function: Function):
        return cls(
            function_declarations=[
                Gemini_Function_Declaration(
                    name=base_function.name,
                    description=base_function.description,
                    parameters=base_function.parameters
                )
            ]
        )
    
    @classmethod
    def get_tools(cls, tools: List[Tool]) -> List[Dict]:
        # For Google/Gemini, we still return a list
        # But with only one item containing all declarations
        if not tools:
            return []
        
        # Create one container with all function declarations
        all_declarations = []
        for tool in tools:
            all_declarations.append(
                Gemini_Function_Declaration(
                    name=tool.value[0].name,
                    description=tool.value[0].description,
                    parameters=tool.value[0].parameters
                ).model_dump()
            )
        
        # Return as a single item in the list
        return [{"function_declarations": all_declarations}]


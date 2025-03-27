from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class Property(BaseModel):
    type: str
    description: str
    enum: Optional[List[Any]] = None    

class Parameters(BaseModel):
    type: str
    properties: Dict[str, Property]
    required: List[str]
    additionalProperties: bool = False

class Function(BaseModel):
    name: str
    description: str
    parameters: Parameters

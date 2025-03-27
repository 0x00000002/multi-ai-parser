from pydantic import BaseModel
from enum import Enum


class Image(BaseModel):
    alt: str
    description: str
    url: str

class Link(BaseModel):
    category: str
    name: str
    url: str

class Format(Enum):
    MARKDOWN = "markdown"
    JSON = "json"
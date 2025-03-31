"""
Prompt version implementation.
Handles versioning for prompt templates to track changes over time.
"""
from typing import Dict, Any, Optional, List
import copy
import uuid
from datetime import datetime


class PromptVersion:
    """
    Handles versioning for prompt templates.
    Tracks changes to templates over time with version history.
    """
    
    def __init__(self, 
                 template_id: str,
                 version_id: Optional[str] = None,
                 version: Optional[int] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 content: Optional[Dict[str, Any]] = None,
                 created_by: Optional[str] = None,
                 is_active: bool = False):
        """
        Initialize a prompt version.
        
        Args:
            template_id: ID of the parent template
            version_id: Unique identifier for this version (auto-generated if None)
            version: Version number (auto-assigned if None)
            name: Human-readable name for this version
            description: Description of this version's changes
            content: The version's content (template, defaults, etc.)
            created_by: User or process that created this version
            is_active: Whether this is the currently active version
        """
        self.template_id = template_id
        self.version_id = version_id or str(uuid.uuid4())
        self.version = version or 1
        self.name = name or f"Version {self.version}"
        self.description = description or ""
        self.content = content or {}
        self.created_by = created_by or "system"
        self.created_at = datetime.utcnow()
        self.is_active = is_active
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the version to a dictionary.
        
        Returns:
            Dictionary representation of the version
        """
        return {
            "template_id": self.template_id,
            "version_id": self.version_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptVersion':
        """
        Create a version from a dictionary.
        
        Args:
            data: Dictionary representation of the version
            
        Returns:
            New version instance
        """
        version = cls(
            template_id=data["template_id"],
            version_id=data.get("version_id"),
            version=data.get("version"),
            name=data.get("name"),
            description=data.get("description"),
            content=data.get("content", {}),
            created_by=data.get("created_by"),
            is_active=data.get("is_active", False)
        )
        
        # Handle dates if present
        if "created_at" in data:
            version.created_at = datetime.fromisoformat(data["created_at"])
            
        return version
    
    @classmethod
    def create_new_version(cls, 
                           template_id: str,
                           previous_version: Optional['PromptVersion'],
                           content: Dict[str, Any],
                           name: Optional[str] = None,
                           description: Optional[str] = None,
                           created_by: Optional[str] = None) -> 'PromptVersion':
        """
        Create a new version based on a previous version.
        
        Args:
            template_id: ID of the parent template
            previous_version: Previous version to base this on (or None if first)
            content: The version's content
            name: Human-readable name for this version
            description: Description of this version's changes
            created_by: User or process that created this version
            
        Returns:
            New version instance
        """
        version_num = 1
        if previous_version:
            version_num = previous_version.version + 1
            
        return cls(
            template_id=template_id,
            version=version_num,
            name=name or f"Version {version_num}",
            description=description,
            content=copy.deepcopy(content),
            created_by=created_by,
            is_active=False  # New versions start as inactive
        ) 
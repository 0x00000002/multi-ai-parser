"""
Prompt template implementation.
Provides a flexible templating system for prompts with variable substitution.
"""
from typing import Dict, Any, Optional, Set
import re
import uuid
from datetime import datetime


class PromptTemplate:
    """
    Template-based prompt with variable substitution.
    Supports Jinja-like syntax for variable interpolation: {{variable_name}}
    """
    
    def __init__(self, 
                 template_id: Optional[str] = None,
                 name: str = "",
                 description: str = "",
                 template: str = "",
                 default_values: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt template.
        
        Args:
            template_id: Unique identifier for the template (auto-generated if None)
            name: Human-readable name for the template
            description: Description of the template's purpose
            template: The template string with variable placeholders
            default_values: Default values for template variables
        """
        self.template_id = template_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.template = template
        self.default_values = default_values or {}
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        
    def get_variables(self) -> Set[str]:
        """
        Extract all variable names from the template.
        
        Returns:
            Set of variable names
        """
        pattern = r'\{\{([^}]+)\}\}'
        return set(re.findall(pattern, self.template))
    
    def render(self, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Render the template with the provided variables.
        
        Args:
            variables: Variables to use for rendering (combined with defaults)
            
        Returns:
            Rendered template string
            
        Raises:
            ValueError: If required variables are missing
        """
        # Combine default values with provided variables
        all_vars = {**self.default_values}
        if variables:
            all_vars.update(variables)
        
        # Get all variables in the template
        required_vars = self.get_variables()
        
        # Check for missing variables
        missing_vars = required_vars - set(all_vars.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {', '.join(missing_vars)}")
        
        # Render the template by replacing placeholders
        result = self.template
        for var_name, var_value in all_vars.items():
            if var_name in required_vars:
                placeholder = f"{{{{{var_name}}}}}"
                result = result.replace(placeholder, str(var_value))
        
        return result
    
    def update(self, 
               name: Optional[str] = None,
               description: Optional[str] = None,
               template: Optional[str] = None,
               default_values: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the template.
        
        Args:
            name: New name (or None to keep current)
            description: New description (or None to keep current)
            template: New template string (or None to keep current)
            default_values: New default values (or None to keep current)
        """
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description
        if template is not None:
            self.template = template
        if default_values is not None:
            self.default_values.update(default_values)
        
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the template to a dictionary.
        
        Returns:
            Dictionary representation of the template
        """
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "default_values": self.default_values,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """
        Create a template from a dictionary.
        
        Args:
            data: Dictionary representation of the template
            
        Returns:
            New template instance
        """
        template = cls(
            template_id=data.get("template_id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            template=data.get("template", ""),
            default_values=data.get("default_values", {})
        )
        
        # Handle dates if present
        if "created_at" in data:
            template.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            template.updated_at = datetime.fromisoformat(data["updated_at"])
            
        return template 
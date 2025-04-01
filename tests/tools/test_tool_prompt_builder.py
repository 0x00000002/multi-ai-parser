import unittest
from unittest.mock import MagicMock

from src.tools.tool_prompt_builder import ToolPromptBuilder
from src.tools.interfaces import ToolStrategy


class TestToolPromptBuilder(unittest.TestCase):
    """Test cases for the ToolPromptBuilder class."""

    def test_build_enhanced_prompt_empty(self):
        """Test building enhanced prompt with no tools."""
        # Call with empty tools list
        original_prompt = "Original prompt"
        enhanced = ToolPromptBuilder.build_enhanced_prompt(original_prompt, [])
        
        # Verify original prompt is returned unchanged
        self.assertEqual(enhanced, original_prompt)

    def test_build_enhanced_prompt_with_tools(self):
        """Test building enhanced prompt with tools."""
        # Create mock tools
        tool1 = MagicMock(spec=ToolStrategy)
        tool1.get_description.return_value = "Tool 1 description"
        tool1.get_schema.return_value = {"param1": "string"}
        
        tool2 = MagicMock(spec=ToolStrategy)
        tool2.get_description.return_value = "Tool 2 description"
        tool2.get_schema.return_value = {"param2": "number"}
        
        # Create tools with names list
        tools_with_names = [
            ("tool1", tool1),
            ("tool2", tool2)
        ]
        
        # Call build_enhanced_prompt
        original_prompt = "Original prompt"
        enhanced = ToolPromptBuilder.build_enhanced_prompt(original_prompt, tools_with_names)
        
        # Verify tools' get_description and get_schema were called
        tool1.get_description.assert_called_once()
        tool1.get_schema.assert_called_once()
        tool2.get_description.assert_called_once()
        tool2.get_schema.assert_called_once()
        
        # Verify enhanced prompt contains original prompt and tool information
        self.assertIn(original_prompt, enhanced)
        self.assertIn("Available tools:", enhanced)
        self.assertIn("- tool1: Tool 1 description", enhanced)
        self.assertIn("Parameters: {'param1': 'string'}", enhanced)
        self.assertIn("- tool2: Tool 2 description", enhanced)
        self.assertIn("Parameters: {'param2': 'number'}", enhanced)


if __name__ == '__main__':
    unittest.main()
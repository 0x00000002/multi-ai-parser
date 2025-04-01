import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.tools.tool_registry import ToolRegistry, DefaultToolStrategy
from src.utils.logger import LoggerInterface
from src.exceptions import AIToolError
from src.tools.models import ToolDefinition
# Create mock Provider enum to avoid dependency on actual Provider enum that seems to be using from_string
class MockProvider:
    OPENAI_GPT_4O = "openai_gpt_4o"
    OPENAI_GPT_4O_MINI = "openai_gpt_4o_mini"
    OPENAI_GPT_4_TURBO = "openai_gpt_4_turbo"
    ANTHROPIC_CLAUDE_3_5_SONNET = "anthropic_claude_3_5_sonnet"
    ANTHROPIC_CLAUDE_3_OPUS = "anthropic_claude_3_opus"
    ANTHROPIC_CLAUDE_3_HAIKU = "anthropic_claude_3_haiku"
    GOOGLE_GEMINI_1_5_PRO = "google_gemini_1_5_pro"
    GOOGLE_GEMINI_1_5_FLASH = "google_gemini_1_5_flash"
    GOOGLE_GEMINI_2_5_PRO = "google_gemini_2_5_pro"


class TestToolRegistry(unittest.TestCase):
    """Test cases for the ToolRegistry class."""

    def setUp(self):
        """Set up the test case with mocks."""
        # Mock dependencies
        self.mock_logger = MagicMock(spec=LoggerInterface)
        
        # Create the registry
        self.registry = ToolRegistry(logger=self.mock_logger)
        
        # Sample tool definition
        self.test_tool_function = MagicMock(return_value="test result")
        self.test_description = "Test tool description"
        self.test_schema = {"type": "object", "properties": {"param1": {"type": "string"}}}

    def test_register_tool(self):
        """Test registering a tool in the registry."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Verify the tool was registered
        self.assertTrue(self.registry.has_tool("test_tool"))
        
        # Verify the tool implementation
        tool = self.registry.get_tool("test_tool")
        self.assertIsNotNone(tool)
        self.assertIsInstance(tool, DefaultToolStrategy)
        
        # Verify the tool metadata
        self.assertEqual(self.registry.get_tool_description("test_tool"), self.test_description)
        self.assertEqual(self.registry.get_tool_schema("test_tool"), self.test_schema)

    def test_register_duplicate_tool(self):
        """Test registering a duplicate tool raises an exception."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Attempt to register a duplicate tool
        with self.assertRaises(AIToolError):
            self.registry.register_tool(
                tool_name="test_tool",
                tool_function=self.test_tool_function,
                description="Another description",
                parameters_schema=self.test_schema
            )

    def test_register_custom_strategy(self):
        """Test registering a tool with a custom strategy."""
        # Create a custom strategy
        custom_strategy = MagicMock(spec=DefaultToolStrategy)
        custom_strategy.get_description.return_value = "Custom description"
        custom_strategy.get_schema.return_value = {"custom": "schema"}
        
        # Register a tool with the custom strategy
        self.registry.register_tool(
            tool_name="custom_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema,
            tool_strategy=custom_strategy
        )
        
        # Verify the custom strategy was used
        self.assertEqual(self.registry.get_tool("custom_tool"), custom_strategy)
        self.assertEqual(self.registry.get_tool_description("custom_tool"), "Custom description")
        self.assertEqual(self.registry.get_tool_schema("custom_tool"), {"custom": "schema"})

    def test_get_nonexistent_tool(self):
        """Test getting a nonexistent tool returns None."""
        # Verify None is returned for nonexistent tools
        self.assertIsNone(self.registry.get_tool("nonexistent"))
        self.assertIsNone(self.registry.get_tool_description("nonexistent"))
        self.assertIsNone(self.registry.get_tool_schema("nonexistent"))

    def test_get_all_tools(self):
        """Test getting all registered tools."""
        # Register multiple tools
        self.registry.register_tool(
            tool_name="tool1",
            tool_function=self.test_tool_function,
            description="Tool 1",
            parameters_schema=self.test_schema
        )
        self.registry.register_tool(
            tool_name="tool2",
            tool_function=self.test_tool_function,
            description="Tool 2",
            parameters_schema=self.test_schema
        )
        
        # Get all tools
        all_tools = self.registry.get_all_tools()
        all_definitions = self.registry.get_all_tool_definitions()
        
        # Verify all tools were returned
        self.assertEqual(len(all_tools), 2)
        self.assertIn("tool1", all_tools)
        self.assertIn("tool2", all_tools)
        
        # Verify all definitions were returned
        self.assertEqual(len(all_definitions), 2)
        self.assertIn("tool1", all_definitions)
        self.assertIn("tool2", all_definitions)

    def test_format_tools_for_openai(self):
        """Test formatting tools for OpenAI."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Mock the Provider enum and its from_string method
        with patch('src.tools.tool_registry.Provider') as mock_provider:
            # Setup mock provider enum
            mock_provider.OPENAI_GPT_4_TURBO = MockProvider.OPENAI_GPT_4_TURBO
            mock_provider.OPENAI_GPT_4O = MockProvider.OPENAI_GPT_4O
            mock_provider.OPENAI_GPT_4O_MINI = MockProvider.OPENAI_GPT_4O_MINI
            
            # Mock the from_string method to return a valid provider
            mock_provider.from_string = MagicMock(return_value=mock_provider.OPENAI_GPT_4_TURBO)
            
            # Format for OpenAI
            formatted = self.registry.format_tools_for_provider("OPENAI_GPT_4_TURBO", {"test_tool"})
            
            # Verify the format
            self.assertEqual(len(formatted), 1)
            self.assertEqual(formatted[0]["type"], "function")
            self.assertEqual(formatted[0]["function"]["name"], "test_tool")
            self.assertEqual(formatted[0]["function"]["description"], self.test_description)
            self.assertEqual(formatted[0]["function"]["parameters"], self.test_schema)

    def test_format_tools_for_anthropic(self):
        """Test formatting tools for Anthropic."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Mock the Provider enum and its from_string method
        with patch('src.tools.tool_registry.Provider') as mock_provider:
            # Setup mock provider enum
            mock_provider.ANTHROPIC_CLAUDE_3_OPUS = MockProvider.ANTHROPIC_CLAUDE_3_OPUS
            mock_provider.ANTHROPIC_CLAUDE_3_5_SONNET = MockProvider.ANTHROPIC_CLAUDE_3_5_SONNET
            mock_provider.ANTHROPIC_CLAUDE_3_HAIKU = MockProvider.ANTHROPIC_CLAUDE_3_HAIKU
            
            # Mock the from_string method to return a valid provider
            mock_provider.from_string = MagicMock(return_value=mock_provider.ANTHROPIC_CLAUDE_3_OPUS)
            
            # Format for Anthropic
            formatted = self.registry.format_tools_for_provider("ANTHROPIC_CLAUDE_3_OPUS", {"test_tool"})
            
            # Verify the format
            self.assertEqual(len(formatted), 1)
            self.assertEqual(formatted[0]["name"], "test_tool")
            self.assertEqual(formatted[0]["description"], self.test_description)
            self.assertEqual(formatted[0]["input_schema"], self.test_schema)

    def test_format_tools_for_gemini(self):
        """Test formatting tools for Google Gemini."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Mock the Provider enum and its from_string method
        with patch('src.tools.tool_registry.Provider') as mock_provider:
            # Setup mock provider enum
            mock_provider.GOOGLE_GEMINI_1_5_PRO = MockProvider.GOOGLE_GEMINI_1_5_PRO
            mock_provider.GOOGLE_GEMINI_1_5_FLASH = MockProvider.GOOGLE_GEMINI_1_5_FLASH
            mock_provider.GOOGLE_GEMINI_2_5_PRO = MockProvider.GOOGLE_GEMINI_2_5_PRO
            
            # Mock the from_string method to return a valid provider
            mock_provider.from_string = MagicMock(return_value=mock_provider.GOOGLE_GEMINI_1_5_PRO)
            
            # Format for Gemini
            formatted = self.registry.format_tools_for_provider("GOOGLE_GEMINI_1_5_PRO", {"test_tool"})
            
            # Verify the format
            self.assertEqual(len(formatted), 1)
            self.assertEqual(formatted[0]["function_declaration"]["name"], "test_tool")
            self.assertEqual(formatted[0]["function_declaration"]["description"], self.test_description)
            self.assertEqual(formatted[0]["function_declaration"]["parameters"], self.test_schema)

    def test_format_tools_unknown_provider(self):
        """Test formatting tools for an unknown provider returns an empty list."""
        # Register a tool
        self.registry.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Since Provider.from_string doesn't exist, we need to patch differently
        with patch('src.tools.tool_registry.Provider') as mock_provider_class:
            # Setup the from_string method to return None for unknown provider
            mock_provider_class.from_string = MagicMock(return_value=None)
            
            # Format for unknown provider
            formatted = self.registry.format_tools_for_provider("UNKNOWN", {"test_tool"})
            
            # Verify empty list is returned
            self.assertEqual(formatted, [])

    def test_format_tools_all(self):
        """Test formatting all tools."""
        # Register multiple tools
        self.registry.register_tool(
            tool_name="tool1",
            tool_function=self.test_tool_function,
            description="Tool 1",
            parameters_schema=self.test_schema
        )
        self.registry.register_tool(
            tool_name="tool2",
            tool_function=self.test_tool_function,
            description="Tool 2",
            parameters_schema=self.test_schema
        )
        
        # Mock the Provider enum and its from_string method
        with patch('src.tools.tool_registry.Provider') as mock_provider:
            # Setup mock provider enum
            mock_provider.OPENAI_GPT_4_TURBO = MockProvider.OPENAI_GPT_4_TURBO
            mock_provider.OPENAI_GPT_4O = MockProvider.OPENAI_GPT_4O
            mock_provider.OPENAI_GPT_4O_MINI = MockProvider.OPENAI_GPT_4O_MINI
            
            # Mock the from_string method to return a valid provider
            mock_provider.from_string = MagicMock(return_value=mock_provider.OPENAI_GPT_4_TURBO)
            
            # Format all tools
            formatted = self.registry.format_tools_for_provider("OPENAI_GPT_4_TURBO", None)
            
            # Verify all tools were formatted
            self.assertEqual(len(formatted), 2)


if __name__ == '__main__':
    unittest.main()
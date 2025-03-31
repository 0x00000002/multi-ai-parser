import unittest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, List, Set

from src.tools.ai_tool_finder import AIToolFinder
from src.tools.models import ToolDefinition
from src.utils.logger import LoggerInterface
from src.config.config_manager import ConfigManager, ModelConfig
from src.exceptions import AIToolError


class TestAIToolFinder(unittest.TestCase):
    """Test cases for the AIToolFinder class."""

    def setUp(self):
        """Set up the test case with mocks."""
        # Mock dependencies
        self.mock_logger = MagicMock(spec=LoggerInterface)
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        
        # Set up model configuration
        self.mock_model_config = MagicMock(spec=ModelConfig)
        self.mock_model_config.model_id = "test-model"
        self.mock_model_config.provider = "test-provider"
        self.mock_config_manager.get_model_config.return_value = self.mock_model_config
        
        # Sample tools for testing - updated with required fields
        self.test_tools = {
            "tool1": ToolDefinition(
                name="tool1", 
                description="Test tool 1", 
                execute_func=MagicMock(),
                requires_confirmation=False,
                category="utility",
                inputs=[],
                version="1.0.0",
                # Add required fields
                parameters_schema={},
                function=MagicMock()
            ),
            "tool2": ToolDefinition(
                name="tool2", 
                description="Test tool 2", 
                execute_func=MagicMock(),
                requires_confirmation=True,
                category="utility",
                inputs=[],
                version="1.0.0",
                # Add required fields
                parameters_schema={},
                function=MagicMock()
            ),
            "search": ToolDefinition(
                name="search", 
                description="Search for information", 
                execute_func=MagicMock(),
                requires_confirmation=False,
                category="search",
                inputs=[],
                version="1.0.0",
                # Add required fields
                parameters_schema={},
                function=MagicMock()
            )
        }

    def tearDown(self):
        """Clean up patches."""
        pass

    @patch('src.core.provider_factory.ProviderFactory')
    def test_init(self, mock_provider_factory):
        # Setup the mock
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        # Test implementation
        finder = AIToolFinder("test-model", self.mock_config_manager, self.mock_logger)
        
        # Assertions
        self.mock_config_manager.get_model_config.assert_called_with("test-model")
        mock_provider_factory.create.assert_called_once_with(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        self.assertEqual(finder._available_tools, {})

    @patch('src.core.provider_factory.ProviderFactory')
    def test_init_error(self, mock_provider_factory):
        """Test error handling during initialization."""
        # Mock the provider factory to raise an exception
        mock_provider_factory.create.side_effect = Exception("Provider creation failed")
        
        with self.assertRaises(AIToolError):
            AIToolFinder(
                model_id="test-model",
                config_manager=self.mock_config_manager,
                logger=self.mock_logger
            )

    @patch('src.core.provider_factory.ProviderFactory')
    def test_set_available_tools(self, mock_provider_factory):
        """Test set_available_tools method."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        finder.set_available_tools(self.test_tools)
        
        self.assertEqual(finder._available_tools, self.test_tools)

    @patch('src.core.provider_factory.ProviderFactory')
    def test_format_tools_for_prompt(self, mock_provider_factory):
        """Test _format_tools_for_prompt method."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        # Test with no tools
        empty_result = finder._format_tools_for_prompt()
        self.assertEqual(empty_result, "No tools available.")
        
        # Test with tools
        finder.set_available_tools(self.test_tools)
        result = finder._format_tools_for_prompt()
        
        # Verify all tools are in the formatted string
        self.assertIn("AVAILABLE TOOLS:", result)
        self.assertIn("tool1: Test tool 1", result)
        self.assertIn("tool2: Test tool 2", result)
        self.assertIn("search: Search for information", result)

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_no_tools_available(self, mock_provider_factory):
        """Test find_tools with no available tools."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        result = finder.find_tools("Find something")
        
        self.assertEqual(result, set())

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_basic(self, mock_provider_factory):
        """Test find_tools with basic request."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response
        mock_provider.request.return_value = json.dumps({"tools": ["tool1", "search"]})
        
        result = finder.find_tools("Find something")
        
        # Verify correct tools were selected
        self.assertEqual(result, {"tool1", "search"})
        
        # Verify AI was called
        mock_provider.request.assert_called_once()

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_with_conversation_history(self, mock_provider_factory):
        """Test find_tools with conversation history."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response
        mock_provider.request.return_value = json.dumps({"tools": ["tool2"]})
        
        # Test with conversation history
        history = ["User: What can you do?", "Assistant: I can help with various tasks."]
        result = finder.find_tools("Help me with task X", history)
        
        # Verify correct tools were selected
        self.assertEqual(result, {"tool2"})
        
        # Verify AI was called with history
        call_args = mock_provider.request.call_args[0][0]
        self.assertIn("CONVERSATION HISTORY:", call_args)
        self.assertIn("User: What can you do?", call_args)

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_with_markdown_response(self, mock_provider_factory):
        """Test find_tools with markdown formatted response."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response with markdown code fences
        mock_provider.request.return_value = "```json\n{\"tools\": [\"tool1\", \"search\"]}\n```"
        
        result = finder.find_tools("Find something")
        
        # Verify correct tools were selected
        self.assertEqual(result, {"tool1", "search"})

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_with_invalid_tool(self, mock_provider_factory):
        """Test find_tools with invalid tool name in response."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response with invalid tool
        mock_provider.request.return_value = json.dumps({"tools": ["tool1", "nonexistent"]})
        
        result = finder.find_tools("Find something")
        
        # Verify only valid tool was selected
        self.assertEqual(result, {"tool1"})

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_with_json_error(self, mock_provider_factory):
        """Test find_tools with JSON parsing error."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response with invalid JSON
        mock_provider.request.return_value = "Not valid JSON"
        
        result = finder.find_tools("Find something")
        
        # Verify empty set is returned on JSON error
        self.assertEqual(result, set())

    @patch('src.core.provider_factory.ProviderFactory')
    def test_find_tools_with_ai_error(self, mock_provider_factory):
        """Test find_tools with AI request error."""
        mock_provider = MagicMock()
        mock_provider_factory.create.return_value = mock_provider
        
        finder = AIToolFinder(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        finder.set_available_tools(self.test_tools)
        
        # Mock AI response with exception
        mock_provider.request.side_effect = Exception("AI request failed")
        
        with self.assertRaises(AIToolError):
            finder.find_tools("Find something")


if __name__ == '__main__':
    unittest.main() 
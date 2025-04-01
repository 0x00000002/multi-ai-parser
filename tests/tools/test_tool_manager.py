import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Set, Any

from src.tools.tool_manager import ToolManager
from src.tools.tool_registry import ToolRegistry
from src.tools.tool_executor import ToolExecutor
from src.tools.ai_tool_finder import AIToolFinder
from src.tools.models import ToolResult, ToolDefinition
from src.utils.logger import LoggerInterface
from src.config.config_manager import ConfigManager
from src.exceptions import AIToolError


class TestToolManager(unittest.TestCase):
    """Test cases for the ToolManager class."""

    def setUp(self):
        """Set up the test case with mocks."""
        # Mock dependencies
        self.mock_logger = MagicMock(spec=LoggerInterface)
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.mock_registry = MagicMock(spec=ToolRegistry)
        self.mock_finder = MagicMock(spec=AIToolFinder)
        self.mock_executor = MagicMock(spec=ToolExecutor)
        
        # Mock config_manager.get_default_model_id
        self.mock_config_manager.get_default_model_id = MagicMock(return_value="default-model")
        
        # Create the manager
        self.manager = ToolManager(
            logger=self.mock_logger,
            config_manager=self.mock_config_manager,
            tool_registry=self.mock_registry,
            tool_finder=self.mock_finder,
            tool_executor=self.mock_executor
        )
        
        # Sample tool data
        self.test_tool_function = MagicMock(return_value="test result")
        self.test_description = "Test tool description"
        self.test_schema = {"type": "object", "properties": {"param1": {"type": "string"}}}

    def test_init_with_dependencies(self):
        """Test initialization with provided dependencies."""
        # Verify dependencies were set
        self.assertEqual(self.manager._logger, self.mock_logger)
        self.assertEqual(self.manager._config_manager, self.mock_config_manager)
        self.assertEqual(self.manager._tool_registry, self.mock_registry)
        self.assertEqual(self.manager._tool_executor, self.mock_executor)
        self.assertEqual(self.manager._tool_finder, self.mock_finder)
        
        # Verify tool finder was initialized
        self.mock_finder.set_available_tools.assert_called_once_with(
            self.mock_registry.get_all_tool_definitions.return_value
        )

    @patch('src.tools.tool_manager.ToolRegistry')
    @patch('src.tools.tool_manager.ToolExecutor')
    @patch('src.tools.tool_manager.LoggerFactory')
    @patch('src.tools.tool_manager.ConfigManager')
    def test_init_default_dependencies(self, mock_config_manager_class, mock_logger_factory, mock_executor_class, mock_registry_class):
        """Test initialization with default dependencies."""
        # Mock return values
        mock_logger = MagicMock(spec=LoggerInterface)
        mock_logger_factory.create.return_value = mock_logger
        mock_registry = MagicMock(spec=ToolRegistry)
        mock_registry_class.return_value = mock_registry
        mock_executor = MagicMock(spec=ToolExecutor)
        mock_executor_class.return_value = mock_executor
        mock_config_manager = MagicMock(spec=ConfigManager)
        mock_config_manager_class.return_value = mock_config_manager
        
        # Create manager with default dependencies
        manager = ToolManager()
        
        # Verify default dependencies were created
        self.assertEqual(manager._logger, mock_logger)
        mock_registry_class.assert_called_once_with(mock_logger)
        mock_executor_class.assert_called_once_with(mock_logger)

    def test_register_tool(self):
        """Test registering a tool through the manager."""
        # Call register_tool
        self.manager.register_tool(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Verify registry.register_tool was called
        self.mock_registry.register_tool.assert_called_once_with(
            tool_name="test_tool",
            tool_function=self.test_tool_function,
            description=self.test_description,
            parameters_schema=self.test_schema
        )
        
        # Verify tool finder was updated
        self.mock_finder.set_available_tools.assert_called_with(
            self.mock_registry.get_all_tool_definitions.return_value
        )

    def test_enable_auto_tool_finding(self):
        """Test enabling auto tool finding."""
        # Test enabling
        self.manager.enable_auto_tool_finding(True)
        
        # Verify setting was updated
        self.assertTrue(self.manager.auto_find_tools)
        
        # Test disabling
        self.manager.enable_auto_tool_finding(False)
        
        # Verify setting was updated
        self.assertFalse(self.manager.auto_find_tools)

    @patch('src.tools.tool_manager.AIToolFinder')
    def test_enable_auto_tool_finding_creates_finder(self, mock_finder_class):
        """Test enabling auto tool finding creates a finder if not provided."""
        # Create manager without finder
        manager = ToolManager(
            logger=self.mock_logger,
            config_manager=self.mock_config_manager,
            tool_registry=self.mock_registry,
            tool_executor=self.mock_executor
        )
        
        # Mock finder
        mock_finder = MagicMock(spec=AIToolFinder)
        mock_finder_class.return_value = mock_finder
        
        # Mock get_default_model_id method
        self.mock_config_manager.get_default_model_id = MagicMock(return_value="default-model")
        
        # Enable auto tool finding
        manager.enable_auto_tool_finding(True, "test-model")
        
        # Verify finder was created
        mock_finder_class.assert_called_once_with(
            model_id="test-model",
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        # Verify finder was set up
        mock_finder.set_available_tools.assert_called_once_with(
            self.mock_registry.get_all_tool_definitions.return_value
        )
        
        # Verify setting was updated
        self.assertTrue(manager.auto_find_tools)

    def test_find_tools(self):
        """Test finding tools for a prompt."""
        # Setup mock tools
        expected_tools = {"tool1", "tool2"}
        self.mock_finder.find_tools.return_value = expected_tools
        
        # Call find_tools
        conversation = ["Message 1", "Message 2"]
        result = self.manager.find_tools("test prompt", conversation)
        
        # Verify finder.find_tools was called
        self.mock_finder.find_tools.assert_called_once_with("test prompt", conversation)
        
        # Verify result
        self.assertEqual(result, expected_tools)

    def test_find_tools_without_finder(self):
        """Test finding tools raises error if no finder is configured."""
        # Create manager without finder
        manager = ToolManager(
            logger=self.mock_logger,
            config_manager=self.mock_config_manager,
            tool_registry=self.mock_registry,
            tool_executor=self.mock_executor
        )
        
        # Call find_tools, expect error
        with self.assertRaises(AIToolError):
            manager.find_tools("test prompt")

    def test_execute_tool(self):
        """Test executing a tool."""
        # Mock tool
        mock_tool = MagicMock()
        self.mock_registry.get_tool.return_value = mock_tool
        
        # Mock tool result
        expected_result = ToolResult(
            success=True,
            result="tool result",
            tool_name="test_tool"
        )
        self.mock_executor.execute.return_value = expected_result
        
        # Call execute_tool
        result = self.manager.execute_tool("test_tool", param1="value1")
        
        # Verify registry.get_tool was called
        self.mock_registry.get_tool.assert_called_once_with("test_tool")
        
        # Verify executor.execute was called
        self.mock_executor.execute.assert_called_once_with(
            "test_tool", mock_tool, param1="value1"
        )
        
        # Verify result
        self.assertEqual(result, expected_result)

    def test_execute_tool_not_found(self):
        """Test executing a nonexistent tool returns error result."""
        # Mock tool not found
        self.mock_registry.get_tool.return_value = None
        
        # Call execute_tool
        result = self.manager.execute_tool("nonexistent", param1="value1")
        
        # Verify result is failure
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertEqual(result.tool_name, "nonexistent")
        self.assertIn("Tool not found", result.message)

    def test_enhance_prompt(self):
        """Test enhancing a prompt with tool information."""
        # Mock tools
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()
        self.mock_registry.get_tool.side_effect = lambda name: {
            "tool1": mock_tool1,
            "tool2": mock_tool2
        }.get(name)
        
        # Mock ToolPromptBuilder
        enhanced_prompt = "Enhanced prompt with tools"
        with patch('src.tools.tool_manager.ToolPromptBuilder') as mock_builder:
            mock_builder.build_enhanced_prompt.return_value = enhanced_prompt
            
            # Call enhance_prompt
            result = self.manager.enhance_prompt("test prompt", {"tool1", "tool2"})
            
            # Verify registry.get_tool was called for each tool
            self.mock_registry.get_tool.assert_any_call("tool1")
            self.mock_registry.get_tool.assert_any_call("tool2")
            
            # Verify builder.build_enhanced_prompt was called, but don't verify the exact order of tools
            # since sets don't guarantee ordering
            mock_builder.build_enhanced_prompt.assert_called_once()
            call_args = mock_builder.build_enhanced_prompt.call_args[0]
            
            # Verify the prompt argument
            self.assertEqual(call_args[0], "test prompt")
            
            # Verify the tools list contains the expected items (regardless of order)
            tool_list = call_args[1]
            self.assertEqual(len(tool_list), 2)
            tool_dict = dict(tool_list)
            self.assertIn("tool1", tool_dict)
            self.assertEqual(tool_dict["tool1"], mock_tool1)
            self.assertIn("tool2", tool_dict)
            self.assertEqual(tool_dict["tool2"], mock_tool2)
            
            # Verify result
            self.assertEqual(result, enhanced_prompt)

    def test_enhance_prompt_no_valid_tools(self):
        """Test enhancing a prompt with no valid tools returns original prompt."""
        # Mock tools not found
        self.mock_registry.get_tool.return_value = None
        
        # Call enhance_prompt
        original_prompt = "Original prompt"
        result = self.manager.enhance_prompt(original_prompt, {"nonexistent"})
        
        # Verify registry.get_tool was called
        self.mock_registry.get_tool.assert_called_once_with("nonexistent")
        
        # Verify original prompt was returned
        self.assertEqual(result, original_prompt)


if __name__ == '__main__':
    unittest.main()
import unittest
from unittest.mock import MagicMock

from src.tools.tool_executor import ToolExecutor
from src.tools.interfaces import ToolStrategy
from src.utils.logger import LoggerInterface
from src.tools.models import ToolResult


class TestToolExecutor(unittest.TestCase):
    """Test cases for the ToolExecutor class."""

    def setUp(self):
        """Set up the test case with mocks."""
        # Mock dependencies
        self.mock_logger = MagicMock(spec=LoggerInterface)
        
        # Create the executor
        self.executor = ToolExecutor(logger=self.mock_logger)
        
        # Mock tool
        self.mock_tool = MagicMock(spec=ToolStrategy)

    def test_execute_successful(self):
        """Test successful tool execution."""
        # Set up the mock tool to return a success result
        self.mock_tool.execute.return_value = "successful result"
        
        # Execute the tool
        result = self.executor.execute("test_tool", self.mock_tool, param1="value1")
        
        # Verify the tool was called with the correct parameters
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        
        # Verify the result
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.result, "successful result")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertIsNone(result.error)

    def test_execute_error(self):
        """Test tool execution with an error."""
        # Set up the mock tool to raise an exception
        self.mock_tool.execute.side_effect = ValueError("Tool execution failed")
        
        # Execute the tool
        result = self.executor.execute("test_tool", self.mock_tool, param1="value1")
        
        # Verify the tool was called with the correct parameters
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        
        # Verify the result contains the error
        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, "Tool execution failed")
        self.assertEqual(result.tool_name, "test_tool")

    def test_execute_with_complex_args(self):
        """Test tool execution with complex arguments."""
        # Set up the mock tool to return a success result
        complex_result = {"key": "value", "nested": {"data": [1, 2, 3]}}
        self.mock_tool.execute.return_value = complex_result
        
        # Complex arguments
        complex_args = {
            "string_arg": "string value",
            "int_arg": 42,
            "list_arg": [1, 2, 3],
            "dict_arg": {"key": "value"}
        }
        
        # Execute the tool with complex arguments
        result = self.executor.execute("complex_tool", self.mock_tool, **complex_args)
        
        # Verify the tool was called with the complex arguments
        self.mock_tool.execute.assert_called_once_with(**complex_args)
        
        # Verify the result contains the complex return value
        self.assertIsInstance(result, ToolResult)
        self.assertTrue(result.success)
        self.assertEqual(result.result, complex_result)
        self.assertEqual(result.tool_name, "complex_tool")


if __name__ == '__main__':
    unittest.main()
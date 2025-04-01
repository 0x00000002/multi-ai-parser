import unittest
from unittest.mock import MagicMock
from typing import Dict, Any, Optional

from src.tools.models import ToolDefinition, ToolCall, ToolCallRequest, ToolResult


class TestToolModels(unittest.TestCase):
    """Test cases for tool models."""

    def test_tool_definition(self):
        """Test ToolDefinition model."""
        # Create test function
        test_function = MagicMock(return_value="result")
        
        # Create ToolDefinition
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool description",
            parameters_schema={"type": "object", "properties": {"param": {"type": "string"}}},
            function=test_function,
            metadata={"category": "test"}
        )
        
        # Verify attributes
        self.assertEqual(tool_def.name, "test_tool")
        self.assertEqual(tool_def.description, "Test tool description")
        self.assertEqual(tool_def.parameters_schema, {"type": "object", "properties": {"param": {"type": "string"}}})
        self.assertEqual(tool_def.function, test_function)
        self.assertEqual(tool_def.metadata, {"category": "test"})
        
        # Test without optional parameters
        tool_def_minimal = ToolDefinition(
            name="minimal_tool",
            description="Minimal tool",
            parameters_schema={},
            function=test_function
        )
        
        # Verify attributes
        self.assertEqual(tool_def_minimal.name, "minimal_tool")
        self.assertIsNone(tool_def_minimal.metadata)

    def test_tool_call(self):
        """Test ToolCall model."""
        # Create ToolCall
        tool_call = ToolCall(
            name="test_tool",
            arguments={"param": "value"}
        )
        
        # Verify attributes
        self.assertEqual(tool_call.name, "test_tool")
        self.assertEqual(tool_call.arguments, {"param": "value"})

    def test_tool_call_request(self):
        """Test ToolCallRequest model."""
        # Create ToolCalls
        tool_call1 = ToolCall(name="tool1", arguments={"param1": "value1"})
        tool_call2 = ToolCall(name="tool2", arguments={"param2": "value2"})
        
        # Create ToolCallRequest
        request = ToolCallRequest(
            tool_calls=[tool_call1, tool_call2],
            content="Tool call content",
            finish_reason="tool_call",
            metadata={"session_id": "12345"}
        )
        
        # Verify attributes
        self.assertEqual(len(request.tool_calls), 2)
        self.assertEqual(request.tool_calls[0].name, "tool1")
        self.assertEqual(request.tool_calls[1].name, "tool2")
        self.assertEqual(request.content, "Tool call content")
        self.assertEqual(request.finish_reason, "tool_call")
        self.assertEqual(request.metadata, {"session_id": "12345"})
        
        # Test without optional parameters
        request_minimal = ToolCallRequest(
            tool_calls=[tool_call1],
            content="Minimal content"
        )
        
        # Verify attributes
        self.assertEqual(len(request_minimal.tool_calls), 1)
        self.assertEqual(request_minimal.content, "Minimal content")
        self.assertIsNone(request_minimal.finish_reason)
        self.assertIsNone(request_minimal.metadata)

    def test_tool_result_success(self):
        """Test ToolResult model with success case."""
        # Create successful ToolResult
        result = ToolResult(
            success=True,
            result="Operation successful",
            tool_name="test_tool",
            message="Completed successfully",
            metadata={"duration_ms": 150}
        )
        
        # Verify attributes
        self.assertTrue(result.success)
        self.assertEqual(result.result, "Operation successful")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.message, "Completed successfully")
        self.assertEqual(result.metadata, {"duration_ms": 150})
        self.assertIsNone(result.error)

    def test_tool_result_failure(self):
        """Test ToolResult model with failure case."""
        # Create failed ToolResult
        result = ToolResult(
            success=False,
            error="Operation failed due to invalid input",
            tool_name="test_tool",
            message="Tool execution failed",
            metadata={"attempts": 2}
        )
        
        # Verify attributes
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Operation failed due to invalid input")
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.message, "Tool execution failed")
        self.assertEqual(result.metadata, {"attempts": 2})
        self.assertIsNone(result.result)

    def test_tool_result_minimal(self):
        """Test ToolResult model with minimal parameters."""
        # Create minimal ToolResult
        result = ToolResult(success=True)
        
        # Verify attributes
        self.assertTrue(result.success)
        self.assertIsNone(result.result)
        self.assertIsNone(result.error)
        self.assertIsNone(result.tool_name)
        self.assertIsNone(result.message)
        self.assertIsNone(result.metadata)


if __name__ == '__main__':
    unittest.main()
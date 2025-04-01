import unittest
import sys
import os

# Add parent directory to sys.path so that tests can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test modules
from test_tool_registry import TestToolRegistry
from test_tool_executor import TestToolExecutor
from test_tool_manager import TestToolManager
from test_tool_prompt_builder import TestToolPromptBuilder
from test_models import TestToolModels

# Create a test suite
def create_test_suite():
    """Create a test suite with all tool tests."""
    test_suite = unittest.TestSuite()
    test_loader = unittest.TestLoader()
    
    # Add test cases using TestLoader (instead of deprecated makeSuite)
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestToolRegistry))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestToolExecutor))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestToolManager))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestToolPromptBuilder))
    test_suite.addTest(test_loader.loadTestsFromTestCase(TestToolModels))
    
    return test_suite

if __name__ == '__main__':
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code based on test result
    sys.exit(not result.wasSuccessful())
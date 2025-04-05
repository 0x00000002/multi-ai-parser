#!/usr/bin/env python
"""
Test for CodingAssistantAgent integration with Orchestrator.

This test verifies that:
1. The CodingAssistantAgent is properly registered and used
2. The Orchestrator correctly routes coding-related requests to the agent
3. The agent provides appropriate coding-focused responses
"""
import os
import sys
import unittest
import logging
from pathlib import Path

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_coding_assistant")


class TestCodingAssistant(unittest.TestCase):
    """Test cases for CodingAssistantAgent integration."""
    
    def setUp(self):
        """Set up test environment."""
        from src.config import configure, UseCasePreset
        from src.agents.agent_factory import AgentFactory
        from src.agents.agent_registry import AgentRegistry
        from src.agents.agent_registrar import register_core_agents, register_extension_agents
        from src.core.tool_enabled_ai import AI
        
        # Configure for coding use case
        configure(
            model="claude-3-5-sonnet",
            use_case=UseCasePreset.CODING,
            temperature=0.7,
            show_thinking=True
        )
        
        # Create AI instance
        self.ai_instance = AI(
            model="claude-3-5-sonnet",
            system_prompt="You are an expert programmer. Provide clean, efficient, and well-documented code.",
            logger=logger
        )
        
        # Create and set up agent registry
        self.registry = AgentRegistry()
        register_core_agents(self.registry, logger)
        register_extension_agents(self.registry, logger)
        
        # Create agent factory
        self.agent_factory = AgentFactory(registry=self.registry)
        
        # Create orchestrator
        self.orchestrator = self.agent_factory.create(
            "orchestrator",
            ai_instance=self.ai_instance
        )
        
        # Verify orchestrator creation
        self.assertIsNotNone(self.orchestrator, "Failed to create orchestrator")
        
        # Verify CodingAssistantAgent is registered
        self.assertTrue(
            self.registry.has_agent_type("coding_assistant"),
            "CodingAssistantAgent not registered"
        )
    
    def test_coding_request_routing(self):
        """Test that coding requests are routed to CodingAssistantAgent."""
        # Create a coding-related request
        request = {
            "prompt": "Write a Python function to calculate the Fibonacci sequence",
            "conversation_history": []
        }
        
        # Process the request
        response = self.orchestrator.process_request(request)
        
        # Verify response structure
        self.assertIsNotNone(response, "No response received")
        self.assertIsInstance(response, dict, "Response should be a dictionary")
        
        # Extract content
        content = response.get("content", "")
        self.assertIsInstance(content, str, "Response content should be a string")
        self.assertGreater(len(content), 0, "Response content should not be empty")
        
        # Verify response contains coding-related content
        self.assertIn("def", content, "Response should contain function definition")
        self.assertIn("fibonacci", content.lower(), "Response should mention Fibonacci")
        
        # Verify use case detection
        use_case = self.orchestrator._determine_use_case(request)
        self.assertEqual(
            use_case.name,
            "CODING",
            "Use case should be detected as CODING"
        )
    
    def test_solidity_request_routing(self):
        """Test that Solidity requests are also handled appropriately."""
        # Create a Solidity-related request
        request = {
            "prompt": "Write a simple ERC20 token contract in Solidity",
            "conversation_history": []
        }
        
        # Process the request
        response = self.orchestrator.process_request(request)
        
        # Verify response structure
        self.assertIsNotNone(response, "No response received")
        self.assertIsInstance(response, dict, "Response should be a dictionary")
        
        # Extract content
        content = response.get("content", "")
        self.assertIsInstance(content, str, "Response content should be a string")
        self.assertGreater(len(content), 0, "Response content should not be empty")
        
        # Verify response contains Solidity-specific content
        self.assertIn("contract", content, "Response should contain contract definition")
        self.assertIn("ERC20", content, "Response should mention ERC20")
        
        # Verify use case detection
        use_case = self.orchestrator._determine_use_case(request)
        self.assertEqual(
            use_case.name,
            "SOLIDITY_CODING",
            "Use case should be detected as SOLIDITY_CODING"
        )


if __name__ == "__main__":
    unittest.main() 
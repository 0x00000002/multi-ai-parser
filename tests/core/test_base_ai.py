import unittest
from unittest.mock import MagicMock, patch
import uuid
from typing import Dict, List, Optional, Any

from src.core.base_ai import AIBase, DEFAULT_SYSTEM_PROMPT
from src.config.config_manager import ConfigManager
from src.utils.logger import LoggerInterface
from src.exceptions import AISetupError, AIProcessingError
from src.prompts import PromptManager


class TestAIBase(unittest.TestCase):
    """Test cases for the AIBase class."""

    def setUp(self):
        """Set up the test case with mocks."""
        # Mock dependencies
        self.mock_logger = MagicMock(spec=LoggerInterface)
        self.mock_config_manager = MagicMock(spec=ConfigManager)
        self.mock_provider = MagicMock()
        self.mock_conversation_manager = MagicMock()
        self.mock_prompt_manager = MagicMock(spec=PromptManager)
        
        # Set up model configuration
        self.mock_model_config = MagicMock()
        self.mock_model_config.model_id = "test-model"
        self.mock_model_config.provider = "test-provider"
        self.mock_config_manager.get_model_config.return_value = self.mock_model_config
        self.mock_config_manager.default_model = "default-model"
        self.mock_config_manager.show_thinking = False
        
        # Patch the ProviderFactory
        self.provider_factory_patch = patch('src.core.base_ai.ProviderFactory')
        self.mock_provider_factory = self.provider_factory_patch.start()
        self.mock_provider_factory.create.return_value = self.mock_provider
        
        # Patch the ConversationManager
        self.conversation_manager_patch = patch('src.core.base_ai.ConversationManager')
        self.mock_conversation_manager_class = self.conversation_manager_patch.start()
        self.mock_conversation_manager_class.return_value = self.mock_conversation_manager
        
        # Set up conversations
        self.mock_conversation_manager.get_messages.return_value = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "test prompt"}
        ]

    def tearDown(self):
        """Clean up patches."""
        self.provider_factory_patch.stop()
        self.conversation_manager_patch.stop()

    def test_init_default_model(self):
        """Test initialization with default model."""
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        # Verify correct model was selected
        self.mock_config_manager.get_model_config.assert_called_with("default-model")
        
        # Verify provider was created
        self.mock_provider_factory.create.assert_called_once()
        
        # Verify system prompt was added
        self.mock_conversation_manager.add_message.assert_called_with(
            role="system",
            content=DEFAULT_SYSTEM_PROMPT
        )

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        custom_model = "custom-model"
        ai = AIBase(
            model=custom_model,
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        # Verify correct model was selected
        self.mock_config_manager.get_model_config.assert_called_with(custom_model)

    def test_init_with_system_prompt(self):
        """Test initialization with custom system prompt."""
        custom_prompt = "Custom system prompt"
        ai = AIBase(
            system_prompt=custom_prompt,
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        # Verify system prompt was added
        self.mock_conversation_manager.add_message.assert_called_with(
            role="system",
            content=custom_prompt
        )

    def test_request_with_string_response(self):
        """Test request method with string response."""
        self.mock_provider.request.return_value = "test response"
        
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        response = ai.request("test prompt")
        
        # Verify user message was added
        self.mock_conversation_manager.add_message.assert_any_call(
            role="user", 
            content="test prompt"
        )
        
        # Verify provider was called
        self.mock_provider.request.assert_called_once()
        
        # Verify assistant message was added
        self.mock_conversation_manager.add_message.assert_any_call(
            role="assistant",
            content="test response",
            extract_thoughts=True,
            show_thinking=False
        )
        
        self.assertEqual(response, "test response")

    def test_request_with_dict_response(self):
        """Test request method with dictionary response."""
        self.mock_provider.request.return_value = {
            "content": "test response",
            "tool_calls": []
        }
        
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        response = ai.request("test prompt")
        
        # Verify assistant message was added with correct content
        self.mock_conversation_manager.add_message.assert_any_call(
            role="assistant",
            content="test response",
            extract_thoughts=True,
            show_thinking=False
        )
        
        self.assertEqual(response, "test response")

    def test_request_error(self):
        """Test request method with an error."""
        self.mock_provider.request.side_effect = Exception("Request failed")
        
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        with self.assertRaises(AIProcessingError):
            ai.request("test prompt")

    def test_stream(self):
        """Test stream method."""
        self.mock_provider.stream.return_value = {
            "content": "streamed response",
            "tool_calls": []
        }
        
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        response = ai.stream("test prompt")
        
        # Verify user message was added
        self.mock_conversation_manager.add_message.assert_any_call(
            role="user", 
            content="test prompt"
        )
        
        # Verify provider was called
        self.mock_provider.stream.assert_called_once()
        
        # Verify assistant message was added
        self.mock_conversation_manager.add_message.assert_any_call(
            role="assistant",
            content="streamed response",
            extract_thoughts=True,
            show_thinking=False
        )
        
        self.assertEqual(response, "streamed response")

    def test_reset_conversation(self):
        """Test reset_conversation method."""
        custom_prompt = "Custom system prompt"
        ai = AIBase(
            system_prompt=custom_prompt,
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        ai.reset_conversation()
        
        # Verify conversation was reset
        self.mock_conversation_manager.reset.assert_called_once()
        
        # Verify system prompt was added back
        self.mock_conversation_manager.add_message.assert_any_call(
            role="system",
            content=custom_prompt
        )

    def test_get_conversation(self):
        """Test get_conversation method."""
        expected_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": "test prompt"}
        ]
        self.mock_conversation_manager.get_messages.return_value = expected_messages
        
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        messages = ai.get_conversation()
        
        self.assertEqual(messages, expected_messages)

    def test_request_with_template(self):
        """Test request_with_template method."""
        # Set up mock prompt manager
        rendered_prompt = "Rendered prompt"
        usage_id = "test-usage-id"
        self.mock_prompt_manager.render_prompt.return_value = (rendered_prompt, usage_id)
        
        # Set up AIBase with prompt manager
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        ai._prompt_manager = self.mock_prompt_manager
        
        # Mock the request method
        with patch.object(ai, 'request', return_value="Template response") as mock_request:
            variables = {"var1": "value1"}
            response = ai.request_with_template("test-template", variables, "user123")
            
            # Verify prompt manager was called
            self.mock_prompt_manager.render_prompt.assert_called_with(
                template_id="test-template",
                variables=variables,
                user_id="user123",
                context={"model": "test-model"}
            )
            
            # Verify request was called with rendered prompt
            mock_request.assert_called_with(rendered_prompt)
            
            # Verify metrics were recorded
            self.mock_prompt_manager.record_prompt_performance.assert_called_once()
            
            self.assertEqual(response, "Template response")

    def test_stream_with_template(self):
        """Test stream_with_template method."""
        # Set up mock prompt manager
        rendered_prompt = "Rendered prompt"
        usage_id = "test-usage-id"
        self.mock_prompt_manager.render_prompt.return_value = (rendered_prompt, usage_id)
        
        # Set up AIBase with prompt manager
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        ai._prompt_manager = self.mock_prompt_manager
        
        # Mock the stream method
        with patch.object(ai, 'stream', return_value="Streamed template response") as mock_stream:
            variables = {"var1": "value1"}
            response = ai.stream_with_template("test-template", variables, "user123")
            
            # Verify prompt manager was called
            self.mock_prompt_manager.render_prompt.assert_called_with(
                template_id="test-template",
                variables=variables,
                user_id="user123",
                context={"model": "test-model"}
            )
            
            # Verify stream was called with rendered prompt
            mock_stream.assert_called_with(rendered_prompt)
            
            # Verify metrics were recorded
            self.mock_prompt_manager.record_prompt_performance.assert_called_once()
            
            self.assertEqual(response, "Streamed template response")

    def test_set_system_prompt(self):
        """Test set_system_prompt method."""
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        new_prompt = "New system prompt"
        ai.set_system_prompt(new_prompt)
        
        # Verify system prompt was updated
        self.mock_conversation_manager.set_system_prompt.assert_called_with(new_prompt)
        self.assertEqual(ai._system_prompt, new_prompt)

    def test_set_prompt_manager(self):
        """Test set_prompt_manager method."""
        ai = AIBase(
            config_manager=self.mock_config_manager,
            logger=self.mock_logger
        )
        
        ai.set_prompt_manager(self.mock_prompt_manager)
        
        self.assertEqual(ai._prompt_manager, self.mock_prompt_manager)

    def test_initialization_error(self):
        """Test error handling during initialization."""
        # Mock the provider factory to raise an exception
        self.mock_provider_factory.create.side_effect = Exception("Provider creation failed")
        
        with self.assertRaises(AISetupError):
            AIBase(
                config_manager=self.mock_config_manager,
                logger=self.mock_logger
            )


if __name__ == '__main__':
    unittest.main() 
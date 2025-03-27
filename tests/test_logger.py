import pytest
from src.Parser import Parser
from src.Logger import Logger, LoggingLevel, LogFormat, NullLogger
import logging
import os
from unittest.mock import patch, Mock

class TestLogger:
    @pytest.fixture
    def temp_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        return str(log_file)

    def test_logger_initialization(self):
        logger = Logger("TestLogger", level=LoggingLevel.INFO)
        assert isinstance(logger._logger, logging.Logger)
        assert logger.level == LoggingLevel.INFO

    def test_logger_levels(self):
        logger = Logger("TestLogger", level=LoggingLevel.INFO)
        assert logger.level == LoggingLevel.INFO
        
        logger = Logger("TestLogger", level=LoggingLevel.DEBUG)
        assert logger.level == LoggingLevel.DEBUG

    def test_null_logger(self):
        logger = NullLogger()
        # These should not raise any exceptions
        logger.info("Test message")
        logger.debug("Test message")
        logger.warning("Test message")
        logger.error("Test message")
        logger.critical("Test message")

    def test_logger_formatting(self):
        logger = Logger("TestLogger", level=LoggingLevel.INFO, format=LogFormat.SIMPLE)
        assert logger.format == LogFormat.SIMPLE
        
        logger = Logger("TestLogger", level=LoggingLevel.INFO, format=LogFormat.VERBOSE)
        assert logger.format == LogFormat.VERBOSE
        
        logger = Logger("TestLogger", level=LoggingLevel.INFO, format=LogFormat.JSON)
        assert logger.format == LogFormat.JSON

    @patch('logging.getLogger')
    def test_logger_exception_handling(self, mock_get_logger):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_logger.handlers = []
        
        # Simulate file permission error
        mock_logger.addHandler.side_effect = PermissionError()
        
        # Create logger - it should handle the PermissionError gracefully
        try:
            logger = Logger("TestLogger", level=LoggingLevel.INFO)
        except PermissionError:
            pytest.fail("Logger should handle PermissionError gracefully")
        
        # Verify logger still works by calling methods
        logger.info("Test message")
        logger.error("Test error")
        logger.warning("Test warning")
        
        # Verify that the logger attempted to add handler but continued after error
        mock_logger.addHandler.assert_called_once()
        assert isinstance(logger, Logger) 
"""
Logger implementation for the AI framework.
"""
import logging
from typing import Protocol, Optional, Dict, List, Union
from typing_extensions import runtime_checkable
from enum import Enum


@runtime_checkable
class LoggerInterface(Protocol):
    """Interface for logging implementations."""
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        ...
    
    def info(self, message: str) -> None:
        """Log an info message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...
    
    def error(self, message: str) -> None:
        """Log an error message."""
        ...
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        ...


class LoggingLevel(Enum):
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogFormat(Enum):
    SIMPLE = '%(levelname)s - %(name)s - %(message)s'
    VERBOSE = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    JSON = '{"time": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'


class LoggerFactory:
    """
    Factory class to create and manage logger instances.
    """
    _instances: Dict[str, 'Logger'] = {}

    @classmethod
    def create(cls, name: str = None, level: LoggingLevel = LoggingLevel.INFO, 
              format: LogFormat = LogFormat.SIMPLE) -> LoggerInterface:
        """
        Create a new logger instance.
        
        Args:
            name: Logger name (optional)
            level: Logging level
            format: Log format
            
        Returns:
            Logger instance
        """
        name = name or "ai_framework"
        if name not in cls._instances:
            cls._instances[name] = Logger(name, level, format)
        return cls._instances[name]
    
    @classmethod
    def reset(cls) -> None:
        """Reset all loggers (useful for testing)."""
        cls._instances.clear()
        

class NullLogger(LoggerInterface):
    """Logger that silently discards all messages."""
    
    def info(self, message: str) -> None:
        pass
        
    def debug(self, message: str) -> None:
        pass
        
    def warning(self, message: str) -> None:
        pass
        
    def error(self, message: str) -> None:
        pass
        
    def critical(self, message: str) -> None:
        pass


class Logger(LoggerInterface):
    """
    Custom logger class that wraps the standard logging module.
    """
    def __init__(
        self, 
        name: str, 
        level: LoggingLevel = LoggingLevel.INFO, 
        format: LogFormat = LogFormat.SIMPLE,
        handlers: List[logging.Handler] = None
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            level: Logging level
            format: Log format
            handlers: Optional list of handlers
        """
        self.name = name
        self.level = level
        self.format = format
        
        # Get a logger instance
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        
        # Disable propagation to avoid duplicate logs
        self._logger.propagate = False
        
        # Clear existing handlers to avoid duplication
        if self._logger.handlers:
            self._logger.handlers.clear()
            
        # Add handlers
        if handlers:
            for handler in handlers:
                self._configure_handler(handler)
        else:
            # Default to console handler if none provided
            handler = logging.StreamHandler()
            self._configure_handler(handler)
            
    def _configure_handler(self, handler: logging.Handler) -> None:
        """Configure the handler with the correct level and format."""
        try:
            handler.setLevel(self.level.value)
            handler.setFormatter(logging.Formatter(self.format.value))
            self._logger.addHandler(handler)
        except PermissionError:
            # If we can't add the handler due to permissions, continue with NullLogger behavior
            self._logger = NullLogger()
            
    def add_file_handler(self, filename: str) -> None:
        """Add a file handler to the logger."""
        handler = logging.FileHandler(filename)
        self._configure_handler(handler)
    
    def info(self, message: str) -> None:
        self._logger.info(message)
        
    def debug(self, message: str) -> None:
        self._logger.debug(message)
        
    def warning(self, message: str) -> None:
        self._logger.warning(message)
        
    def error(self, message: str) -> None:
        self._logger.error(message)
        
    def critical(self, message: str) -> None:
        self._logger.critical(message) 
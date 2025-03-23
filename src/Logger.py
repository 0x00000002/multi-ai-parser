import logging
from typing import Optional, Dict, List, Union
from enum import Enum


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
    def get_logger(cls, name: str, level: LoggingLevel = LoggingLevel.INFO, 
                 format: LogFormat = LogFormat.SIMPLE) -> 'Logger':
        """
        Get or create a logger instance with the given name.
        """
        if name not in cls._instances:
            cls._instances[name] = Logger(name, level, format)
        return cls._instances[name]
    
    @classmethod
    def reset_loggers(cls) -> None:
        """
        Reset all loggers (useful for testing).
        """
        cls._instances.clear()
        

class NullLogger:
    """Logger that silently discards all messages."""
    def info(self, message: str, *args, **kwargs) -> None:
        pass
        
    def debug(self, message: str, *args, **kwargs) -> None:
        pass
        
    def warning(self, message: str, *args, **kwargs) -> None:
        pass
        
    def error(self, message: str, *args, **kwargs) -> None:
        pass
        
    def critical(self, message: str, *args, **kwargs) -> None:
        pass

class Logger:
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
        """Configure a handler with the appropriate formatter and level."""
        handler.setFormatter(logging.Formatter(self.format.value))
        handler.setLevel(self.level.value)
        self._logger.addHandler(handler)
            
    def add_file_handler(self, filename: str) -> None:
        """Add a file handler to the logger."""
        handler = logging.FileHandler(filename)
        self._configure_handler(handler)
    
    def info(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, **kwargs)
        
    def debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, **kwargs)
        
    def warning(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, **kwargs)
        
    def error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, **kwargs)
        
    def critical(self, message: str, *args, **kwargs) -> None:
        self._logger.critical(message, *args, **kwargs)
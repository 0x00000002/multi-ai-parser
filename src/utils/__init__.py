"""
Utility module containing shared functionality.
"""
from .interfaces import LoggerInterface
from .logger import LoggerFactory, Logger, NullLogger

__all__ = [
    'LoggerInterface',
    'LoggerFactory',
    'Logger',
    'NullLogger'
] 
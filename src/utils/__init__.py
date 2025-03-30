"""
Utility module containing shared functionality.
"""
from .logger import LoggerInterface, LoggerFactory, Logger, NullLogger

__all__ = [
    'LoggerInterface',
    'LoggerFactory',
    'Logger',
    'NullLogger'
] 
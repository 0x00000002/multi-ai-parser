from typing import Protocol
from dataclasses import dataclass

class logger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def info(self, message):
        self.logger.info(message)

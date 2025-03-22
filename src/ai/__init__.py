# Re-export key components from the ai package
from .AI import AI
from .AIConfig import Model, Privacy, Quality, Speed

# This allows users to do:
# from src.ai import AI, Model, Privacy, Quality, Speed
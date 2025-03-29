# Re-export key components from the ai package
from .enhanced_ai import AI
from .ai_config import Model, Privacy, Quality, Speed

# This allows users to do:
# from src.ai import AI, Model, Privacy, Quality, Speed
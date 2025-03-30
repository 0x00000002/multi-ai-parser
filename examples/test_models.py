"""
Simple script to test Model enum with the new approach.
"""
import os
import sys
import importlib
from pathlib import Path

# Add the src directory to the path
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)
print(f"Python path: {sys.path}")

# First, check if models.py exists and what it contains
print("\nChecking models.py initially:")
models_path = os.path.join(src_path, "src", "config", "models.py")
with open(models_path, 'r') as f:
    initial_content = f.read()
    print(initial_content)

# Import Model and see what's available
print("\nInitial Model enum:")
from src.config.models import Model
print(f"Model members: {[member.name for member in Model]}")

# Load config and update models
print("\nLoading ConfigManager to update Model enum:")
from src.config.config_manager import ConfigManager
config_path = os.path.join(src_path, "src", "config", "config.yml")
config_manager = ConfigManager(config_path=config_path)

# Re-import Model to get updated version
print("\nReloading Model module:")
import src.config.models
importlib.reload(src.config.models)
from src.config.models import Model

# Check what's available now
print("\nUpdated Model enum:")
print(f"Model members: {[member.name for member in Model]}")

# Try to access a specific model
if hasattr(Model, "CLAUDE_3_7_SONNET"):
    print(f"Found Claude 3.7 Sonnet: {Model.CLAUDE_3_7_SONNET.value}")
else:
    print("Claude 3.7 Sonnet not found!")

# Now try to use it in the AI example
print("\nTesting in AI example:")
from src.core.tool_enabled_ai import ToolEnabledAI

try:
    ai = ToolEnabledAI(
        model=Model.CLAUDE_3_7_SONNET,
        config_manager=config_manager
    )
    print("Successfully created AI with Model.CLAUDE_3_7_SONNET")
except Exception as e:
    print(f"Error creating AI: {e}") 
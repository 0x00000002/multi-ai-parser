"""
Test script to debug Model enum creation.
"""
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import from there
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)

print(f"Python path: {sys.path}")

# Import config manager
from src.config.config_manager import ConfigManager, Model, create_models

# Get absolute path to the config file
config_path = os.path.join(src_path, "src", "config", "config.yml")
print(f"Config path: {config_path}")

# Check if config file exists
if not os.path.exists(config_path):
    print(f"Config file not found at {config_path}")
    sys.exit(1)

# Read config file directly
with open(config_path, 'r') as f:
    content = f.read()
    print(f"Config file content length: {len(content)} bytes")
    print("First few lines:")
    print("\n".join(content.split("\n")[:10]))

# Create a test enum
import yaml
config_data = yaml.safe_load(open(config_path, 'r'))
print(f"Config data keys: {list(config_data.keys())}")

if 'models' in config_data:
    print(f"Found {len(config_data['models'])} models in config")
    models = list(config_data['models'].keys())
    print(f"Model IDs: {models}")
    
    # Try to create the enum directly
    print("Creating enum manually...")
    
    class_definition = """
class TestModel(str, Enum):
    \"\"\"Test model enum.\"\"\"
"""
    for key in models:
        enum_name = key.upper().replace('-', '_')
        class_definition += f"    {enum_name} = \"{key}\"\n"
    
    print("Class definition:")
    print(class_definition)
    
    # Execute the class definition
    from enum import Enum
    namespace = {}
    exec(class_definition, {"Enum": Enum, "str": str}, namespace)
    
    # Get the new Model class
    TestModel = namespace["TestModel"]
    
    print("Enum members:")
    for member in TestModel:
        print(f" - {member.name} = {member.value}")
    
    # Try to access a member
    if hasattr(TestModel, 'CLAUDE_3_7_SONNET'):
        print("Has CLAUDE_3_7_SONNET:", TestModel.CLAUDE_3_7_SONNET)
    else:
        print("No CLAUDE_3_7_SONNET attribute found!")

# Try to create models directly
print("\nTrying create_models()...")
from src.config.config_manager import create_models
create_models(config_data)

# Print Model enum values
print("Model enum values after create_models():")
for model in Model:
    print(f" - {model.name} = {model.value}")

# Check if the enum has the expected attributes
if hasattr(Model, 'CLAUDE_3_7_SONNET'):
    print("Has CLAUDE_3_7_SONNET:", Model.CLAUDE_3_7_SONNET)
else:
    print("No CLAUDE_3_7_SONNET attribute found in Model enum!")

# Try using the ConfigManager
print("\nTrying ConfigManager...")
config_manager = ConfigManager(config_path=config_path)

# Print Model enum values again after ConfigManager initialization
print("Model enum values after ConfigManager initialization:")
for model in Model:
    print(f" - {model.name} = {model.value}")

# Check if the enum has the expected attributes
if hasattr(Model, 'CLAUDE_3_7_SONNET'):
    print("Has CLAUDE_3_7_SONNET:", Model.CLAUDE_3_7_SONNET)
else:
    print("No CLAUDE_3_7_SONNET attribute found in Model enum!") 
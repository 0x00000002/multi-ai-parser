from src.core.tool_enabled_ai import AI
from src.config.models import Model
from src.config.config_manager import Quality, Speed
from src.core.model_selector import UseCase, ModelSelector
from src.utils.logger import LoggerFactory, LoggingLevel, LogFormat
from src.config.config_manager import ConfigManager

# Example usage with shared model selection
def main():
    
    # Create a logger with DEBUG level - explicitly using real logger
    simple_logger = LoggerFactory.create("logger", LoggingLevel.INFO, LogFormat.SIMPLE, use_real_logger=False)
    debugger = LoggerFactory.create("debugger", LoggingLevel.DEBUG, LogFormat.VERBOSE, use_real_logger=True)
    
    # Example 1: Using AI with use cases
    config_manager = ConfigManager(config_path="src/config/config.yml", logger=simple_logger)
    model_selector = ModelSelector(config_manager)
    model = model_selector.select_model(use_case=UseCase.TRANSLATION, quality=Quality.HIGH, speed=Speed.FAST)
    
    translation_ai = AI(model, logger=simple_logger)
    
    # assert translation_ai.model == Model.CHATGPT_O3_MINI
    result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
    print(result)
    result = translation_ai.request("To Russian?")
    print(result)


if __name__ == "__main__":
    main()

from src.ai.enhanced_ai import AI
from src.ai.ai_config import Quality, Speed, Model, Privacy
from src.ai.model_selector import UseCase
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():
    
    simple_logger = LoggerFactory.get_logger("logger", LoggingLevel.INFO, LogFormat.SIMPLE)
    debugger = LoggerFactory.get_logger("debugger", LoggingLevel.DEBUG, LogFormat.VERBOSE)
    
    
    # Example 1: Using AI with use cases
    simple_logger.info("\n------------------------------- Example 1: -------------------------------")
    translation_ai = AI.for_use_case(use_case=UseCase.TRANSLATION, quality=Quality.HIGH, speed=Speed.FAST, logger=debugger)
    assert translation_ai.model == Model.CHATGPT_O3_MINI
    result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
    print(result)
    result = translation_ai.request("To Russian?")
    print(result)

    # simple_logger.info("\n------------------------------- Example 2: -------------------------------")

    coding_ai = AI.for_use_case(use_case=UseCase.CODING)
    print(coding_ai.model)
    assert coding_ai.model == Model.CHATGPT_O3_MINI
    coding_ai.logger = debugger
    code_result = coding_ai.request("Generate a picture of elephants")
    print(code_result)

if __name__ == "__main__":
    main()

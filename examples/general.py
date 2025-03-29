from src.ai.enhanced_ai import AI
from src.ai.ai_config import Quality, Speed, Model, Privacy
from src.ai.model_selector import UseCase
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():

    # Disable root logger handlers to avoid duplicates
    logging.getLogger().handlers = []
    
    simple_logger = LoggerFactory.get_logger("simple_logger", LoggingLevel.INFO, LogFormat.SIMPLE)
    debugger = LoggerFactory.get_logger("debugger", LoggingLevel.DEBUG, LogFormat.VERBOSE)
    
    
    # Example 1: Using AI with use cases
    simple_logger.info("\n------------------------------- Example 1: -------------------------------")
    translation_ai = AI.for_use_case(use_case=UseCase.TRANSLATION, quality=Quality.MEDIUM, speed=Speed.FAST, logger=debugger)
    assert translation_ai.model == Model.CHATGPT_4O_MINI
    result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
    print(result)
    result = translation_ai.request("To Russian?")
    print(result)

    # Example 2: 
    simple_logger.info("\n------------------------------- Example 2: -------------------------------")
    chatgpt = AI(Model.CLAUDE_SONNET_3_5, system_prompt="You are a helpful assistant that can answer questions and help with tasks.")
    response = chatgpt.request("What's the time difference between Paris and Christchurch?")
    print(response)
    response = chatgpt.request("And with New York?")
    print(response) 

    # Example 3:
    simple_logger.info("\n------------------------------- Example 3: -------------------------------")
    
    ai_by_config = AI({
        'privacy': Privacy.EXTERNAL,
        'quality': Quality.LOW,
        'speed': Speed.FAST
    })
    assert ai_by_config.model == Model.CLAUDE_HAIKU_3_5
    result = ai_by_config.request("How far is Paris from Christchurch?")
    print(result)
    result = ai_by_config.request("And from New York?")
    print(result)

    # Example 4: Create AI instance for code generation
    simple_logger.info("\n------------------------------- Example 4: -------------------------------")
    coding_ai = AI.for_use_case(use_case=UseCase.CODING)
    assert coding_ai.model == Model.CHATGPT_O3_MINI
    coding_ai.logger = simple_logger
    code_result = coding_ai.request("Write a Python function that checks if a string is a palindrome")
    print(code_result)



if __name__ == "__main__":
    main()

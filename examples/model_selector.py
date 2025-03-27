#!/usr/bin/env python
# coding: utf-8

# Change this import:
# from src import AI
# To this:
from src.ai.AI import AI
from src.ai.AIConfig import Quality, Speed, Model, Privacy
from src.ai.ModelSelector import ModelSelector, UseCase
from src.Website import Website
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():
    
    simple_logger = LoggerFactory.get_logger("translator", LoggingLevel.INFO, LogFormat.SIMPLE)
    debugger = LoggerFactory.get_logger("debugger", LoggingLevel.DEBUG, LogFormat.VERBOSE)
    
    
    # # Example 1: Using AI with use cases
    # simple_logger.info("\n------------------------------- Example 1: -------------------------------")
    # translation_ai = AI.for_use_case(use_case=UseCase.TRANSLATION, quality=Quality.HIGH, speed=Speed.FAST, logger=debugger)
    # assert translation_ai.model == Model.GEMINI_2_5_PRO
    # result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
    # print(result)
    # result = translation_ai.request("To Russian?")
    # print(result)

    # simple_logger.info("\n------------------------------- Example 2: -------------------------------")

    coding_ai = AI.for_use_case(use_case=UseCase.CODING)
    print(coding_ai.model)
    assert coding_ai.model == Model.GEMINI_2_5_PRO
    coding_ai.logger = debugger
    code_result = coding_ai.request("Generate a picture of elephants")
    print(code_result)

if __name__ == "__main__":
    main()

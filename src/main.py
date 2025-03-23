#!/usr/bin/env python
# coding: utf-8

# Change this import:
# from src import AI
# To this:
from src.ai.AI import AI
from src.ai.AIConfig import Quality, Speed, Model
from src.ai.ModelSelector import ModelSelector, UseCase
from src.Website import Website
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging  # Add this import

# Example usage with shared model selection
def main():

    # Disable root logger handlers to avoid duplicates
    logging.getLogger().handlers = []
    
    simple_logger = LoggerFactory.get_logger("translator", LoggingLevel.INFO, LogFormat.SIMPLE)
    debugger = LoggerFactory.get_logger("debugger", LoggingLevel.DEBUG, LogFormat.VERBOSE)
    
    
    # Example 1: Using AI with use cases
    simple_logger.info("\n------------------------------- Example 1: -------------------------------")
    translation_ai = AI.for_use_case(use_case=UseCase.TRANSLATION, logger=debugger)
        
    result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
    print(result)
    result = translation_ai.request("To Russian?")
    print(result)

    # Example 2: Create AI instance for code generation
    simple_logger.info("\n------------------------------- Example 2: -------------------------------")
    coding_ai = AI.for_use_case(
        use_case=UseCase.CODING,
        quality=Quality.MEDIUM,
        use_local=True  # Use local models for coding
    )
    coding_ai.logger = simple_logger
    code_result = coding_ai.request("Write a Python function that checks if a string is a palindrome")
    print(code_result)

    # Example 3: 
    simple_logger.info("\n------------------------------- Example 3: -------------------------------")
    chatgpt = AI(Model.CLAUDE_SONNET_3_5, system_prompt="You are a helpful assistant that can answer questions and help with tasks.")
    response = chatgpt.request("What's the time difference between Paris and Christchurch?")
    print(response)
    response = chatgpt.request("And with New York?")
    print(response) 

    # Example 4:
    simple_logger.info("\n------------------------------- Example 4: -------------------------------")
    gemini = AI(Model.GEMINI_1_5_PRO, system_prompt="You are a helpful assistant that can answer questions and help with tasks.")
    response = gemini.request("What's the time difference between Paris and Christchurch?")
    print(response)
    response = gemini.request("And with New York?")
    print(response)

if __name__ == "__main__":
    main()
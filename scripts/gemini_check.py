#!/usr/bin/env python
# coding: utf-8

import sys
import os
import src.ai.AIConfig as config

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ai.Google import Gemini

def main():
    try:
        # Initialize ClaudeAI with a system prompt
        print("Initializing Gemini...")
        ai = Gemini(config.Models.GEMINI_2_FLASH, "You are an expert in Solidity smart contract development.")
        
        # Test the request method with a simple prompt
        print("Sending request to Gemini...")
        response = ai.stream("What's the capital of France?")
        
        # Print the response
        print("\nResponse received successfully:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
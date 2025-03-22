#!/usr/bin/env python
# coding: utf-8

import sys
import os
import src.ai.AIConfig as config

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ai.Ollama import Ollama

def main():
    try:
        # Initialize ClaudeAI with a system prompt
        print("Initializing Ollama...")
        ai = Ollama(config.Models.OLLAMA_GEMMA3, "You are an expert in Solidity smart contract development.")
        
        # Test the request method with a simple prompt
        print("Sending request to Ollama...")
        response = ai.stream("Explain me ERC7201 standard")
        
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
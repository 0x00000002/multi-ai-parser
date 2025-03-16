#!/usr/bin/env python
# coding: utf-8

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ai.ChatGPT import ChatGPT

def main():
    try:
        # Initialize ChatGPT with a system prompt
        print("Initializing ChatGPT...")
        ai = ChatGPT("Answer all questions in a single sentence.")
        
        # Test the request method with a simple prompt
        print("Sending request to ChatGPT...")
        response = ai.stream("Tell me 'Hello'")
        
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
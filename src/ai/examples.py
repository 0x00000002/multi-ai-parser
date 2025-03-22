#!/usr/bin/env python
# coding: utf-8

from src.ai.AI import AI
from src.ai.AIConfig import Quality, Speed, Privacy
from src.Website import Website
from src.ai.ModelSelector import UseCase

# Example 1: Using Website with shared ModelSelector and AIConfig enums
website = Website("https://example.com")

# Translate with different quality levels
website.translateTo("Spanish", quality=Quality.MEDIUM, speed=Speed.STANDARD)
website.translateTo("French", quality=Quality.HIGH, speed=Speed.STANDARD)
website.translateTo("German", quality=Quality.LOW, speed=Speed.FAST, use_local=True)

# Example 2: Using AI with use cases
# Create AI instance for translation
translation_ai = AI.for_use_case(
    use_case=UseCase.TRANSLATION,
    quality=Quality.HIGH,
    speed=Speed.STANDARD
)
result = translation_ai.request("Translate 'Hello, how are you?' to Spanish")
print(result)

# Create AI instance for code generation
coding_ai = AI.for_use_case(
    use_case=UseCase.CODING,
    quality=Quality.MEDIUM,
    use_local=True
)
code_result = coding_ai.request("Write a Python function that checks if a string is a palindrome")
print(code_result)

# Example 3: Using dictionary params with AIConfig enums
ai3 = AI({
    'privacy': Privacy.LOCAL,
    'quality': Quality.MEDIUM,
    'speed': Speed.FAST
})

# Example 4: Chain multiple operations with different models
website = Website("https://news-example.com")
website.translateTo("French", quality=Quality.HIGH)

summarizer = AI.for_use_case(UseCase.SUMMARIZATION, quality=Quality.HIGH)
summary = summarizer.request(f"Summarize this translated content: {website.title_translated}")
print(f"Summary of translated title: {summary}")
#!/usr/bin/env python
# coding: utf-8

class AI_API_Key_Error(Exception):
    """Exception raised when there is an issue with the API key."""
    pass

class AI_Processing_Error(Exception):
    """Exception raised when there is an error during AI processing."""
    pass

class AI_Streaming_Error(Exception):
    """Exception raised when there is an error during AI streaming."""
    pass
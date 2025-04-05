"""
Simple Chat UI for the Agentic AI Framework

This module provides a simple chat interface using Gradio for interacting with the orchestrator.
"""
import gradio as gr
import uuid
from typing import List, Tuple, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_chat_ui")


class SimpleChatUI:
    """
    A simple chat interface for interacting with the orchestrator.
    Uses Gradio to create a web-based chat interface.
    """
    
    def __init__(self, orchestrator, title: str = "Agentic AI Chat"):
        """
        Initialize the chat UI.
        
        Args:
            orchestrator: The orchestrator instance to use for processing requests
            title: The title of the chat interface
        """
        self.orchestrator = orchestrator
        self.title = title
        self.logger = logger
        
    def process_message(self, message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Process a user message and return the updated chat history.
        
        Args:
            message: The user message
            history: The chat history
            
        Returns:
            Updated chat history
        """
        if not message:
            return history
            
        self.logger.info(f"Processing message: {message[:50]}...")
        
        # Create a request for the orchestrator
        request = {
            "prompt": message,
            "request_id": str(uuid.uuid4()),
            "conversation_history": history
        }
        
        # Process the request
        try:
            response = self.orchestrator.process_request(request)
            
            # Extract the content from the response
            if isinstance(response, dict):
                content = response.get("content", str(response))
            else:
                content = str(response)
                
            # Add the response to the history
            history.append((message, content))
            
            self.logger.info("Response processed successfully")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            history.append((message, f"Error: {str(e)}"))
            
        return history
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.
        
        Returns:
            The Gradio interface
        """
        with gr.Blocks(title=self.title) as interface:
            gr.Markdown(f"# {self.title}")
            
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=3
                )
                submit = gr.Button("Send")
                
            clear = gr.Button("Clear Chat")
            
            # Set up event handlers
            submit.click(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                None,
                msg
            )
            
            msg.submit(
                self.process_message,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                None,
                msg
            )
            
            clear.click(lambda: [], None, chatbot)
            
        return interface
    
    def launch(self, **kwargs) -> None:
        """
        Launch the chat interface.
        
        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        interface = self.build_interface()
        interface.launch(**kwargs) 
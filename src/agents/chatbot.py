import openai
from loguru import logger

from src.config import SETTINGS

openai.api_key = SETTINGS.openai_api_key


def process_message(message: str, chat_history: str = None) -> str:
    """
    Process a message and return a response.
    
    Args:
        message: The user's message
        chat_history: Optional conversation history
        
    Returns:
        str: The chatbot's response
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that engages in general conversation."
                },
                {
                    "role": "user", 
                    "content": _build_prompt(message, chat_history)
                }
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
            
    except Exception as e:
        logger.exception(f"Error processing message: {str(e)}")
        return "I apologize, but I encountered an error processing your message."


def _build_prompt(message: str, chat_history: str = None) -> str:
    """Build the conversation prompt for the OpenAI model."""
    parts = []
    
    if chat_history:
        parts.extend([
            "Previous conversation:",
            chat_history,
            "",
        ])
        
    parts.extend([
        "User's message:",
        message
    ])
    
    return "\n".join(parts)

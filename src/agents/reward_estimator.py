import openai
from loguru import logger

from src.config import SETTINGS
from .prompt_cache import PromptCache

openai.api_key = SETTINGS.openai_api_key


def estimate_reward(background: str, chat_messages: str = None, max_credit_per_instance: float = 1.0) -> float:
    """
    Estimate the reward for a provider based on their work quality.
    
    Args:
        background: The background context of the task
        chat_messages: Optional conversation history
        max_credit_per_instance: Maximum possible reward value
        
    Returns:
        float: Estimated reward value between 0 and max_credit_per_instance
    """
    prompt_cache = PromptCache()
    prompt_cache.cleanup_expired()
    
    prompt = _build_prompt(background, chat_messages, max_credit_per_instance)
    
    cached_response = prompt_cache.get(prompt, "gpt-4")
    if cached_response:
        logger.info("Using cached response")
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI that evaluates provider work quality and determines appropriate rewards."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        result = response.choices[0].message.content.strip()
        
        try:
            reward = float(result)
            reward = min(max(0.0, reward), max_credit_per_instance)
            prompt_cache.store(prompt, "gpt-4", str(reward))
            return reward
        except ValueError:
            logger.error(f"Invalid response format from OpenAI: {result}")
            return 0.0
            
    except Exception as e:
        logger.exception(f"Error estimating reward: {str(e)}")
        return 0.0


def _build_prompt(background: str, chat_messages: str = None, max_credit_per_instance: float = 1.0) -> str:
    """Build the evaluation prompt for the OpenAI model."""
    parts = [
        f"Evaluate the quality of work and determine a reward between 0 and {max_credit_per_instance}.",
        "Consider:",
        "- Completeness of the solution",
        "- Code quality and best practices", 
        "- Communication clarity",
        "- Problem-solving approach",
        "",
        "Background context:",
        background,
    ]
    
    if chat_messages:
        parts.extend([
            "",
            "Conversation history:",
            chat_messages
        ])
        
    parts.extend([
        "",
        f"Provide ONLY a single float number between 0 and {max_credit_per_instance}.",
        "Do not include any other text or explanations."
    ])
    
    return "\n".join(parts)

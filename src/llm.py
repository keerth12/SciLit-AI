"""
llm.py
------
Sets up the LangChain LLM object for Groq.

CONCEPT: In LangChain, every LLM provider (Groq, Anthropic, OpenAI,
Gemini...) exposes the SAME interface once wrapped in its LangChain class.
That means chain.py and search_chain.py never need to know we're using
Groq specifically — they just call `.invoke()` on whatever LLM object
they're given. Swapping Groq for Anthropic later means changing this one
file; nothing downstream changes.
"""

from langchain_groq import ChatGroq

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Returns a configured LangChain chat model.

    temperature=0.2 (fairly low) is intentional for a research assistant:
    we want consistent, grounded answers rather than creative variety.
    Turn it up (toward 1.0) if you ever want more exploratory/creative
    responses for a different use case.
    """
    logger.debug(f"Initializing Groq LLM: {settings.groq_model}")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=temperature,
    )
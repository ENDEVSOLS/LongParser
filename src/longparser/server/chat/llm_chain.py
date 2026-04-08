"""LangChain LLM abstraction for LongParser Chat.

Replaces custom llm_router.py with LangChain's provider-specific chat models.
Supports: OpenAI, Gemini, Groq, OpenRouter.
Includes: with_structured_output, with_retry.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .schemas import ChatConfig

logger = logging.getLogger(__name__)


# Default models per provider
DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-5.3",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",
    "openrouter": "openai/gpt-5.3",
}

SUPPORTED_PROVIDERS = list(DEFAULT_MODELS.keys())


def _create_openai(model: str, temperature: float, max_tokens: int,
                   max_retries: int, callbacks: Optional[list] = None):
    """Create OpenAI chat model."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        callbacks=callbacks or [],
    )


def _create_gemini(model: str, temperature: float, max_tokens: int,
                   max_retries: int, callbacks: Optional[list] = None):
    """Create Google Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=max_retries,
        callbacks=callbacks or [],
    )


def _create_groq(model: str, temperature: float, max_tokens: int,
                 max_retries: int, callbacks: Optional[list] = None):
    """Create Groq chat model."""
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        callbacks=callbacks or [],
    )


def _create_openrouter(model: str, temperature: float, max_tokens: int,
                       max_retries: int, callbacks: Optional[list] = None):
    """Create OpenRouter chat model (OpenAI-compatible)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        callbacks=callbacks or [],
    )


_CREATORS = {
    "openai": _create_openai,
    "gemini": _create_gemini,
    "groq": _create_groq,
    "openrouter": _create_openrouter,
}


def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ChatConfig] = None,
    *,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    callbacks: Optional[list] = None,
):
    """Create a LangChain chat model for any supported provider.

    Args:
        provider: LLM provider name (openai, gemini, groq, openrouter).
        model: Model name. If None, uses config or provider default.
        config: ChatConfig for defaults and reliability settings.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        json_mode: If True, wraps with .with_structured_output(LLMAnswer).
        callbacks: Optional LangChain callback handlers.

    Returns:
        A LangChain BaseChatModel (or structured output wrapper).
    """
    config = config or ChatConfig()
    provider = provider or config.llm_provider
    model = model or config.llm_model or DEFAULT_MODELS.get(provider, "gpt-4o")
    max_tokens = max_tokens or config.max_output_tokens

    creator = _CREATORS.get(provider)
    if not creator:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported: {', '.join(_CREATORS)}"
        )

    llm = creator(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=config.llm_max_retries,
        callbacks=callbacks,
    )

    # Structured output: returns Pydantic LLMAnswer directly
    if json_mode:
        from .schemas import LLMAnswer
        llm = llm.with_structured_output(LLMAnswer)

    return llm


def get_plain_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[ChatConfig] = None,
):
    """Get a plain (non-structured) chat model for summarization / plain text tasks."""
    return get_chat_model(
        provider=provider,
        model=model,
        config=config,
        json_mode=False,
    )

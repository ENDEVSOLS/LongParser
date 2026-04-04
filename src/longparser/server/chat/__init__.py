"""LongParser chat engine subpackage.

Provides the full RAG chat stack:

- :class:`~longparser.server.chat.engine.ChatEngine` — end-to-end chat orchestration
- :class:`~longparser.server.chat.retriever.LongParserRetriever` — LangChain retriever
- :class:`~longparser.server.chat.callbacks.LongParserCallbackHandler` — observability
- :func:`~longparser.server.chat.llm_chain.get_chat_model` — multi-provider LLM factory
- :mod:`~longparser.server.chat.graph` — LangGraph Human-in-the-Loop workflow
- :mod:`~longparser.server.chat.schemas` — Pydantic models for chat API
"""

from .engine import ChatEngine
from .retriever import LongParserRetriever
from .callbacks import LongParserCallbackHandler
from .llm_chain import get_chat_model, get_plain_chat_model, DEFAULT_MODELS
from .schemas import (
    ChatConfig,
    ChatRequest,
    ChatResponse,
    LLMAnswer,
    SourceRef,
    Turn,
)

__all__ = [
    "ChatEngine",
    "LongParserRetriever",
    "LongParserCallbackHandler",
    "get_chat_model",
    "get_plain_chat_model",
    "DEFAULT_MODELS",
    "ChatConfig",
    "ChatRequest",
    "ChatResponse",
    "LLMAnswer",
    "SourceRef",
    "Turn",
]

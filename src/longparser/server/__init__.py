"""LongParser REST API server package.

Start the server::

    uv run uvicorn longparser.server.app:app --reload --port 8000

Subpackages:
    - :mod:`~longparser.server.chat` — RAG chat engine, retriever, LangGraph HITL
    - :mod:`~longparser.server.routers` — modular FastAPI route groups (future)

Key modules:
    - :mod:`~longparser.server.app` — FastAPI application factory and all routes
    - :mod:`~longparser.server.db` — Motor async MongoDB database layer
    - :mod:`~longparser.server.queue` — ARQ async job queue (Redis-backed)
    - :mod:`~longparser.server.worker` — ARQ background worker definitions
    - :mod:`~longparser.server.embeddings` — multi-backend embedding engine
    - :mod:`~longparser.server.vectorstores` — Chroma / FAISS / Qdrant adapters
"""

from .app import app

__all__ = ["app"]

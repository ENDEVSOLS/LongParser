"""Optional integration adapters for LangChain and LlamaIndex.

Install the extras to use these adapters::

    pip install clean_rag[langchain]
    pip install clean_rag[llamaindex]
    pip install clean_rag[all]
"""

from __future__ import annotations


def _has_langchain() -> bool:
    """Check if langchain-core is installed."""
    try:
        import langchain_core  # noqa: F401
        return True
    except ImportError:
        return False


def _has_llamaindex() -> bool:
    """Check if llama-index-core is installed."""
    try:
        import llama_index.core  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["_has_langchain", "_has_llamaindex"]

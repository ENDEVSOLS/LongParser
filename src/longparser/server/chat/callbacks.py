"""LangChain callback handler for LongParser Chat observability.

Replaces custom observability middleware with structured logging
at the LLM, retriever, and chain level.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)


class LongParserCallbackHandler(BaseCallbackHandler):
    """Structured logging for all LangChain operations."""

    def __init__(self, tenant_id: str = "", session_id: str = ""):
        super().__init__()
        self.tenant_id = tenant_id
        self.session_id = session_id
        self._llm_start_time: Optional[float] = None

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_start_time = time.monotonic()
        model_name = serialized.get("kwargs", {}).get("model_name", "unknown")
        logger.info(
            "llm_call_start",
            extra={
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "model": model_name,
                "prompt_count": len(prompts),
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        latency_ms = 0.0
        if self._llm_start_time:
            latency_ms = (time.monotonic() - self._llm_start_time) * 1000

        token_usage = {}
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        logger.info(
            "llm_call_end",
            extra={
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            },
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.error(
            "llm_call_error",
            extra={
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "error": str(error),
            },
        )

    def on_retriever_end(
        self,
        documents: list[Document],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        scores = [d.metadata.get("score", 0) for d in documents]
        logger.info(
            "retriever_results",
            extra={
                "tenant_id": self.tenant_id,
                "session_id": self.session_id,
                "doc_count": len(documents),
                "top_score": max(scores) if scores else 0,
                "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            },
        )

"""Embedding engine for LongParser — wraps LangChain providers.

Supported: HuggingFace (local), OpenAI, Gemini.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Global lock for single-flight dimension discovery within a process
_dim_lock = threading.Lock()


class EmbeddingEngine:
    """Multi-provider embedding engine.

    Parameters
    ----------
    provider:
        "huggingface", "openai", or "gemini".
    model_name:
        Model ID (e.g., "BAAI/bge-base-en-v1.5", "text-embedding-3-small").
    dimensions:
        Optional override for embedding dimensions (useful for OpenAI/Gemini
        to avoid API calls or specifically configure text-embedding-3).
    """

    def __init__(
        self,
        provider: str = "huggingface",
        model_name: str = "BAAI/bge-base-en-v1.5",
        dimensions: Optional[int] = None,
    ) -> None:
        self.provider = provider.lower()
        self.model_name = model_name
        self.configured_dimensions = dimensions
        self._dim: Optional[int] = dimensions

        # Provider-specific configurations
        self._gemini_doc_task = "RETRIEVAL_DOCUMENT"
        self._gemini_query_task = "RETRIEVAL_QUERY"
        self._hf_normalize = True

        if self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            # Pass dimensions kwargs if provided (supported for text-embedding-3*)
            kwargs = {}
            if self.configured_dimensions:
                kwargs["dimensions"] = self.configured_dimensions
            self.model = OpenAIEmbeddings(model=self.model_name, **kwargs)
            logger.info(f"EmbeddingEngine: Initialized OpenAI {model_name}")

        elif self.provider == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            kwargs = {}
            if self.configured_dimensions:
                kwargs["output_dimensionality"] = self.configured_dimensions
            self.model = GoogleGenerativeAIEmbeddings(model=self.model_name, **kwargs)
            logger.info(f"EmbeddingEngine: Initialized Gemini {model_name}")

        elif self.provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={"normalize_embeddings": self._hf_normalize}
            )
            logger.info(f"EmbeddingEngine: Initialized HuggingFace {model_name}")

        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def get_fingerprint(self) -> str:
        """Return a stable 10-char SHA1 hash of the full embedding configuration.
        This is used to isolate vector spaces when models/configs change.
        """
        config = {
            "p": self.provider,
            "m": self.model_name,
            "d": self.configured_dimensions,
        }
        if self.provider == "gemini":
            config["t_doc"] = self._gemini_doc_task
            config["t_qry"] = self._gemini_query_task
        elif self.provider == "huggingface":
            config["n"] = self._hf_normalize

        # Stable json dump
        cfg_str = json.dumps(config, sort_keys=True)
        return hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()[:10]

    @property
    def dim(self) -> int:
        """Lazy-evaluated, thread-safe, cross-process cached embedding dimension."""
        if self._dim is not None:
            return self._dim

        # Global dimension lock
        with _dim_lock:
            # Check if another thread resolved it while waiting
            if self._dim is not None:
                return self._dim

            fp = self.get_fingerprint()
            cache_key = f"cleanrag:embed_dim:{fp}"

            # 1) Try Redis cross-process cache if available
            try:
                import redis
                redis_url = os.getenv("LONGPARSER_REDIS_URL", "redis://localhost:6379/0")
                r = redis.from_url(redis_url, socket_connect_timeout=1)
                r.ping()  # Fail fast if unavailable
                cached = r.get(cache_key)
                if cached:
                    self._dim = int(cached)
                    logger.debug(f"Resolved dim from Redis cache: {self._dim}")
                    return self._dim
            except Exception as e:
                logger.debug(f"Redis cache check failed (likely no redis): {e}")

            # 2) Fallback: perform an API call to discover it.
            # We strictly use the Document retrieval task setting to ensure dimension matches index.
            logger.info(f"Discovering embedding dimension for {self.provider}/{self.model_name}...")
            test_doc = ["test"]
            try:
                if self.provider == "gemini":
                    # Explicitly pass the document task type
                    test_vec = self.model.embed_documents(test_doc, task_type=self._gemini_doc_task)[0]
                else:
                    test_vec = self.model.embed_documents(test_doc)[0]
            except Exception as e:
                logger.error(f"Failed to discover embedding dimension: {e}")
                raise  # Fail loudly! Missing API keys should crash here.

            self._dim = len(test_vec)
            logger.info(f"Discovered dimension: {self._dim}")

            # 3) Cache it in Redis
            try:
                if 'r' in locals():
                    r.set(cache_key, self._dim)
            except Exception:
                pass

            return self._dim

    def embed_chunks(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """Embed a list of text chunks.

        Returns list of float vectors (one per chunk).
        """
        if not texts:
            return []

        if self.provider == "gemini":
            # Gemini strictly enforces maximum 100 texts per batch request
            effective_batch = min(batch_size, 100)
            all_embeddings = []
            for i in range(0, len(texts), effective_batch):
                batch = texts[i:i + effective_batch]
                # Pass explicit task_type parameter per call
                all_embeddings.extend(
                    self.model.embed_documents(batch, task_type=self._gemini_doc_task)
                )
            return all_embeddings
        else:
            return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        if self.provider == "gemini":
            # For queries, Gemini optimizes with a different task_type
            return self.model.embed_query(query, task_type=self._gemini_query_task)
        else:
            return self.model.embed_query(query)

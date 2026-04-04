"""LangChain retriever for LongParser Chat.

Wraps existing vector store + embeddings as a LangChain BaseRetriever,
enabling plugging into LCEL chains.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

logger = logging.getLogger(__name__)


class LongParserRetriever(BaseRetriever):
    """LangChain retriever backed by LongParser's existing vector store infra.

    Connects to the same Chroma/FAISS/Qdrant indexes built by the embed pipeline.
    Uses LangChain-native embeddings for query encoding.
    """

    db: Any = Field(exclude=True)
    tenant_id: str
    job_id: str
    top_k: int = 5

    # Resolved at runtime from index_version
    _vector_db: Optional[str] = None
    _model_name: Optional[str] = None
    _provider: Optional[str] = None
    _configured_dimensions: Optional[int] = None
    _collection: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    async def _resolve_index(self) -> None:
        """Load index metadata from MongoDB (lazy, once)."""
        if self._model_name is not None:
            return
        iv_doc = await self.db.get_latest_index_version(self.tenant_id, self.job_id)
        if not iv_doc:
            raise ValueError(f"No embedding index for job {self.job_id}")
        self._vector_db = iv_doc.get("vector_db", "chroma")
        self._model_name = iv_doc["model"]
        self._provider = iv_doc.get("provider", "huggingface")
        self._configured_dimensions = iv_doc.get("configured_dimensions")
        self._collection = iv_doc.get("collection", "longparser")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """Sync retrieval — delegates to existing vector store."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> list[Document]:
        """Async retrieval using existing EmbeddingEngine + vector store."""
        await self._resolve_index()

        from ..embeddings import EmbeddingEngine
        from ..vectorstores import get_vector_store

        # Embed query using same model that built the index
        engine = EmbeddingEngine(
            provider=self._provider,
            model_name=self._model_name,
            dimensions=self._configured_dimensions
        )
        query_embedding = engine.embed_query(query)

        # Search vector DB
        store = get_vector_store(
            self._vector_db,
            collection_name=self._collection,
            index_fingerprint=engine.get_fingerprint(),
        )
        filters = {"tenant_id": self.tenant_id, "job_id": self.job_id}
        results = store.search(query_embedding, top_k=self.top_k, filters=filters)

        # Convert to LangChain Documents
        documents = []
        for r in results:
            meta = r.get("metadata", {})
            documents.append(Document(
                page_content=r.get("document", ""),
                metadata={
                    "chunk_id": meta.get("chunk_id", r.get("id", "")),
                    "score": r.get("score", 0),
                    "chunk_type": meta.get("chunk_type", ""),
                    "page_numbers": meta.get("page_numbers", []),
                    "block_ids": meta.get("block_ids", []),
                },
            ))

        return documents

"""Pluggable vector store adapters for LongParser.

Supports: Chroma, FAISS, Qdrant.
Each adapter follows the BaseVectorStore interface.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract vector store — all adapters implement this interface."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        """Add vectors with metadata."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar vectors. Returns list of {id, score, metadata, document}."""
        ...

    @abstractmethod
    def delete_by_job(self, job_id: str, tenant_id: str = "") -> None:
        """Delete all vectors for a job (idempotent)."""
        ...


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

class ChromaStore(BaseVectorStore):
    """ChromaDB vector store adapter."""

    def __init__(
        self,
        collection_name: str = "longparser",
        persist_directory: str = "./chroma_data",
        index_fingerprint: str = "",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required. Install: pip install clean_rag[chroma]"
            )

        # Securely isolate vector spaces based on model config
        if index_fingerprint:
            collection_name = f"{collection_name}_{index_fingerprint}"
            
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaStore: collection={collection_name}")

    def add(self, ids, embeddings, metadatas, documents) -> None:
        # Chroma metadata must be flat (no nested lists/dicts)
        flat_metas = []
        for m in metadatas:
            flat = {}
            for k, v in m.items():
                if isinstance(v, list):
                    flat[k] = json.dumps(v)
                else:
                    flat[k] = v
            flat_metas.append(flat)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=flat_metas,
            documents=documents,
        )

    def search(self, query_embedding, top_k=5, filters=None) -> list[dict]:
        where = None
        if filters:
            # Build Chroma where clause
            conditions = []
            for k, v in filters.items():
                conditions.append({k: {"$eq": v}})
            if len(conditions) == 1:
                where = conditions[0]
            elif conditions:
                where = {"$and": conditions}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if results["ids"] and results["ids"][0]:
            for i, vid in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # Restore lists from JSON strings
                for k, v in meta.items():
                    if isinstance(v, str) and v.startswith("["):
                        try:
                            meta[k] = json.loads(v)
                        except (json.JSONDecodeError, ValueError):
                            pass
                output.append({
                    "id": vid,
                    "score": 1.0 - (results["distances"][0][i] if results["distances"] else 0),
                    "metadata": meta,
                    "document": results["documents"][0][i] if results["documents"] else "",
                })
        return output

    def delete_by_job(self, job_id: str, tenant_id: str = "") -> None:
        try:
            where = {"job_id": {"$eq": job_id}}
            if tenant_id:
                where = {"$and": [
                    {"job_id": {"$eq": job_id}},
                    {"tenant_id": {"$eq": tenant_id}},
                ]}
            self.collection.delete(where=where)
        except Exception as e:
            logger.warning(f"ChromaStore delete_by_job failed (idempotent): {e}")


# ---------------------------------------------------------------------------
# FAISS (per-job artifact)
# ---------------------------------------------------------------------------

class FAISSStore(BaseVectorStore):
    """FAISS vector store — per-job file-based index with atomic writes."""

    def __init__(
        self,
        collection_name: str = "longparser",
        base_dir: str = "./faiss_data",
        index_fingerprint: str = "",
    ):
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install: pip install clean_rag[faiss]"
            )

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.index_fingerprint = index_fingerprint
        self._indexes: dict = {}  # job_id → (index, id_map, meta_map, doc_map)

    def _index_path(self, job_id: str) -> Path:
        # Isolate semantic space using fingerprint suffix explicitly
        dirname = f"{job_id}_{self.index_fingerprint}" if self.index_fingerprint else job_id
        return self.base_dir / dirname

    def _load_index(self, job_id: str):
        import faiss
        import numpy as np

        idx_dir = self._index_path(job_id)
        index_file = idx_dir / "index.faiss"
        meta_file = idx_dir / "metadata.json"

        if index_file.exists() and meta_file.exists():
            index = faiss.read_index(str(index_file))
            with open(meta_file) as f:
                data = json.load(f)
            return index, data.get("ids", []), data.get("metadatas", []), data.get("documents", [])

        return None, [], [], []

    def _save_index(self, job_id: str, index, ids, metadatas, documents):
        """Atomic write: temp → fsync → rename."""
        import faiss

        idx_dir = self._index_path(job_id)
        idx_dir.mkdir(parents=True, exist_ok=True)

        # Write index atomically
        with tempfile.NamedTemporaryFile(dir=idx_dir, suffix=".faiss", delete=False) as tmp:
            tmp_path = tmp.name
            faiss.write_index(index, tmp_path)
            os.fsync(tmp.fileno())
        os.rename(tmp_path, str(idx_dir / "index.faiss"))

        # Write metadata atomically
        meta = {"ids": ids, "metadatas": metadatas, "documents": documents}
        with tempfile.NamedTemporaryFile(dir=idx_dir, suffix=".json", mode="w", delete=False) as tmp:
            tmp_meta_path = tmp.name
            json.dump(meta, tmp, default=str)
            os.fsync(tmp.fileno())
        os.rename(tmp_meta_path, str(idx_dir / "metadata.json"))

    def add(self, ids, embeddings, metadatas, documents) -> None:
        import faiss
        import numpy as np

        if not embeddings:
            return

        # Determine job_id from first metadata
        job_id = metadatas[0].get("job_id", "default") if metadatas else "default"

        # Load or create index
        index, existing_ids, existing_metas, existing_docs = self._load_index(job_id)

        dim = len(embeddings[0])
        if index is None:
            index = faiss.IndexFlatIP(dim)  # inner product (cosine with normalized vectors)

        # Add vectors
        vectors = np.array(embeddings, dtype="float32")
        index.add(vectors)

        # Append metadata
        existing_ids.extend(ids)
        existing_metas.extend(metadatas)
        existing_docs.extend(documents)

        self._save_index(job_id, index, existing_ids, existing_metas, existing_docs)

    def search(self, query_embedding, top_k=5, filters=None) -> list[dict]:
        import faiss
        import numpy as np

        job_id = filters.get("job_id", "default") if filters else "default"
        index, ids, metadatas, documents = self._load_index(job_id)

        if index is None or index.ntotal == 0:
            return []

        query = np.array([query_embedding], dtype="float32")
        scores, indices = index.search(query, min(top_k, index.ntotal))

        output = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(ids):
                continue
            output.append({
                "id": ids[idx],
                "score": float(scores[0][i]),
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "document": documents[idx] if idx < len(documents) else "",
            })
        return output

    def delete_by_job(self, job_id: str, tenant_id: str = "") -> None:
        """Delete entire per-job index directory (rebuild pattern)."""
        import shutil
        idx_dir = self._index_path(job_id)
        if idx_dir.exists():
            shutil.rmtree(idx_dir)
            logger.info(f"FAISSStore: deleted index for job {job_id}")


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class QdrantStore(BaseVectorStore):
    """Qdrant vector store adapter."""

    def __init__(
        self,
        collection_name: str = "longparser",
        url: str = "http://localhost:6333",
        index_fingerprint: str = "",
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install: pip install clean_rag[qdrant]"
            )

        self.client = QdrantClient(url=url)
        # Securely isolate vector spaces based on model config
        self.collection_name = f"{collection_name}_{index_fingerprint}" if index_fingerprint else collection_name
        self._distance = Distance.COSINE
        logger.info(f"QdrantStore: collection={collection_name}, url={url}")

    def _ensure_collection(self, dim: int) -> None:
        """Create or validate collection. Mismatch → new collection name."""
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name in collections:
            # Validate dim + metric
            info = self.client.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != dim:
                # Mismatch — create new collection with hash suffix
                import hashlib
                suffix = hashlib.md5(f"{dim}".encode()).hexdigest()[:8]
                self.collection_name = f"{self.collection_name}_{suffix}"
                logger.warning(
                    f"QdrantStore: dim mismatch, using collection: {self.collection_name}"
                )

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=self._distance),
            )

    def add(self, ids, embeddings, metadatas, documents) -> None:
        from qdrant_client.models import PointStruct

        if not embeddings:
            return

        dim = len(embeddings[0])
        self._ensure_collection(dim)

        points = []
        for i, (vid, emb, meta, doc) in enumerate(zip(ids, embeddings, metadatas, documents)):
            # Flatten lists in payload for Qdrant filtering
            payload = {**meta, "document": doc}
            for k, v in payload.items():
                if isinstance(v, list):
                    payload[k] = json.dumps(v)

            points.append(PointStruct(
                id=i,  # Qdrant needs int or UUID
                vector=emb,
                payload={**payload, "vector_id": vid},
            ))

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k=5, filters=None) -> list[dict]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        search_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            if conditions:
                search_filter = Filter(must=conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter,
        ).points

        output = []
        for hit in results:
            payload = hit.payload or {}
            # Restore lists
            for k, v in payload.items():
                if isinstance(v, str) and v.startswith("["):
                    try:
                        payload[k] = json.loads(v)
                    except (json.JSONDecodeError, ValueError):
                        pass
            output.append({
                "id": payload.get("vector_id", ""),
                "score": hit.score,
                "metadata": payload,
                "document": payload.get("document", ""),
            })
        return output

    def delete_by_job(self, job_id: str, tenant_id: str = "") -> None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            conditions = [FieldCondition(key="job_id", match=MatchValue(value=job_id))]
            if tenant_id:
                conditions.append(
                    FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
                )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(must=conditions),
            )
        except Exception as e:
            logger.warning(f"QdrantStore delete_by_job failed (idempotent): {e}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store(
    backend: str,
    collection_name: str = "longparser",
    index_fingerprint: str = "",
    **kwargs,
) -> BaseVectorStore:
    """Create a vector store adapter by name.

    Args:
        backend: "chroma", "faiss", or "qdrant"
        collection_name: Name for the collection/index
        index_fingerprint: 10-char hash to isolate different embedding models
        **kwargs: Backend-specific options

    Returns:
        Configured BaseVectorStore instance
    """
    if backend == "chroma":
        return ChromaStore(collection_name=collection_name, index_fingerprint=index_fingerprint, **kwargs)
    elif backend == "faiss":
        return FAISSStore(collection_name=collection_name, index_fingerprint=index_fingerprint, **kwargs)
    elif backend == "qdrant":
        return QdrantStore(collection_name=collection_name, index_fingerprint=index_fingerprint, **kwargs)
    else:
        raise ValueError(
            f"Unknown vector store backend: {backend}. "
            f"Supported: chroma, faiss, qdrant"
        )

"""Vector adapter contracts with controlled SDK clients and real file writes."""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from longparser.server.vectorstores import ChromaStore, FAISSStore, QdrantStore, get_vector_store


@pytest.fixture
def chroma(monkeypatch):
    collection = Mock()
    collection.query.return_value = {"ids": [[]], "metadatas": [], "documents": [], "distances": []}
    client = Mock()
    client.get_or_create_collection.return_value = collection
    module = ModuleType("chromadb")
    module.PersistentClient = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "chromadb", module)
    return module, client, collection


def test_chroma_add_search_and_delete_preserve_metadata_and_tenant_filter(chroma):
    module, client, collection = chroma
    store = get_vector_store("chroma", "chunks", index_fingerprint="model-a", persist_directory="vectors")
    module.PersistentClient.assert_called_once_with(path="vectors")
    client.get_or_create_collection.assert_called_once_with(name="chunks_model-a", metadata={"hnsw:space": "cosine"})
    store.add(["a"], [[1, 0]], [{"job_id": "job", "pages": [1, 2]}], ["text"])
    collection.upsert.assert_called_once_with(ids=["a"], embeddings=[[1, 0]], metadatas=[{"job_id": "job", "pages": "[1, 2]"}], documents=["text"])
    collection.query.return_value = {"ids": [["a"]], "metadatas": [[{"pages": "[1, 2]", "malformed": "[oops"}]], "documents": [["text"]], "distances": [[0.25]]}
    assert store.search([1, 0], top_k=3, filters={"tenant_id": "tenant", "job_id": "job"}) == [{"id": "a", "score": 0.75, "metadata": {"pages": [1, 2], "malformed": "[oops"}, "document": "text"}]
    assert collection.query.call_args.kwargs["where"] == {"$and": [{"tenant_id": {"$eq": "tenant"}}, {"job_id": {"$eq": "job"}}]}
    store.delete_by_job("job", "tenant")
    collection.delete.assert_called_once_with(where={"$and": [{"job_id": {"$eq": "job"}}, {"tenant_id": {"$eq": "tenant"}}]})
    collection.delete.side_effect = RuntimeError("already removed")
    store.delete_by_job("job")


@pytest.mark.parametrize("filters,expected", [(None, None), ({}, None), ({"job_id": "j"}, {"job_id": {"$eq": "j"}})])
def test_chroma_empty_results_and_optional_fields(chroma, filters, expected):
    _, _, collection = chroma
    store = ChromaStore()
    assert store.search([1], filters=filters) == []
    assert collection.query.call_args.kwargs["where"] == expected
    collection.query.return_value = {"ids": [["a"]], "metadatas": None, "documents": None, "distances": None}
    assert store.search([1]) == [{"id": "a", "score": 1.0, "metadata": {}, "document": ""}]


@pytest.fixture
def faiss(monkeypatch):
    index = Mock()
    index.ntotal = 2
    index.search.return_value = (np.array([[0.9, 0.2]]), np.array([[1, 0]]))
    module = ModuleType("faiss")
    module.IndexFlatIP = Mock(return_value=index)
    module.read_index = Mock(return_value=index)
    module.write_index = Mock(side_effect=lambda obj, path: Path(path).write_bytes(b"fake-sdk-index"))
    monkeypatch.setitem(sys.modules, "faiss", module)
    return module, index


def test_faiss_creates_atomic_files_reloads_appends_searches_and_deletes(faiss, tmp_path):
    module, index = faiss
    store = get_vector_store("faiss", base_dir=str(tmp_path), index_fingerprint="space")
    assert store.search([1, 0], filters={"job_id": "j"}) == []
    store.add([], [], [], [])
    store.add(["a", "b"], [[1, 0], [0, 1]], [{"job_id": "j", "tenant_id": "t"}] * 2, ["first", "second"])
    module.IndexFlatIP.assert_called_once_with(2)
    np.testing.assert_array_equal(index.add.call_args.args[0], np.array([[1, 0], [0, 1]], dtype="float32"))
    folder = tmp_path / "j_space"
    assert set(path.name for path in folder.iterdir()) == {"index.faiss", "metadata.json"}
    assert json.loads((folder / "metadata.json").read_text())["ids"] == ["a", "b"]
    assert store.search([0, 1], top_k=10, filters={"job_id": "j"}) == [{"id": "b", "score": 0.9, "metadata": {"job_id": "j", "tenant_id": "t"}, "document": "second"}, {"id": "a", "score": 0.2, "metadata": {"job_id": "j", "tenant_id": "t"}, "document": "first"}]
    store.add(["c"], [[1, 1]], [{"job_id": "j"}], ["third"])
    assert json.loads((folder / "metadata.json").read_text())["ids"] == ["a", "b", "c"]
    store.delete_by_job("j", "t")
    assert not folder.exists()
    store.delete_by_job("j", "t")


def test_faiss_default_job_and_bad_sdk_indices_are_handled(faiss, tmp_path):
    _, index = faiss
    store = FAISSStore(base_dir=str(tmp_path))
    store.add(["a"], [[1, 0]], [], [])
    index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[-1, 20, 0]]))
    assert store.search([1, 0]) == [{"id": "a", "score": 0.7, "metadata": {}, "document": ""}]
    index.ntotal = 0
    assert store.search([1, 0], filters={}) == []


@pytest.fixture
def qdrant(monkeypatch):
    client = Mock()
    client.get_collections.return_value = SimpleNamespace(collections=[])
    module = ModuleType("qdrant_client")
    module.QdrantClient = Mock(return_value=client)
    models = ModuleType("qdrant_client.models")
    models.Distance = SimpleNamespace(COSINE="cosine")
    for name in ("VectorParams", "PointStruct", "FieldCondition", "Filter", "MatchValue"):
        setattr(models, name, lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", models)
    return module, client


def test_qdrant_collection_creation_payload_search_and_tenant_delete(qdrant):
    module, client = qdrant
    store = get_vector_store("qdrant", "chunks", index_fingerprint="space", url="http://qdrant.test")
    module.QdrantClient.assert_called_once_with(url="http://qdrant.test")
    store.add([], [], [], [])
    client.upsert.assert_not_called()
    store.add(["a"], [[1, 0]], [{"pages": [2], "tenant_id": "t"}], ["text"])
    assert client.create_collection.call_args.kwargs["collection_name"] == "chunks_space"
    assert client.create_collection.call_args.kwargs["vectors_config"].size == 2
    point = client.upsert.call_args.kwargs["points"][0]
    assert point.vector == [1, 0] and point.payload == {"pages": "[2]", "tenant_id": "t", "document": "text", "vector_id": "a"}
    client.query_points.return_value = SimpleNamespace(points=[SimpleNamespace(score=0.8, payload={"vector_id": "a", "pages": "[2]", "bad": "[oops", "document": "text"}), SimpleNamespace(score=0.1, payload=None)])
    hits = store.search([1, 0], filters={"tenant_id": "t", "job_id": "j"})
    assert hits[0]["id"] == "a" and hits[0]["metadata"]["pages"] == [2]
    assert hits[0]["metadata"]["bad"] == "[oops"
    assert hits[1] == {"id": "", "score": 0.1, "metadata": {}, "document": ""}
    assert [(condition.key, condition.match.value) for condition in client.query_points.call_args.kwargs["query_filter"].must] == [("tenant_id", "t"), ("job_id", "j")]
    store.search([1, 0])
    assert client.query_points.call_args.kwargs["query_filter"] is None
    store.delete_by_job("j", "t")
    assert [(condition.key, condition.match.value) for condition in client.delete.call_args.kwargs["points_selector"].must] == [("job_id", "j"), ("tenant_id", "t")]
    client.delete.side_effect = RuntimeError("already deleted")
    store.delete_by_job("j")


@pytest.mark.parametrize("existing_dim,expected_name", [(2, "chunks"), (3, "chunks_d4735e3a")])
def test_qdrant_existing_collection_dimension_policy(qdrant, existing_dim, expected_name):
    _, client = qdrant
    client.get_collections.return_value = SimpleNamespace(collections=[SimpleNamespace(name="chunks")])
    client.get_collection.return_value = SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=existing_dim))))
    store = QdrantStore("chunks")
    store.add(["a"], [[1, 0]], [{}], ["text"])
    assert store.collection_name == expected_name
    assert client.create_collection.call_count == (0 if existing_dim == 2 else 1)


@pytest.mark.parametrize("backend,module", [("chroma", "chromadb"), ("faiss", "faiss"), ("qdrant", "qdrant_client")])
def test_optional_vector_dependency_errors_are_actionable(monkeypatch, backend, module):
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == module:
            raise ImportError("SDK unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="Install") as caught:
        get_vector_store(backend)
    assert isinstance(caught.value.__cause__, ImportError)


def test_unknown_vector_backend_is_rejected():
    with pytest.raises(ValueError, match="Supported: chroma, faiss, qdrant"):
        get_vector_store("unsupported")

"""HTTP-level API workflows against controlled storage, queue and vector SDKs."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import socket
import sys
import zipfile
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from tests.unit.test_chat_flow import chat_sdks as _chat_sdk_fixture  # noqa: F401
from tests.unit.test_chat_flow import graph_storage as _graph_fixture  # noqa: F401


@pytest.fixture
async def api(memory_database, monkeypatch, tmp_path):
    def deny_network(*args, **kwargs):
        raise AssertionError("API tests must not contact external services")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    module = importlib.import_module("longparser.server.app")
    from longparser.server.queue import ARQBackend

    monkeypatch.setattr(module, "db", memory_database)
    monkeypatch.setattr(module, "queue", ARQBackend())
    monkeypatch.setattr(module, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(module, "_ADMIN_KEYS", set())
    monkeypatch.setattr(module.app.state, "_state", {})
    redis_pipeline = Mock()
    redis_pipeline.execute = AsyncMock(return_value=[0, 1, 1, True])
    monkeypatch.setattr(module._rate_limiter, "redis", SimpleNamespace(pipeline=Mock(return_value=redis_pipeline)))
    import arq

    pool = SimpleNamespace(enqueue_job=AsyncMock(return_value=SimpleNamespace(job_id="task")), close=AsyncMock())
    monkeypatch.setattr(arq, "create_pool", AsyncMock(return_value=pool))
    detector = ModuleType("fast_langdetect")
    detector.detect = lambda text: {"score": 1.0, "lang": "en"}
    monkeypatch.setitem(sys.modules, "fast_langdetect", detector)
    embeddings = ModuleType("langchain_huggingface")
    embeddings.HuggingFaceEmbeddings = Mock(return_value=SimpleNamespace(embed_query=Mock(return_value=[1, 0])))
    monkeypatch.setitem(sys.modules, "langchain_huggingface", embeddings)
    chroma = ModuleType("chromadb")
    collection = Mock()
    collection.query.return_value = {"ids": [["v"]], "metadatas": [[{"chunk_id": "c", "page_numbers": "[0]", "block_ids": "[\"b\"]"}]], "documents": [["policy"]], "distances": [[0.1]]}
    chroma.PersistentClient = Mock(return_value=SimpleNamespace(get_or_create_collection=Mock(return_value=collection)))
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://test", headers={"X-API-Key": "tenant-key"}) as client:
        yield SimpleNamespace(client=client, module=module, db=memory_database, root=tmp_path,
                              tenant=hashlib.sha256(b"tenant-key").hexdigest()[:32], pool=pool,
                              collection=collection, rate=redis_pipeline)


async def seed_review(api):
    await api.db.create_job(api.tenant, "job", "document.pdf", "hash")
    await api.db.update_job(api.tenant, "job", {"status": "ready_for_review"})
    await api.db.upsert_block(api.tenant, "job", {"block_id": "b", "text": "Original policy text.", "type": "paragraph", "page_number": 0, "confidence": {"overall": 0.9}})
    await api.db.upsert_chunk(api.tenant, "job", {"chunk_id": "c", "text": "Original policy text.", "chunk_type": "section"})


async def test_upload_streaming_filename_safety_listing_and_cancellation(api):
    response = await api.client.post("/jobs", files={"file": ("../../../document.pdf", b"%PDF test", "application/pdf")})
    assert response.status_code == 201, response.text
    job = response.json()
    assert job["source_file"] == "document.pdf" and job["file_hash"] == hashlib.sha256(b"%PDF test").hexdigest()
    assert (api.root / api.tenant / job["job_id"] / "document.pdf").read_bytes() == b"%PDF test"
    assert api.pool.enqueue_job.call_args.args == ("extract_job",)
    assert (await api.client.get("/jobs")).json()["total"] == 1
    assert (await api.client.get(f'/jobs/{job["job_id"]}')).json()["status"] == "queued"
    assert (await api.client.post(f'/jobs/{job["job_id"]}/cancel')).json()["status"] == "cancelled"
    assert (await api.client.get(f'/jobs/{job["job_id"]}', headers={"X-API-Key": "other-key"})).status_code == 404
    assert (await api.client.get("/jobs", headers={"X-API-Key": "short"})).status_code == 401
    assert (await api.client.post("/jobs", files={"file": ("script.exe", b"x", "application/x-executable")})).status_code == 415


async def test_upload_limit_and_default_filename(api, monkeypatch):
    from fastapi import HTTPException, UploadFile

    monkeypatch.setattr(api.module, "MAX_UPLOAD_SIZE", 3)
    response = await api.client.post("/jobs", files={"file": ("large.pdf", b"1234", "application/pdf")})
    assert response.status_code == 413
    assert not list(api.root.rglob("large.pdf"))
    monkeypatch.setattr(api.module, "MAX_UPLOAD_SIZE", 100)
    job = await api.module.create_job(UploadFile(io.BytesIO(b"text"), filename=".."), x_api_key="tenant-key")
    assert job.source_file == "document"
    with pytest.raises(HTTPException) as error:
        await api.module._stream_upload(UploadFile(io.BytesIO(b"x" * 101)), api.root / "oversize")
    assert error.value.status_code == 413


async def test_block_and_chunk_review_versions_rechunk_and_audit(api):
    await seed_review(api)
    blocks = (await api.client.get("/jobs/job/blocks", params={"status": "pending", "type": "paragraph", "page": 0})).json()
    assert blocks[0]["text"] == "Original policy text." and "confidence" not in blocks[0]
    response = await api.client.patch("/jobs/job/blocks/b", json={"status": "edited", "edited_text": "Corrected policy.", "edited_type": "heading", "version": 1})
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "Corrected policy." and response.json()["type"] == "heading"
    assert (await api.client.patch("/jobs/job/blocks/b", json={"status": "approved", "version": 1})).status_code == 409
    chunks = (await api.client.get("/jobs/job/chunks")).json()
    assert chunks and "Corrected policy." in chunks[0]["text"]
    chunk_id = chunks[0]["chunk_id"]
    edited = await api.client.patch(f"/jobs/job/chunks/{chunk_id}", json={"status": "edited", "edited_text": "Final chunk.", "version": 1})
    assert edited.json()["text"] == "Final chunk."
    assert (await api.client.patch(f"/jobs/job/chunks/{chunk_id}", json={"status": "approved", "version": 1})).status_code == 409
    assert len((await api.client.get("/jobs/job/audit")).json()) >= 2
    assert (await api.client.post("/jobs/job/rechunk")).status_code == 200


async def test_finalize_requires_review_then_embedding_and_export(api):
    await seed_review(api)
    assert (await api.client.post("/jobs/job/embed", json={})).status_code == 400
    assert (await api.client.post("/jobs/job/finalize", json={"finalize_policy": "require_all_approved"})).status_code == 400
    response = await api.client.post("/jobs/job/finalize", json={"finalize_policy": "approve_pending"})
    assert response.json()["status"] == "finalized"
    assert (await api.client.post("/jobs/job/cancel")).status_code == 400
    assert (await api.client.post("/jobs/job/rechunk")).status_code == 400
    exported = await api.client.get("/jobs/job/export")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert set(archive.namelist()) == {"blocks.json", "chunks.json", "document.md"}
        assert "Original policy text." in archive.read("document.md").decode()
        assert json.loads(archive.read("blocks.json"))[0]["review_status"] == "approved"
    embedded = await api.client.post("/jobs/job/embed", json={"provider": "huggingface", "model": "model", "vector_db": "chroma"})
    assert embedded.json()["status"] == "embedding"
    assert api.pool.enqueue_job.call_args.args == ("embed_job",)


async def test_admin_purge_scrubs_tombstone_and_updates_counts(api, monkeypatch):
    await seed_review(api)
    monkeypatch.setattr(api.module, "_ADMIN_KEYS", {"admin-key"})
    assert (await api.client.post("/jobs/job/chunks/c/purge")).status_code == 403
    monkeypatch.setattr(api.module, "_ADMIN_KEYS", {"tenant-key"})
    chunk = await api.client.post("/jobs/job/chunks/c/purge")
    assert chunk.json()["status"] == "purged"
    block = await api.client.post("/jobs/job/blocks/b/purge")
    assert block.json()["chunks_after_rechunk"] == 0
    history = (await api.client.get("/jobs/job/audit")).json()
    assert all("Original policy text." not in row["original_text"] for row in history)
    assert all("[PURGED]" in row["original_text"] for row in history)


@pytest.mark.parametrize("method,path,payload", [("get", "/jobs/missing", None), ("delete", "/jobs/missing", None), ("post", "/jobs/missing/cancel", {}), ("post", "/jobs/missing/rechunk", {}), ("post", "/jobs/missing/finalize", {}), ("post", "/jobs/missing/embed", {}), ("get", "/jobs/missing/export", None), ("patch", "/jobs/job/blocks/missing", {"status": "approved", "version": 1}), ("patch", "/jobs/job/chunks/missing", {"status": "approved", "version": 1}), ("post", "/jobs/job/blocks/missing/purge", {}), ("post", "/jobs/job/chunks/missing/purge", {})])
async def test_missing_entities_are_not_found(api, method, path, payload):
    response = await api.client.request(method, path, **({"json": payload} if payload is not None else {}))
    assert response.status_code == 404, response.text


async def test_search_reconstructs_index_and_preserves_source_mapping(api):
    await seed_review(api)
    assert (await api.client.post("/search", json={"query": "policy", "job_id": "missing"})).status_code == 404
    assert (await api.client.post("/search", json={"query": "policy", "job_id": "job"})).status_code == 404
    await api.db.create_index_version(api.tenant, "job", "iv", {"status": "indexed", "provider": "huggingface", "model": "model", "vector_db": "chroma", "collection": "chunks", "configured_dimensions": 2})
    for index in (None, "iv"):
        response = await api.client.post("/search", json={"query": "policy", "job_id": "job", "index_version": index, "filters": {"chunk_type": "section"}})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["index_version"] == "iv" and result["results"][0]["chunk_id"] == "c"
        assert result["results"][0]["page_numbers"] == [0]


async def test_search_filters_cannot_override_authenticated_tenant_or_job(api):
    await seed_review(api)
    await api.db.create_index_version(api.tenant, "job", "iv", {"status": "indexed", "provider": "huggingface", "model": "model", "vector_db": "chroma", "configured_dimensions": 2})
    response = await api.client.post("/search", json={"query": "policy", "job_id": "job", "filters": {"tenant_id": "another-tenant", "job_id": "another-job"}})
    assert response.status_code == 200, response.text
    assert api.collection.query.call_args.kwargs["where"] == {"$and": [{"tenant_id": {"$eq": api.tenant}}, {"job_id": {"$eq": "job"}}]}


async def test_job_deletion_removes_artifacts_and_vector_indexes(api):
    await seed_review(api)
    await api.db.create_index_version(api.tenant, "job", "iv", {"model": "model", "vector_db": "chroma"})
    folder = api.root / api.tenant / "job"
    folder.mkdir(parents=True)
    (folder / "document.pdf").write_bytes(b"test")
    assert (await api.client.delete("/jobs/job")).status_code == 204
    assert not folder.exists() and await api.db.get_job(api.tenant, "job") is None
    api.collection.delete.assert_called_once()


async def test_session_binding_history_deletion_health_and_rate_limit(api):
    await seed_review(api)
    assert (await api.client.post("/chat/sessions", json={"job_id": "missing"})).status_code == 404
    session = (await api.client.post("/chat/sessions", json={"job_id": "job"})).json()["session_id"]
    assert (await api.client.get(f"/chat/sessions/{session}")).json()["turns"] == []
    assert (await api.client.post("/chat", json={"session_id": session, "job_id": "different", "question": "policy?"})).status_code == 400
    assert (await api.client.post("/chat", json={"session_id": "missing", "job_id": "job", "question": "policy?"})).status_code == 404
    assert (await api.client.post("/chat/resume", json={"session_id": "missing", "thread_id": "thread", "action": "approve"})).status_code == 404
    assert (await api.client.delete(f"/chat/sessions/{session}")).json()["status"] == "deleted"
    assert (await api.client.get(f"/chat/sessions/{session}")).status_code == 404
    assert (await api.client.delete(f"/chat/sessions/{session}")).status_code == 404
    assert (await api.client.get("/health")).json()["status"] == "ok"
    api.rate.execute.return_value = [0, 1, 1000, True]
    assert (await api.client.get("/jobs")).status_code == 429


async def test_chat_http_hitl_edit_and_lifecycle(api, request):
    request.getfixturevalue("_chat_sdk_fixture")
    graph_storage = request.getfixturevalue("_graph_fixture")
    from longparser.server.chat.checkpointer import init_checkpointer
    await init_checkpointer("mongodb://memory.test", "reviews")
    await seed_review(api)
    await api.db.create_index_version(api.tenant, "job", "iv", {"status": "indexed", "model": "model", "provider": "huggingface", "vector_db": "chroma", "collection": "chunks", "configured_dimensions": 2})
    await api.db.create_chat_session(api.tenant, "session", "job")
    response = await api.client.post("/chat", json={"session_id": "session", "job_id": "job", "question": "When can I return it?", "require_approval": True})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "pending_review"
    assert response.json()["answer"] == "Thirty days."
    resumed = await api.client.post("/chat/resume", json={"session_id": "session", "thread_id": response.json()["thread_id"], "action": "edit", "edited_answer": "Within 45 days."})
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["answer"] == "Within 45 days."
    turns = await api.db.get_all_turns(api.tenant, "session")
    assert turns[-1]["answer"] == "Within 45 days."
    async with api.module.lifespan(api.module.app):
        assert api.db.jobs.indexes
    assert api.db.client.closed
    graph_storage[0].close.assert_called_once()

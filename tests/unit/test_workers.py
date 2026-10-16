"""Worker workflows execute real processing/storage adapters at SDK boundaries."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from longparser.server.worker import (
    WorkerSettings,
    embed_job,
    enrich_summaries_job,
    extract_facts,
    extract_job,
    purge_expired_sessions,
    summarize_session,
)


@pytest.fixture
def worker_sdks(monkeypatch, tmp_path):
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    doc = DoclingDocument(name="fixture")
    doc.add_heading(text="Policy", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Return unused items within thirty days.")
    converter = Mock(convert=Mock(return_value=SimpleNamespace(document=doc)))
    module = importlib.import_module("longparser.extractors.docling_extractor")
    monkeypatch.setattr(module, "DocumentConverter", Mock(return_value=converter))
    for name, value in {"LONGPARSER_DO_OCR": "false", "LONGPARSER_FORMULA_OCR": "false", "LONGPARSER_GENERATE_SUMMARIES": "true", "LONGPARSER_LLM_PROVIDER": "openai", "LONGPARSER_LLM_MODEL": "fixture", "LONGPARSER_CHAT_SHORT_TERM_TURNS": "1", "LONGPARSER_CHAT_MAX_FACTS": "2"}.items():
        monkeypatch.setenv(name, value)
    detector = ModuleType("fast_langdetect")
    detector.detect = lambda text: {"lang": "en", "score": 1.0}
    monkeypatch.setitem(sys.modules, "fast_langdetect", detector)
    embedding_model = Mock(embed_documents=Mock(side_effect=lambda texts: [[1, 0] for _ in texts]))
    embeddings = ModuleType("langchain_huggingface")
    embeddings.HuggingFaceEmbeddings = Mock(return_value=embedding_model)
    monkeypatch.setitem(sys.modules, "langchain_huggingface", embeddings)
    vector_collection = Mock()
    chroma = ModuleType("chromadb")
    chroma.PersistentClient = Mock(return_value=SimpleNamespace(get_or_create_collection=Mock(return_value=vector_collection)))
    monkeypatch.setitem(sys.modules, "chromadb", chroma)
    llm = SimpleNamespace(ainvoke=AsyncMock(return_value=SimpleNamespace(content="A concise summary.")))
    provider = ModuleType("langchain_openai")
    provider.ChatOpenAI = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "langchain_openai", provider)
    import tiktoken

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: SimpleNamespace(encode=lambda text: text.split(), decode=lambda tokens: " ".join(tokens)))
    file_path = tmp_path / "fixture.docx"
    file_path.write_bytes(b"external conversion boundary")
    redis = SimpleNamespace(enqueue_job=AsyncMock())
    return SimpleNamespace(converter=converter, file=file_path, llm=llm, embeddings=embedding_model,
                           vectors=vector_collection, context={"redis": redis})


async def test_extraction_worker_persists_content_and_enqueues_summaries(memory_database, worker_sdks):
    await memory_database.create_job("tenant", "job", "fixture.docx", "hash")
    result = await extract_job(worker_sdks.context, "tenant", "job", str(worker_sdks.file))
    assert result == {"status": "ready_for_review", "blocks": 2, "chunks": 1}
    blocks = await memory_database.get_blocks("tenant", "job")
    chunks = await memory_database.get_chunks("tenant", "job")
    assert blocks[1]["text"] == "Return unused items within thirty days."
    assert blocks[1]["text_hash"] and "thirty days" in chunks[0]["text"]
    worker_sdks.context["redis"].enqueue_job.assert_awaited_once_with("enrich_summaries_job", "tenant", "job", _job_id="summary-job")
    assert memory_database.client.closed


@pytest.mark.parametrize("status", ["cancelled", "ready_for_review", "finalized", "indexed"])
async def test_extraction_worker_skips_terminal_states(memory_database, worker_sdks, status):
    await memory_database.create_job("tenant", "job", "fixture.docx", "hash")
    await memory_database.update_job("tenant", "job", {"status": status})
    assert await extract_job({}, "tenant", "job", "unused") == {"status": status}
    worker_sdks.converter.convert.assert_not_called()


async def test_extraction_worker_reports_missing_failure_and_midflight_cancel(memory_database, worker_sdks):
    assert await extract_job({}, "tenant", "missing", "unused") == {"error": "job_not_found"}
    await memory_database.create_job("tenant", "job", "fixture.docx", "hash")
    worker_sdks.converter.convert.side_effect = RuntimeError("converter failed")
    assert "converter failed" in (await extract_job({}, "tenant", "job", str(worker_sdks.file)))["error"]
    assert (await memory_database.get_job("tenant", "job"))["status"] == "failed"
    result = worker_sdks.converter.convert.return_value

    def cancellation(*args, **kwargs):
        memory_database.jobs.documents[0]["status"] = "cancelled"
        return result

    await memory_database.update_job("tenant", "job", {"status": "queued"})
    worker_sdks.converter.convert.side_effect = cancellation
    assert await extract_job({}, "tenant", "job", str(worker_sdks.file)) == {"status": "cancelled"}


async def test_embedding_worker_preserves_edited_text_and_records_index_space(memory_database, worker_sdks, monkeypatch):
    monkeypatch.setenv("LONGPARSER_EMBED_DIMENSIONS", "2")
    await memory_database.create_job("tenant", "job", "fixture.docx", "hash")
    await memory_database.upsert_chunk("tenant", "job", {"chunk_id": "chunk", "text": "original", "edited_text": "reviewed", "review_status": "edited", "section_path": ["Policy"], "page_numbers": [0]})
    result = await embed_job({}, "tenant", "job", "model", "chroma", "collection", "iv")
    assert result == {"status": "indexed", "embedded": 1}
    worker_sdks.embeddings.embed_documents.assert_called_once_with(["reviewed"])
    call = worker_sdks.vectors.upsert.call_args.kwargs
    assert call["ids"] == ["tenant:job:chunk:iv"] and call["documents"] == ["reviewed"]
    assert call["metadatas"][0]["page_numbers"] == "[0]"
    index = await memory_database.get_latest_index_version("tenant", "job")
    assert index["configured_dimensions"] == 2 and index["dim"] == 2
    assert index["fingerprint"] and index["status"] == "indexed"


async def test_embedding_worker_cancel_empty_and_provider_failure(memory_database, worker_sdks):
    args = ({}, "tenant", "job", "model", "chroma", "collection", "iv")
    assert await embed_job(*args) == {"status": "cancelled"}
    await memory_database.create_job("tenant", "job", "file", "hash")
    assert await embed_job(*args) == {"status": "indexed", "embedded": 0}
    await memory_database.upsert_chunk("tenant", "job", {"chunk_id": "chunk", "text": "text", "review_status": "approved"})
    worker_sdks.embeddings.embed_documents.side_effect = RuntimeError("provider failed")
    assert await embed_job(*args) == {"error": "provider failed"}
    assert (await memory_database.get_job("tenant", "job"))["status"] == "failed"


async def test_summary_worker_generates_section_chunks_with_real_enricher(memory_database, worker_sdks):
    assert await enrich_summaries_job({}, "tenant", "missing") == {"status": "cancelled"}
    await memory_database.create_job("tenant", "job", "file", "hash")
    await memory_database.upsert_chunk("tenant", "job", {"chunk_id": "source", "text": "Policy content.", "chunk_type": "section", "section_path": ["Policy"]})
    result = await enrich_summaries_job({}, "tenant", "job", provider="openai", model="fixture")
    assert result == {"status": "enriched", "summary_chunks": 1}
    chunks = await memory_database.get_chunks("tenant", "job")
    summary = next(chunk for chunk in chunks if chunk["chunk_type"] == "summary")
    assert summary["text"] == "A concise summary."


async def seed_session_turns(database):
    await database.create_chat_session("tenant", "session", "job")
    now = datetime.now(UTC)
    for index in range(3):
        await database.save_turn("tenant", "session", SimpleNamespace(model_dump=lambda mode, index=index: {"turn_id": str(index), "question": f"question {index}", "answer": "answer", "created_at": now + timedelta(minutes=index), "archived": False}))


async def test_session_summary_archives_only_older_turns(memory_database, worker_sdks):
    assert await summarize_session({}, "tenant", "missing") == {"error": "session_not_found"}
    await memory_database.create_chat_session("tenant", "session", "job")
    assert (await summarize_session({}, "tenant", "session"))["status"] == "skipped"
    await seed_session_turns(memory_database)
    assert await summarize_session({}, "tenant", "session") == {"status": "summarized", "archived": 2}
    assert (await memory_database.get_chat_session("tenant", "session"))["rolling_summary"] == "A concise summary."
    assert [turn["turn_id"] for turn in await memory_database.get_unarchived_turns("tenant", "session")] == ["2"]


async def test_session_summary_conflict_and_provider_error(memory_database, worker_sdks):
    await seed_session_turns(memory_database)

    async def competing_update(messages):
        memory_database.chat_sessions.documents[0]["version"] += 1
        return SimpleNamespace(content="new")

    worker_sdks.llm.ainvoke.side_effect = competing_update
    assert await summarize_session({}, "tenant", "session") == {"status": "conflict"}
    worker_sdks.llm.ainvoke.side_effect = RuntimeError("provider down")
    assert await summarize_session({}, "tenant", "session") == {"error": "provider down"}


async def test_fact_extraction_filters_sources_types_caps_history_and_handles_bad_json(memory_database, worker_sdks):
    assert await extract_facts({}, "tenant", "missing", "job") == {"error": "session_not_found"}
    await memory_database.create_chat_session("tenant", "empty", "job")
    assert (await extract_facts({}, "tenant", "empty", "job"))["status"] == "skipped"
    await seed_session_turns(memory_database)
    facts = [{"type": "entities_from_doc", "source": "doc", "fact": "first"}, {"type": "user_preferences", "source": "user", "fact": "second"}, {"type": "decisions", "source": "user", "fact": "third", "confidence": 0.9}, {"type": "unsupported", "source": "doc", "fact": "ignore"}, {"type": "decisions", "source": "assistant_inference", "fact": "ignore"}]
    worker_sdks.llm.ainvoke.return_value = SimpleNamespace(content=json.dumps({"facts": facts}))
    assert await extract_facts({}, "tenant", "session", "job") == {"status": "extracted", "new_facts": 3, "total": 2}
    session = await memory_database.get_chat_session("tenant", "session")
    assert [fact["fact"] for fact in session["long_term_facts"]] == ["second", "third"]
    worker_sdks.llm.ainvoke.return_value = SimpleNamespace(content="not json")
    assert await extract_facts({}, "tenant", "session", "job") == {"error": "invalid_json"}
    worker_sdks.llm.ainvoke.side_effect = RuntimeError("provider down")
    assert await extract_facts({}, "tenant", "session", "job") == {"error": "provider down"}


async def test_fact_extraction_version_conflict(memory_database, worker_sdks):
    await seed_session_turns(memory_database)

    async def competing_update(messages):
        memory_database.chat_sessions.documents[0]["version"] += 1
        return SimpleNamespace(content='{"facts": []}')

    worker_sdks.llm.ainvoke.side_effect = competing_update
    assert await extract_facts({}, "tenant", "session", "job") == {"status": "conflict"}


async def test_retention_worker_purges_expired_sessions_and_keeps_active_ones(memory_database, worker_sdks):
    await seed_session_turns(memory_database)
    await memory_database.create_chat_session("tenant", "active", "job")
    await memory_database.chat_sessions.update_one({"session_id": "session"}, {"$set": {"deleted_at": datetime.now(UTC) - timedelta(days=40)}})
    assert await purge_expired_sessions({}) == {"status": "purged", "sessions": 1, "turns": 3}
    assert await memory_database.get_chat_session("tenant", "active")
    assert await purge_expired_sessions({}) == {"status": "purged", "sessions": 0, "turns": 0}
    await WorkerSettings.on_startup({})
    await WorkerSettings.on_shutdown({})
    assert len(WorkerSettings.functions) == 6 and WorkerSettings.cron_jobs

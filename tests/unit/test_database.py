"""Tenant-scoped persistence, review/versioning and memory through Database."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from longparser.server.db import Database
from longparser.server.schemas import FinalizePolicy, ReviewStatus, Revision
from tests.memory_mongo import MotorClient


@pytest.fixture
def database(monkeypatch):
    import longparser.server.db as module

    monkeypatch.setattr(module, "AsyncIOMotorClient", MotorClient)
    monkeypatch.setenv("LONGPARSER_MONGO_URL", "mongodb://memory.test")
    return Database()


async def test_jobs_are_tenant_scoped_paginated_and_deleted_with_dependents(database):
    job = await database.create_job("tenant", "job", "report.pdf", "hash")
    assert job["status"] == "queued" and job["total_chunks"] == 0
    assert await database.get_job("other", "job") is None
    assert await database.update_job("tenant", "job", {"status": "finalized"})
    assert not await database.update_job("other", "job", {"status": "finalized"})
    await database.create_job("other", "job", "private.pdf", "hash")
    await database.create_job("tenant", "second", "other.pdf", "hash")
    jobs, total = await database.list_jobs("tenant", status="finalized", limit=1)
    assert total == 1 and jobs[0]["source_file"] == "report.pdf"
    assert len((await database.list_jobs("tenant", skip=1))[0]) == 1
    await database.upsert_block("tenant", "job", {"block_id": "b", "text": "text"})
    await database.upsert_chunk("tenant", "job", {"chunk_id": "c", "text": "text"})
    await database.delete_job("tenant", "job")
    assert await database.get_job("tenant", "job") is None
    assert await database.get_blocks("tenant", "job") == []
    assert (await database.get_job("other", "job"))["source_file"] == "private.pdf"


@pytest.mark.parametrize("kind", ["block", "chunk"])
async def test_upsert_and_review_use_optimistic_versions_and_filters(database, kind):
    entity = {f"{kind}_id": "item", "text": "original", "type": "paragraph", "chunk_type": "section", "page_number": 0}
    upsert = getattr(database, f"upsert_{kind}")
    await upsert("tenant", "job", entity)
    await upsert("tenant", "job", entity)
    await upsert("other", "job", {**entity, "text": "private"})
    retrieve = getattr(database, f"get_{kind}s")
    filters = {"status": "pending", "block_type": "paragraph", "page": 0} if kind == "block" else {"status": "pending", "chunk_type": "section"}
    rows = await retrieve("tenant", "job", **filters)
    assert len(rows) == 1 and rows[0]["text"] == "original" and rows[0]["version"] == 1
    update = getattr(database, f"update_{kind}_review")
    edits = {"edited_text": "corrected", "revision_id": "revision"}
    if kind == "block":
        edits["edited_type"] = "heading"
    reviewed = await update("tenant", "job", "item", "edited", 1, **edits)
    assert reviewed["version"] == 2 and reviewed["edited_text"] == "corrected"
    assert await update("tenant", "job", "item", "approved", 1) is None
    assert await update("other", "different", "item", "approved", 1) is None


async def test_review_progress_finalize_and_approved_chunk_selection(database):
    for index, status in enumerate(["pending", "approved", "edited", "rejected", "unknown"]):
        await database.upsert_block("tenant", "job", {"block_id": str(index), "review_status": status})
    await database.upsert_chunk("tenant", "job", {"chunk_id": "pending"})
    assert (await database.get_review_progress("tenant", "job")).model_dump() == {"approved": 1, "edited": 1, "rejected": 1, "pending": 1}
    assert await database.apply_finalize_policy("tenant", "job", FinalizePolicy.REQUIRE_ALL_APPROVED) == 2
    assert await database.apply_finalize_policy("tenant", "job", FinalizePolicy.APPROVE_PENDING) == 2
    assert [chunk["chunk_id"] for chunk in await database.get_approved_chunks("tenant", "job")] == ["pending"]
    await database.upsert_chunk("tenant", "job", {"chunk_id": "reject"})
    assert await database.apply_finalize_policy("tenant", "job", FinalizePolicy.REJECT_PENDING) == 1


async def test_revision_history_is_ordered_and_tenant_scoped(database):
    now = datetime.now(UTC)
    for kind, stamp in [("block", now), ("chunk", now - timedelta(minutes=1))]:
        revision = Revision(entity_type=kind, entity_id="id", action=ReviewStatus.EDITED, original_text="old", edited_text="new", timestamp=stamp)
        await database.create_revision("tenant", "job", revision)
    assert [row["entity_type"] for row in await database.get_audit_trail("tenant", "job")] == ["chunk", "block"]
    assert len(await database.get_audit_trail("tenant", "job", skip=1, limit=1)) == 1
    assert await database.get_audit_trail("other", "job") == []


async def test_index_versions_select_latest_indexed_and_keep_config(database):
    assert await database.get_latest_index_version("tenant", "job") is None
    await database.create_index_version("tenant", "job", "a", {"model": "model-a"})
    await database.create_index_version("tenant", "job", "b", {"model": "model-b", "status": "indexed"})
    assert (await database.get_latest_index_version("tenant", "job"))["model"] == "model-b"
    assert len(await database.list_index_versions("tenant", "job")) == 2
    assert await database.get_latest_index_version("other", "job") is None


async def test_chat_memory_updates_lock_versions_and_exclude_deleted_sessions(database):
    session = await database.create_chat_session("tenant", "session", "job")
    assert session["version"] == 1 and session["turn_count"] == 0
    assert await database.get_chat_session("other", "session") is None
    assert await database.update_rolling_summary("tenant", "session", "summary", 1)
    assert not await database.update_long_term_facts("tenant", "session", [], 1)
    assert await database.update_long_term_facts("tenant", "session", [{"fact": "preference"}], 2)
    updated = await database.get_chat_session("tenant", "session")
    assert updated["rolling_summary"] == "summary" and updated["long_term_facts"] == [{"fact": "preference"}]
    assert await database.soft_delete_chat_session("tenant", "session")
    assert await database.get_chat_session("tenant", "session") is None
    assert not await database.soft_delete_chat_session("tenant", "session")


async def test_turn_history_archival_idempotency_lookup_and_retention(database):
    await database.create_chat_session("tenant", "session", "job")
    now = datetime.now(UTC)
    for index in range(3):
        turn = SimpleNamespace(model_dump=lambda mode, index=index: {"turn_id": str(index), "idempotency_key": f"key-{index}", "created_at": now + timedelta(minutes=index), "archived": False})
        await database.save_turn("tenant", "session", turn)
    assert (await database.get_chat_session("tenant", "session"))["turn_count"] == 3
    assert [turn["turn_id"] for turn in await database.get_recent_turns("tenant", "session", n=2)] == ["1", "2"]
    assert (await database.get_turn_by_idempotency_key("tenant", "session", "key-0"))["turn_id"] == "0"
    assert await database.get_turn_by_idempotency_key("other", "session", "key-0") is None
    assert await database.archive_turns("tenant", "session", ["0", "1"]) == 2
    assert len(await database.get_unarchived_turns("tenant", "session")) == 1
    assert len(await database.get_all_turns("tenant", "session")) == 3
    await database.chat_sessions.update_one({"session_id": "session"}, {"$set": {"deleted_at": now - timedelta(days=40)}})
    assert await database.get_expired_sessions() == [{"tenant_id": "tenant", "session_id": "session"}]
    assert await database.purge_turns_for_session("tenant", "session") == 3


async def test_indexes_cover_unique_tenant_keys_and_retention(database):
    await database.create_indexes()
    assert ([('tenant_id', 1), ('job_id', 1)], {"unique": True}) in database.jobs.indexes
    assert any(options.get("expireAfterSeconds") == 2592000 for _, options in database.chat_sessions.indexes)
    assert any(options.get("unique") and keys[-1] == ("idempotency_key", 1) for keys, options in database.chat_turns.indexes)
    await database.close()
    assert database.client.closed

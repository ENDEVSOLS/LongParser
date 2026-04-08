"""MongoDB client and CRUD operations for LongParser API.

Uses Motor (async MongoDB driver) with tenant-scoped queries and
materialized path hierarchy indexes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .schemas import (
    JobStatus,
    ReviewStatus,
    FinalizePolicy,
    Revision,
    JobResponse,
    ReviewProgress,
    BlockResponse,
    ChunkResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database client
# ---------------------------------------------------------------------------

class Database:
    """Async MongoDB interface for LongParser."""

    def __init__(self, mongo_url: Optional[str] = None, db_name: str = "longparser"):
        import os
        if mongo_url is None:
            mongo_url = os.getenv("LONGPARSER_MONGO_URL", "mongodb://localhost:27017")
        self.client = AsyncIOMotorClient(mongo_url)
        self.db: AsyncIOMotorDatabase = self.client[db_name]

        # Collections
        self.jobs = self.db["jobs"]
        self.blocks = self.db["blocks"]
        self.chunks = self.db["chunks"]
        self.block_revisions = self.db["block_revisions"]
        self.chunk_revisions = self.db["chunk_revisions"]
        self.index_versions = self.db["index_versions"]
        self.chat_sessions = self.db["chat_sessions"]
        self.chat_turns = self.db["chat_turns"]

    async def create_indexes(self) -> None:
        """Create all required indexes (idempotent)."""
        # Jobs
        await self.jobs.create_index(
            [("tenant_id", 1), ("job_id", 1)], unique=True
        )
        await self.jobs.create_index([("tenant_id", 1), ("status", 1)])

        # Blocks
        await self.blocks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("block_id", 1)], unique=True
        )
        await self.blocks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("review_status", 1)]
        )
        await self.blocks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("type", 1)]
        )
        await self.blocks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("hierarchy_path", 1)]
        )

        # Chunks
        await self.chunks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("chunk_id", 1)], unique=True
        )
        await self.chunks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("review_status", 1)]
        )
        await self.chunks.create_index(
            [("tenant_id", 1), ("job_id", 1), ("section_path", 1)]
        )

        # Revisions
        await self.block_revisions.create_index(
            [("tenant_id", 1), ("job_id", 1), ("block_id", 1), ("timestamp", 1)]
        )
        await self.chunk_revisions.create_index(
            [("tenant_id", 1), ("job_id", 1), ("chunk_id", 1), ("timestamp", 1)]
        )

        # Index versions
        await self.index_versions.create_index(
            [("tenant_id", 1), ("job_id", 1), ("index_version", 1)], unique=True
        )
        await self.index_versions.create_index(
            [("tenant_id", 1), ("job_id", 1), ("status", 1)]
        )

        # Chat sessions
        await self.chat_sessions.create_index(
            [("tenant_id", 1), ("session_id", 1)], unique=True
        )
        await self.chat_sessions.create_index(
            [("tenant_id", 1), ("job_id", 1)]
        )
        # TTL index — auto-purge soft-deleted sessions after 30 days
        await self.chat_sessions.create_index(
            [("deleted_at", 1)],
            expireAfterSeconds=30 * 24 * 3600,
            partialFilterExpression={"deleted_at": {"$type": "date"}},
        )

        # Chat turns
        await self.chat_turns.create_index(
            [("tenant_id", 1), ("session_id", 1), ("created_at", 1)]
        )
        await self.chat_turns.create_index(
            [("tenant_id", 1), ("session_id", 1), ("idempotency_key", 1)],
            unique=True,
            partialFilterExpression={"idempotency_key": {"$type": "string"}},
        )
        await self.chat_turns.create_index(
            [("session_id", 1), ("archived", 1)]
        )

        logger.info("MongoDB indexes created.")

    # -----------------------------------------------------------------------
    # Jobs CRUD
    # -----------------------------------------------------------------------

    async def create_job(
        self, tenant_id: str, job_id: str, source_file: str, file_hash: str
    ) -> dict:
        """Create a new processing job."""
        doc = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "source_file": source_file,
            "file_hash": file_hash,
            "total_pages": 0,
            "total_blocks": 0,
            "total_chunks": 0,
            "progress": {"pages_done": 0, "blocks_saved": 0, "chunks_saved": 0, "embeddings_done": 0},
            "created_at": datetime.now(timezone.utc),
            "finalized_at": None,
            "error": None,
        }
        await self.jobs.insert_one(doc)
        return doc

    async def get_job(self, tenant_id: str, job_id: str) -> Optional[dict]:
        """Get a job by tenant + job_id."""
        return await self.jobs.find_one(
            {"tenant_id": tenant_id, "job_id": job_id}, {"_id": 0}
        )

    async def list_jobs(
        self, tenant_id: str, status: Optional[str] = None,
        skip: int = 0, limit: int = 50
    ) -> tuple[list[dict], int]:
        """List jobs for a tenant with optional status filter."""
        query = {"tenant_id": tenant_id}
        if status:
            query["status"] = status
        total = await self.jobs.count_documents(query)
        cursor = self.jobs.find(query, {"_id": 0}).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return docs, total

    async def update_job(self, tenant_id: str, job_id: str, updates: dict) -> bool:
        """Update job fields."""
        result = await self.jobs.update_one(
            {"tenant_id": tenant_id, "job_id": job_id},
            {"$set": updates},
        )
        return result.modified_count > 0

    async def delete_job(self, tenant_id: str, job_id: str) -> None:
        """Delete a job and all associated data."""
        query = {"tenant_id": tenant_id, "job_id": job_id}
        await self.blocks.delete_many(query)
        await self.chunks.delete_many(query)
        await self.block_revisions.delete_many(query)
        await self.chunk_revisions.delete_many(query)
        await self.index_versions.delete_many(query)
        await self.jobs.delete_one(query)

    # -----------------------------------------------------------------------
    # Blocks CRUD
    # -----------------------------------------------------------------------

    async def upsert_block(self, tenant_id: str, job_id: str, block: dict) -> None:
        """Upsert a block (idempotent for retries)."""
        block["tenant_id"] = tenant_id
        block["job_id"] = job_id
        block.setdefault("review_status", ReviewStatus.PENDING.value)
        block.setdefault("current_revision_id", None)
        block.setdefault("version", 1)
        await self.blocks.update_one(
            {"tenant_id": tenant_id, "job_id": job_id, "block_id": block["block_id"]},
            {"$set": block},
            upsert=True,
        )

    async def get_blocks(
        self, tenant_id: str, job_id: str,
        status: Optional[str] = None,
        block_type: Optional[str] = None,
        page: Optional[int] = None,
        skip: int = 0, limit: int = 100,
    ) -> list[dict]:
        """Get blocks with optional filters."""
        query: dict = {"tenant_id": tenant_id, "job_id": job_id}
        if status:
            query["review_status"] = status
        if block_type:
            query["type"] = block_type
        if page is not None:
            query["page_number"] = page
        cursor = self.blocks.find(query, {"_id": 0, "confidence": 0}).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def update_block_review(
        self, tenant_id: str, job_id: str, block_id: str,
        review_status: str, version: int,
        edited_text: Optional[str] = None,
        edited_type: Optional[str] = None,
        revision_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Update block review status with optimistic locking."""
        updates: dict = {
            "review_status": review_status,
            "current_revision_id": revision_id,
        }
        if edited_text is not None:
            updates["edited_text"] = edited_text
        if edited_type is not None:
            updates["edited_type"] = edited_type

        result = await self.blocks.find_one_and_update(
            {
                "tenant_id": tenant_id,
                "job_id": job_id,
                "block_id": block_id,
                "version": version,
            },
            {"$set": updates, "$inc": {"version": 1}},
            return_document=True,
            projection={"_id": 0, "confidence": 0},
        )
        return result

    # -----------------------------------------------------------------------
    # Chunks CRUD
    # -----------------------------------------------------------------------

    async def upsert_chunk(self, tenant_id: str, job_id: str, chunk: dict) -> None:
        """Upsert a chunk (idempotent for retries)."""
        chunk["tenant_id"] = tenant_id
        chunk["job_id"] = job_id
        chunk.setdefault("review_status", ReviewStatus.PENDING.value)
        chunk.setdefault("current_revision_id", None)
        chunk.setdefault("version", 1)
        await self.chunks.update_one(
            {"tenant_id": tenant_id, "job_id": job_id, "chunk_id": chunk["chunk_id"]},
            {"$set": chunk},
            upsert=True,
        )

    async def get_chunks(
        self, tenant_id: str, job_id: str,
        status: Optional[str] = None,
        chunk_type: Optional[str] = None,
        skip: int = 0, limit: int = 100,
    ) -> list[dict]:
        """Get chunks with optional filters."""
        query: dict = {"tenant_id": tenant_id, "job_id": job_id}
        if status:
            query["review_status"] = status
        if chunk_type:
            query["chunk_type"] = chunk_type
        cursor = self.chunks.find(query, {"_id": 0}).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def update_chunk_review(
        self, tenant_id: str, job_id: str, chunk_id: str,
        review_status: str, version: int,
        edited_text: Optional[str] = None,
        revision_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Update chunk review status with optimistic locking."""
        updates: dict = {
            "review_status": review_status,
            "current_revision_id": revision_id,
        }
        if edited_text is not None:
            updates["edited_text"] = edited_text

        result = await self.chunks.find_one_and_update(
            {
                "tenant_id": tenant_id,
                "job_id": job_id,
                "chunk_id": chunk_id,
                "version": version,
            },
            {"$set": updates, "$inc": {"version": 1}},
            return_document=True,
            projection={"_id": 0},
        )
        return result

    # -----------------------------------------------------------------------
    # Revisions (append-only)
    # -----------------------------------------------------------------------

    async def create_revision(self, tenant_id: str, job_id: str, revision: Revision) -> None:
        """Append a revision record."""
        doc = revision.model_dump()
        doc["tenant_id"] = tenant_id
        doc["job_id"] = job_id
        collection = (
            self.block_revisions if revision.entity_type == "block"
            else self.chunk_revisions
        )
        await collection.insert_one(doc)

    async def get_audit_trail(
        self, tenant_id: str, job_id: str, skip: int = 0, limit: int = 200
    ) -> list[dict]:
        """Get combined revision history for a job."""
        block_revs = await self.block_revisions.find(
            {"tenant_id": tenant_id, "job_id": job_id}, {"_id": 0}
        ).sort("timestamp", 1).to_list(length=limit)
        chunk_revs = await self.chunk_revisions.find(
            {"tenant_id": tenant_id, "job_id": job_id}, {"_id": 0}
        ).sort("timestamp", 1).to_list(length=limit)

        combined = sorted(
            block_revs + chunk_revs,
            key=lambda r: r.get("timestamp", datetime.min),
        )
        return combined[skip:skip + limit]

    # -----------------------------------------------------------------------
    # Review progress
    # -----------------------------------------------------------------------

    async def get_review_progress(self, tenant_id: str, job_id: str) -> ReviewProgress:
        """Count blocks by review status."""
        pipeline = [
            {"$match": {"tenant_id": tenant_id, "job_id": job_id}},
            {"$group": {"_id": "$review_status", "count": {"$sum": 1}}},
        ]
        result = ReviewProgress()
        async for doc in self.blocks.aggregate(pipeline):
            status = doc["_id"]
            count = doc["count"]
            if status == "approved":
                result.approved = count
            elif status == "edited":
                result.edited = count
            elif status == "rejected":
                result.rejected = count
            elif status == "pending":
                result.pending = count
        return result

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------

    async def apply_finalize_policy(
        self, tenant_id: str, job_id: str, policy: FinalizePolicy
    ) -> int:
        """Apply finalize policy to pending blocks/chunks. Returns count affected."""
        query = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "review_status": ReviewStatus.PENDING.value,
        }

        if policy == FinalizePolicy.REQUIRE_ALL_APPROVED:
            count = await self.blocks.count_documents(query)
            count += await self.chunks.count_documents(query)
            return count  # caller checks if > 0 → 400

        new_status = (
            ReviewStatus.REJECTED.value
            if policy == FinalizePolicy.REJECT_PENDING
            else ReviewStatus.APPROVED.value
        )

        r1 = await self.blocks.update_many(query, {"$set": {"review_status": new_status}})
        r2 = await self.chunks.update_many(query, {"$set": {"review_status": new_status}})
        return r1.modified_count + r2.modified_count

    async def get_approved_chunks(self, tenant_id: str, job_id: str) -> list[dict]:
        """Get all approved/edited chunks for embedding."""
        return await self.chunks.find(
            {
                "tenant_id": tenant_id,
                "job_id": job_id,
                "review_status": {"$in": [
                    ReviewStatus.APPROVED.value,
                    ReviewStatus.EDITED.value,
                ]},
            },
            {"_id": 0},
        ).to_list(length=10000)  # Cap: embedding batches

    # -----------------------------------------------------------------------
    # Index versions
    # -----------------------------------------------------------------------

    async def create_index_version(
        self, tenant_id: str, job_id: str, index_version: str, config: dict
    ) -> None:
        """Create an immutable index version record."""
        doc = {
            "tenant_id": tenant_id,
            "job_id": job_id,
            "index_version": index_version,
            "status": "embedding",
            "created_at": datetime.now(timezone.utc),
            **config,
        }
        await self.index_versions.update_one(
            {"tenant_id": tenant_id, "job_id": job_id, "index_version": index_version},
            {"$set": doc},
            upsert=True,
        )

    async def get_latest_index_version(
        self, tenant_id: str, job_id: str
    ) -> Optional[dict]:
        """Get the latest successful index version for a job."""
        cursor = self.index_versions.find(
            {"tenant_id": tenant_id, "job_id": job_id, "status": "indexed"},
            projection={"_id": 0}
        ).sort("created_at", -1).limit(1)
        docs = await cursor.to_list(length=1)
        return docs[0] if docs else None

    async def list_index_versions(self, tenant_id: str, job_id: str) -> list[dict]:
        """List all index versions for a job (for cleanup on delete)."""
        return await self.index_versions.find(
            {"tenant_id": tenant_id, "job_id": job_id}, {"_id": 0}
        ).to_list(length=100)  # Cap: index versions per job

    # -----------------------------------------------------------------------
    # Chat Sessions
    # -----------------------------------------------------------------------

    async def create_chat_session(
        self, tenant_id: str, session_id: str, job_id: str
    ) -> dict:
        """Create a new chat session (server-generated session_id)."""
        doc = {
            "tenant_id": tenant_id,
            "session_id": session_id,
            "job_id": job_id,
            "turn_count": 0,
            "rolling_summary": "",
            "long_term_facts": [],
            "version": 1,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "deleted_at": None,
        }
        await self.chat_sessions.insert_one(doc)
        return doc

    async def get_chat_session(
        self, tenant_id: str, session_id: str
    ) -> Optional[dict]:
        """Get a chat session (excludes soft-deleted)."""
        return await self.chat_sessions.find_one(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "deleted_at": None,
            },
            {"_id": 0},
        )

    async def soft_delete_chat_session(
        self, tenant_id: str, session_id: str
    ) -> bool:
        """Soft-delete a session (sets deleted_at, excluded from queries)."""
        result = await self.chat_sessions.update_one(
            {"tenant_id": tenant_id, "session_id": session_id, "deleted_at": None},
            {"$set": {"deleted_at": datetime.now(timezone.utc)}},
        )
        return result.modified_count > 0

    async def update_rolling_summary(
        self, tenant_id: str, session_id: str, summary: str, version: int
    ) -> bool:
        """Update rolling summary with optimistic locking."""
        result = await self.chat_sessions.update_one(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "version": version,
            },
            {
                "$set": {
                    "rolling_summary": summary,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"version": 1},
            },
        )
        return result.modified_count > 0

    async def update_long_term_facts(
        self, tenant_id: str, session_id: str, facts: list[dict], version: int
    ) -> bool:
        """Update long-term facts with optimistic locking."""
        result = await self.chat_sessions.update_one(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "version": version,
            },
            {
                "$set": {
                    "long_term_facts": facts,
                    "updated_at": datetime.now(timezone.utc),
                },
                "$inc": {"version": 1},
            },
        )
        return result.modified_count > 0

    # -----------------------------------------------------------------------
    # Chat Turns
    # -----------------------------------------------------------------------

    async def save_turn(
        self, tenant_id: str, session_id: str, turn
    ) -> None:
        """Save a turn and atomically increment turn_count."""
        doc = turn.model_dump(mode="json")
        doc["tenant_id"] = tenant_id
        doc["session_id"] = session_id
        await self.chat_turns.insert_one(doc)

        # Atomic turn count increment
        await self.chat_sessions.update_one(
            {"tenant_id": tenant_id, "session_id": session_id},
            {
                "$inc": {"turn_count": 1},
                "$set": {"updated_at": datetime.now(timezone.utc)},
            },
        )

    async def get_recent_turns(
        self, tenant_id: str, session_id: str, n: int = 8
    ) -> list[dict]:
        """Get the N most recent non-archived turns (short-term memory)."""
        cursor = self.chat_turns.find(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "archived": False,
            },
            {"_id": 0},
        ).sort("created_at", -1).limit(n)
        turns = await cursor.to_list(length=n)
        turns.reverse()  # chronological order
        return turns

    async def get_turn_by_idempotency_key(
        self, tenant_id: str, session_id: str, idempotency_key: str
    ) -> Optional[dict]:
        """Check for existing turn with same idempotency key (exactly-once)."""
        return await self.chat_turns.find_one(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "idempotency_key": idempotency_key,
            },
            {"_id": 0},
        )

    async def get_all_turns(
        self, tenant_id: str, session_id: str
    ) -> list[dict]:
        """Get all turns for a session (including archived), for session history."""
        cursor = self.chat_turns.find(
            {"tenant_id": tenant_id, "session_id": session_id},
            {"_id": 0},
        ).sort("created_at", 1)
        return await cursor.to_list(length=5000)  # Cap: session history

    async def get_unarchived_turns(
        self, tenant_id: str, session_id: str
    ) -> list[dict]:
        """Get all non-archived turns (for summarization)."""
        cursor = self.chat_turns.find(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "archived": False,
            },
            {"_id": 0},
        ).sort("created_at", 1)
        return await cursor.to_list(length=5000)  # Cap: summarization batch

    async def archive_turns(
        self, tenant_id: str, session_id: str, turn_ids: list[str]
    ) -> int:
        """Mark turns as archived (excluded from prompt, kept in DB)."""
        result = await self.chat_turns.update_many(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "turn_id": {"$in": turn_ids},
            },
            {"$set": {"archived": True}},
        )
        return result.modified_count

    async def purge_turns_for_session(
        self, tenant_id: str, session_id: str
    ) -> int:
        """Hard-delete all turns for a session (used by purge worker)."""
        result = await self.chat_turns.delete_many(
            {"tenant_id": tenant_id, "session_id": session_id}
        )
        return result.deleted_count

    async def get_expired_sessions(
        self, ttl_days: int = 30
    ) -> list[dict]:
        """Find soft-deleted sessions past the retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        cursor = self.chat_sessions.find(
            {"deleted_at": {"$lte": cutoff}},
            {"session_id": 1, "tenant_id": 1, "_id": 0},
        )
        return await cursor.to_list(length=1000)  # Cap: purge batch

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def close(self) -> None:
        """Close the database connection."""
        self.client.close()

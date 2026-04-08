"""ARQ worker tasks for LongParser — extraction and embedding.

All tasks are idempotent: upserts by (tenant_id, job_id, block_id / chunk_id).
Workers check job.status == 'cancelled' between steps.

Start with:
    uv run arq longparser.server.worker.WorkerSettings
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


async def extract_job(ctx: dict, tenant_id: str, job_id: str, file_path: str) -> dict:
    """Extract blocks and chunks from a document.

    Steps:
      1. Check if already extracted (idempotent)
      2. Run LongParser pipeline
      3. Chunk the result
      4. Upsert blocks + chunks into MongoDB
      5. Update job status → ready_for_review
    """
    from .db import Database
    from ..pipeline import PipelineOrchestrator
    from ..schemas import ProcessingConfig, ChunkingConfig

    db = Database()

    try:
        # Check if job is still valid
        job = await db.get_job(tenant_id, job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return {"error": "job_not_found"}
        if job["status"] in ("cancelled", "ready_for_review", "finalized", "indexed"):
            logger.info(f"Job {job_id} already {job['status']}, skipping extraction")
            return {"status": job["status"]}

        # Update status → extracting
        await db.update_job(tenant_id, job_id, {"status": "extracting"})

        # Run pipeline
        import os
        do_ocr = os.getenv("LONGPARSER_DO_OCR", "true").lower() in ("true", "1", "yes")
        formula_ocr = os.getenv("LONGPARSER_FORMULA_OCR", "true").lower() in ("true", "1", "yes")
        logger.info(f"[Worker] Extracting {file_path} for job {job_id} (OCR={'on' if do_ocr else 'off'}, formula={'on' if formula_ocr else 'off'})")
        pipeline = PipelineOrchestrator()
        result = pipeline.process_file(Path(file_path), config=ProcessingConfig(do_ocr=do_ocr, formula_ocr=formula_ocr))

        # Check cancellation
        job = await db.get_job(tenant_id, job_id)
        if job and job["status"] == "cancelled":
            logger.info(f"Job {job_id} cancelled during extraction")
            return {"status": "cancelled"}

        # Upsert blocks into MongoDB (confidence included internally, excluded from API)
        block_count = 0
        for page in result.document.pages:
            for block in page.blocks:
                block_doc = block.model_dump(mode="json")
                block_doc["page_number"] = page.page_number
                # Compute text hash for change detection
                block_doc["text_hash"] = hashlib.sha256(
                    block.text.encode()
                ).hexdigest()[:16]
                await db.upsert_block(tenant_id, job_id, block_doc)
                block_count += 1

        # Chunk
        logger.info(f"[Worker] Chunking {block_count} blocks for job {job_id}")
        from ..chunkers import HybridChunker
        chunker = HybridChunker(ChunkingConfig())
        all_blocks = [
            block for page in result.document.pages for block in page.blocks
        ]
        chunks = chunker.chunk(all_blocks)

        # Upsert chunks
        chunk_count = 0
        for chunk in chunks:
            chunk_doc = chunk.model_dump(mode="json")
            chunk_doc["text_hash"] = hashlib.sha256(
                chunk.text.encode()
            ).hexdigest()[:16]
            await db.upsert_chunk(tenant_id, job_id, chunk_doc)
            chunk_count += 1

        # Update job
        await db.update_job(tenant_id, job_id, {
            "status": "ready_for_review",
            "total_pages": len(result.document.pages),
            "total_blocks": block_count,
            "total_chunks": chunk_count,
            "progress": {
                "pages_done": len(result.document.pages),
                "blocks_saved": block_count,
                "chunks_saved": chunk_count,
                "embeddings_done": 0,
            },
        })

        logger.info(
            f"[Worker] Job {job_id} done: {block_count} blocks, {chunk_count} chunks"
        )
        return {"status": "ready_for_review", "blocks": block_count, "chunks": chunk_count}

    except Exception as e:
        logger.exception(f"[Worker] Job {job_id} failed: {e}")
        await db.update_job(tenant_id, job_id, {
            "status": "failed",
            "error": str(e),
        })
        return {"error": str(e)}
    finally:
        await db.close()


async def embed_job(
    ctx: dict, tenant_id: str, job_id: str,
    model: str, vector_db: str, collection_name: str, index_version: str,
    provider: str = "huggingface",
) -> dict:
    """Embed approved chunks and store in vector DB.

    Steps:
      1. Load approved/edited chunks from Mongo
      2. Embed with sentence-transformers
      3. Store in chosen vector DB
      4. Update job status → indexed
    """
    from .db import Database
    from .embeddings import EmbeddingEngine
    from .vectorstores import get_vector_store

    db = Database()

    try:
        job = await db.get_job(tenant_id, job_id)
        if not job or job["status"] == "cancelled":
            return {"status": "cancelled"}

        await db.update_job(tenant_id, job_id, {"status": "embedding"})

        # Get approved chunks
        chunks = await db.get_approved_chunks(tenant_id, job_id)
        if not chunks:
            await db.update_job(tenant_id, job_id, {"status": "indexed"})
            return {"status": "indexed", "embedded": 0}

        # Check cancellation
        job = await db.get_job(tenant_id, job_id)
        if job and job["status"] == "cancelled":
            return {"status": "cancelled"}

        # Grab explicit dimension override if any
        import os
        env_dim = os.getenv("LONGPARSER_EMBED_DIMENSIONS")
        configured_dimensions = int(env_dim) if env_dim else None

        # Embed
        logger.info(f"[Worker] Embedding {len(chunks)} chunks with {provider}/{model}")
        engine = EmbeddingEngine(
            provider=provider,
            model_name=model,
            dimensions=configured_dimensions
        )
        texts = [
            c.get("edited_text") or c["text"]
            for c in chunks
        ]
        embeddings = engine.embed_chunks(texts)
        dim = len(embeddings[0]) if embeddings else 0

        # Record index version
        await db.create_index_version(tenant_id, job_id, index_version, {
            "provider": provider,
            "model": model,
            "configured_dimensions": configured_dimensions,
            "dim": dim,
            "normalize": True,
            "distance_metric": "cosine",
            "vector_db": vector_db,
            "collection": collection_name,
            "fingerprint": engine.get_fingerprint(),
        })

        # Store in vector DB
        store = get_vector_store(
            vector_db, 
            collection_name=collection_name,
            index_fingerprint=engine.get_fingerprint(),
        )

        # Deterministic vector IDs: {tenant_id}:{job_id}:{chunk_id}:{index_version}
        ids = [
            f"{tenant_id}:{job_id}:{c['chunk_id']}:{index_version}"
            for c in chunks
        ]
        metadatas = [
            {
                "tenant_id": tenant_id,
                "job_id": job_id,
                "chunk_id": c["chunk_id"],
                "chunk_type": c.get("chunk_type", ""),
                "section_path": c.get("section_path", []),
                "page_numbers": c.get("page_numbers", []),
                "block_ids": c.get("block_ids", []),
                "index_version": index_version,
                "text_hash": c.get("text_hash", ""),
            }
            for c in chunks
        ]

        store.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

        # Update index version + job
        await db.index_versions.update_one(
            {"tenant_id": tenant_id, "job_id": job_id, "index_version": index_version},
            {"$set": {"status": "indexed"}},
        )
        await db.update_job(tenant_id, job_id, {
            "status": "indexed",
            "progress.embeddings_done": len(chunks),
        })

        logger.info(f"[Worker] Job {job_id} indexed: {len(chunks)} vectors in {vector_db}")
        return {"status": "indexed", "embedded": len(chunks)}

    except Exception as e:
        logger.exception(f"[Worker] Embed job {job_id} failed: {e}")
        await db.update_job(tenant_id, job_id, {
            "status": "failed",
            "error": str(e),
        })
        return {"error": str(e)}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Chat Background Tasks
# ---------------------------------------------------------------------------

async def summarize_session(ctx: dict, tenant_id: str, session_id: str) -> dict:
    """Compress older turns into a rolling summary (mid-term memory).

    Steps:
      1. Get session + unarchived turns
      2. Keep last N as short-term; summarize the rest
      3. Update rolling_summary with optimistic lock
      4. Archive summarized turns
    """
    from .db import Database
    from .chat.schemas import ChatConfig
    from .chat.llm_chain import get_plain_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage

    db = Database()
    config = ChatConfig()
    llm = get_plain_chat_model(config=config)

    try:
        session = await db.get_chat_session(tenant_id, session_id)
        if not session:
            return {"error": "session_not_found"}

        turns = await db.get_unarchived_turns(tenant_id, session_id)
        if len(turns) <= config.short_term_turns:
            return {"status": "skipped", "reason": "not enough turns"}

        # Keep last N as short-term, summarize the rest
        to_summarize = turns[:-config.short_term_turns]
        if not to_summarize:
            return {"status": "skipped", "reason": "nothing to summarize"}

        # Build summarization prompt
        existing_summary = session.get("rolling_summary", "")
        turn_text = "\n".join(
            f"User: {t['question']}\nAssistant: {t['answer']}"
            for t in to_summarize
        )
        messages = [
            SystemMessage(content="You are a conversation summarizer. Produce a concise summary that preserves key facts, decisions, and context. Return plain text, no JSON."),
            HumanMessage(content=f"Existing summary:\n{existing_summary}\n\nNew turns to incorporate:\n{turn_text}\n\nProduce an updated summary:"),
        ]

        response = await llm.ainvoke(messages)
        new_summary = response.content

        # Update with optimistic lock
        updated = await db.update_rolling_summary(
            tenant_id, session_id, new_summary, session["version"]
        )
        if not updated:
            logger.warning(f"[Worker] Summary version conflict for session {session_id}")
            return {"status": "conflict"}

        # Archive summarized turns
        turn_ids = [t["turn_id"] for t in to_summarize]
        archived = await db.archive_turns(tenant_id, session_id, turn_ids)

        logger.info(f"[Worker] Summarized session {session_id}: {archived} turns archived")
        return {"status": "summarized", "archived": archived}

    except Exception as e:
        logger.exception(f"[Worker] Summarize session {session_id} failed: {e}")
        return {"error": str(e)}
    finally:
        await db.close()


async def extract_facts(
    ctx: dict, tenant_id: str, session_id: str, job_id: str
) -> dict:
    """Extract long-term facts from recent conversation (Layer 3 memory).

    Only persists facts from allowlisted types with chunk provenance.
    """
    from .db import Database
    from .chat.schemas import ChatConfig, FactSourceType
    from .chat.llm_chain import get_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage

    db = Database()
    config = ChatConfig()
    llm = get_chat_model(config=config, json_mode=False)

    ALLOWED_FACT_TYPES = {"entities_from_doc", "user_preferences", "decisions"}

    try:
        session = await db.get_chat_session(tenant_id, session_id)
        if not session:
            return {"error": "session_not_found"}

        turns = await db.get_recent_turns(tenant_id, session_id, n=20)
        if not turns:
            return {"status": "skipped", "reason": "no turns"}

        turn_text = "\n".join(
            f"User: {t['question']}\nAssistant: {t['answer']}"
            for t in turns
        )
        messages = [
            SystemMessage(content=(
                "Extract key facts from this conversation. Return JSON:\n"
                '{"facts": [{"type": "entities_from_doc"|"user_preferences"|"decisions", '
                '"source": "doc"|"user", "fact": "...", "confidence": 0.0-1.0}]}\n'
                "Only extract facts clearly stated in the conversation. "
                "Do NOT infer or guess. Maximum 10 facts."
            )),
            HumanMessage(content=turn_text),
        ]

        response = await llm.ainvoke(messages)
        raw = response.content

        import json
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"error": "invalid_json"}

        # Filter: only allowlisted types, only doc/user sources
        existing_facts = session.get("long_term_facts", [])
        new_facts = []
        for f in data.get("facts", []):
            if f.get("type") not in ALLOWED_FACT_TYPES:
                continue
            if f.get("source") not in ("doc", "user"):
                continue
            new_facts.append({
                "type": f["type"],
                "source": f["source"],
                "fact": f["fact"],
                "supporting_chunk_ids": [],
                "confidence": f.get("confidence", 0.5),
            })

        # Merge + cap at max_facts
        merged = existing_facts + new_facts
        merged = merged[-config.max_facts:]

        updated = await db.update_long_term_facts(
            tenant_id, session_id, merged, session["version"]
        )
        if not updated:
            logger.warning(f"[Worker] Facts version conflict for session {session_id}")
            return {"status": "conflict"}

        logger.info(f"[Worker] Extracted {len(new_facts)} facts for session {session_id}")
        return {"status": "extracted", "new_facts": len(new_facts), "total": len(merged)}

    except Exception as e:
        logger.exception(f"[Worker] Extract facts {session_id} failed: {e}")
        return {"error": str(e)}
    finally:
        await db.close()


async def purge_expired_sessions(ctx: dict) -> dict:
    """Scheduled task: hard-delete turns for soft-deleted sessions past TTL."""
    from .db import Database
    from .chat.schemas import ChatConfig

    db = Database()
    config = ChatConfig()

    try:
        expired = await db.get_expired_sessions(config.ttl_days)
        purged = 0
        for session in expired:
            count = await db.purge_turns_for_session(
                session["tenant_id"], session["session_id"]
            )
            await db.chat_sessions.delete_one({
                "tenant_id": session["tenant_id"],
                "session_id": session["session_id"],
            })
            purged += count

        if purged > 0:
            logger.info(f"[Worker] Purged {purged} turns from {len(expired)} expired sessions")
        return {"status": "purged", "sessions": len(expired), "turns": purged}

    except Exception as e:
        logger.exception(f"[Worker] Purge expired sessions failed: {e}")
        return {"error": str(e)}
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# ARQ Worker Settings
# ---------------------------------------------------------------------------

class WorkerSettings:
    """ARQ worker configuration — start with `arq longparser.server.worker.WorkerSettings`."""

    functions = [
        extract_job,
        embed_job,
        summarize_session,
        extract_facts,
        purge_expired_sessions,
    ]
    
    # 10-min timeout: ~72s Docling + up to 420s formula OCR + headroom
    job_timeout = 420
    import os
    from arq.connections import RedisSettings
    _redis_url = os.getenv("LONGPARSER_REDIS_URL", "redis://localhost:6379/0")
    redis_settings = RedisSettings.from_dsn(_redis_url)

    # Scheduled cron tasks
    cron_jobs = None  # set below after import

    @staticmethod
    async def on_startup(ctx: dict) -> None:
        logger.info("[ARQ Worker] Starting up")

    @staticmethod
    async def on_shutdown(ctx: dict) -> None:
        logger.info("[ARQ Worker] Shutting down")


# Cron: purge expired sessions once per hour
try:
    from arq import cron
    WorkerSettings.cron_jobs = [
        cron(purge_expired_sessions, hour=None, minute=0),  # every hour at :00
    ]
except ImportError:
    pass  # arq cron not available in all versions

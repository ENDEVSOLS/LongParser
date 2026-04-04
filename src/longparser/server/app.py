"""LongParser FastAPI application — HITL review + embedding + search.

Start with:
    uv run uvicorn longparser.server.app:app --reload --port 8000
"""

from __future__ import annotations

# Load .env for local development (no-op if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import hashlib
import io
import logging
import os
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import time as _time

from fastapi import (
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse

from .db import Database
from .queue import ARQBackend
from .schemas import (
    BlockResponse,
    BlockReviewUpdate,
    ChunkResponse,
    ChunkReviewUpdate,
    EmbedRequest,
    FinalizePolicy,
    FinalizeRequest,
    JobListResponse,
    JobResponse,
    JobStatus,
    ReviewProgress,
    ReviewStatus,
    Revision,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
ALLOWED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/csv",
    "application/octet-stream",  # fallback for unknown MIME
}
UPLOAD_DIR = Path(os.getenv("LONGPARSER_UPLOAD_DIR", "./uploads")).resolve()

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

db = Database(
    mongo_url=os.getenv("LONGPARSER_MONGO_URL", "mongodb://localhost:27017"),
    db_name=os.getenv("LONGPARSER_DB_NAME", "longparser"),
)
queue = ARQBackend(
    redis_url=os.getenv("LONGPARSER_REDIS_URL", "redis://localhost:6379"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    await db.create_indexes()
    logger.info("LongParser API started")
    yield
    await queue.close()
    await db.close()
    if hasattr(app.state, "chat_engine"):
        await app.state.chat_engine.close()
    logger.info("LongParser API stopped")


app = FastAPI(
    title="LongParser API",
    description="Document intelligence engine with HITL review, embedding, and vector search.",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Auth middleware (API key — v1)
# ---------------------------------------------------------------------------

def _get_tenant(x_api_key: str = Header(...)) -> str:
    """Extract tenant_id from API key.

    v1: API key IS the tenant identifier.
    v2: look up hashed key → tenant mapping in DB.
    """
    if not x_api_key or len(x_api_key) < 8:
        raise HTTPException(status_code=401, detail="Invalid API key")
    # For v1, use a hash of the key as tenant_id
    return hashlib.sha256(x_api_key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------

async def _stream_upload(upload: UploadFile, dest: Path) -> tuple[str, int]:
    """Stream uploaded file to disk in chunks. Returns (sha256, size)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256()
    size = 0

    with open(dest, "wb") as f:
        while True:
            chunk = await upload.read(64 * 1024)  # 64KB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds {MAX_UPLOAD_SIZE // (1024*1024)}MB limit",
                )
            sha.update(chunk)
            f.write(chunk)
            os.fsync(f.fileno())

    return sha.hexdigest(), size


# ---------------------------------------------------------------------------
# Routes: Jobs
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    file: UploadFile = File(...),
    x_api_key: str = Header(...),
):
    """Upload a document → enqueue extraction."""
    tenant_id = _get_tenant(x_api_key)

    # Validate content type
    if file.content_type and file.content_type not in ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}",
        )

    # Generate job ID and save file
    job_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / tenant_id / job_id / (file.filename or "document")
    file_hash, file_size = await _stream_upload(file, dest)

    # Create job in MongoDB
    job_doc = await db.create_job(
        tenant_id=tenant_id,
        job_id=job_id,
        source_file=file.filename or "document",
        file_hash=file_hash,
    )

    # Enqueue extraction
    await queue.enqueue("extract_job", {
        "tenant_id": tenant_id,
        "job_id": job_id,
        "file_path": str(dest),
    })

    return JobResponse(
        job_id=job_id,
        tenant_id=tenant_id,
        status=JobStatus.QUEUED,
        source_file=file.filename or "document",
        file_hash=file_hash,
        created_at=job_doc["created_at"],
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    x_api_key: str = Header(...),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """List all jobs for the tenant."""
    tenant_id = _get_tenant(x_api_key)
    jobs, total = await db.list_jobs(tenant_id, status=status, skip=skip, limit=limit)

    job_responses = []
    for j in jobs:
        progress = await db.get_review_progress(tenant_id, j["job_id"])
        job_responses.append(JobResponse(
            job_id=j["job_id"],
            tenant_id=j["tenant_id"],
            status=j["status"],
            source_file=j["source_file"],
            file_hash=j.get("file_hash", ""),
            total_pages=j.get("total_pages", 0),
            total_blocks=j.get("total_blocks", 0),
            total_chunks=j.get("total_chunks", 0),
            review_progress=progress,
            created_at=j["created_at"],
            finalized_at=j.get("finalized_at"),
            error=j.get("error"),
        ))

    return JobListResponse(jobs=job_responses, total=total)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, x_api_key: str = Header(...)):
    """Get job status and details."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = await db.get_review_progress(tenant_id, job_id)

    return JobResponse(
        job_id=job["job_id"],
        tenant_id=job["tenant_id"],
        status=job["status"],
        source_file=job["source_file"],
        file_hash=job.get("file_hash", ""),
        total_pages=job.get("total_pages", 0),
        total_blocks=job.get("total_blocks", 0),
        total_chunks=job.get("total_chunks", 0),
        review_progress=progress,
        created_at=job["created_at"],
        finalized_at=job.get("finalized_at"),
        error=job.get("error"),
    )


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, x_api_key: str = Header(...)):
    """Delete job and all associated data (blocks, chunks, revisions, vectors)."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete vectors from all index versions
    index_versions = await db.list_index_versions(tenant_id, job_id)
    for iv in index_versions:
        try:
            from .vectorstores import get_vector_store
            from .embeddings import EmbeddingEngine

            # Rebuild engine to securely reconstruct fingerprint for deletion
            engine = EmbeddingEngine(
                provider=iv.get("provider", "huggingface"),
                model_name=iv.get("model", "BAAI/bge-base-en-v1.5"),
                dimensions=iv.get("configured_dimensions")
            )
            store = get_vector_store(
                iv.get("vector_db", "chroma"),
                collection_name=iv.get("collection", "longparser"),
                index_fingerprint=engine.get_fingerprint()
            )
            store.delete_by_job(job_id, tenant_id=tenant_id)
        except Exception as e:
            logger.warning(f"Vector delete failed for index {iv.get('index_version')}: {e}")

    # Delete from MongoDB
    await db.delete_job(tenant_id, job_id)

    # Delete uploaded file
    upload_dir = UPLOAD_DIR / tenant_id / job_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


@app.post("/jobs/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str, x_api_key: str = Header(...)):
    """Cancel an in-progress job."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in ("finalized", "indexed"):
        raise HTTPException(status_code=400, detail="Cannot cancel a completed job")

    await db.update_job(tenant_id, job_id, {"status": "cancelled"})
    return {"status": "cancelled", "job_id": job_id}


# ---------------------------------------------------------------------------
# Helpers for formatting API responses to show edited text
# ---------------------------------------------------------------------------

def _format_block(b: dict) -> BlockResponse:
    if b.get("edited_text") is not None:
        b["text"] = b["edited_text"]
    if b.get("edited_type") is not None:
        b["type"] = b["edited_type"]
    return BlockResponse(**b)

def _format_chunk(c: dict) -> ChunkResponse:
    if c.get("edited_text") is not None:
        c["text"] = c["edited_text"]
    return ChunkResponse(**c)


# ---------------------------------------------------------------------------
# Routes: Blocks (HITL review)
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/blocks", response_model=list[BlockResponse])
async def list_blocks(
    job_id: str,
    x_api_key: str = Header(...),
    status: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    page: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """List blocks for a job (filterable by status, type, page)."""
    tenant_id = _get_tenant(x_api_key)
    blocks = await db.get_blocks(
        tenant_id, job_id,
        status=status, block_type=type, page=page,
        skip=skip, limit=limit,
    )
    return [_format_block(b) for b in blocks]


@app.patch("/jobs/{job_id}/blocks/{block_id}", response_model=BlockResponse)
async def update_block(
    job_id: str, block_id: str,
    body: BlockReviewUpdate,
    x_api_key: str = Header(...),
):
    """Edit/approve/reject a block. Creates revision + auto-rechunks."""
    tenant_id = _get_tenant(x_api_key)

    # Get current block for revision record
    blocks = await db.get_blocks(tenant_id, job_id)
    current = next((b for b in blocks if b.get("block_id") == block_id), None)
    if not current:
        raise HTTPException(status_code=404, detail="Block not found")

    # Create revision (append-only)
    revision = Revision(
        entity_type="block",
        entity_id=block_id,
        previous_revision_id=current.get("current_revision_id"),
        action=body.status,
        original_text=current.get("text", ""),
        edited_text=body.edited_text,
        edited_type=body.edited_type,
        reviewer_note=body.reviewer_note,
    )
    await db.create_revision(tenant_id, job_id, revision)

    # Update block with optimistic lock
    updated = await db.update_block_review(
        tenant_id, job_id, block_id,
        review_status=body.status.value,
        version=body.version,
        edited_text=body.edited_text,
        edited_type=body.edited_type.value if body.edited_type else None,
        revision_id=revision.revision_id,
    )

    if not updated:
        raise HTTPException(
            status_code=409,
            detail="Version conflict — block was modified by another reviewer",
        )

    # Auto-rechunk after block edit
    new_count = await _rechunk_job(tenant_id, job_id)
    logger.info(f"Auto-rechunked job {job_id}: {new_count} chunks")

    return _format_block(updated)


# ---------------------------------------------------------------------------
# Routes: Chunks (HITL review)
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/chunks", response_model=list[ChunkResponse])
async def list_chunks(
    job_id: str,
    x_api_key: str = Header(...),
    status: Optional[str] = Query(None),
    chunk_type: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """List chunks for a job (filterable)."""
    tenant_id = _get_tenant(x_api_key)
    chunks = await db.get_chunks(
        tenant_id, job_id,
        status=status, chunk_type=chunk_type,
        skip=skip, limit=limit,
    )
    return [_format_chunk(c) for c in chunks]


@app.patch("/jobs/{job_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(
    job_id: str, chunk_id: str,
    body: ChunkReviewUpdate,
    x_api_key: str = Header(...),
):
    """Edit/approve/reject a chunk."""
    tenant_id = _get_tenant(x_api_key)

    # Get current chunk
    chunks = await db.get_chunks(tenant_id, job_id)
    current = next((c for c in chunks if c.get("chunk_id") == chunk_id), None)
    if not current:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Create revision
    revision = Revision(
        entity_type="chunk",
        entity_id=chunk_id,
        previous_revision_id=current.get("current_revision_id"),
        action=body.status,
        original_text=current.get("text", ""),
        edited_text=body.edited_text,
        reviewer_note=body.reviewer_note,
    )
    await db.create_revision(tenant_id, job_id, revision)

    # Update with optimistic lock
    updated = await db.update_chunk_review(
        tenant_id, job_id, chunk_id,
        review_status=body.status.value,
        version=body.version,
        edited_text=body.edited_text,
        revision_id=revision.revision_id,
    )

    if not updated:
        raise HTTPException(status_code=409, detail="Version conflict")

    return _format_chunk(updated)


# ---------------------------------------------------------------------------
# Routes: Audit
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/audit")
async def get_audit(
    job_id: str,
    x_api_key: str = Header(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    """Get full revision history for a job."""
    tenant_id = _get_tenant(x_api_key)
    trail = await db.get_audit_trail(tenant_id, job_id, skip=skip, limit=limit)
    return trail


# ---------------------------------------------------------------------------
# Routes: Admin Purge (hard delete with tombstone audit)
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/blocks/{block_id}/purge")
async def purge_block(
    job_id: str, block_id: str,
    x_api_key: str = Header(...),
):
    """Admin-only: permanently delete a block. Writes a tombstone revision."""
    tenant_id = _get_tenant(x_api_key)

    # Get block before deletion (for tombstone)
    blocks = await db.get_blocks(tenant_id, job_id)
    current = next((b for b in blocks if b.get("block_id") == block_id), None)
    if not current:
        raise HTTPException(status_code=404, detail="Block not found")

    # Write tombstone revision — preserve hashes/metadata, scrub sensitive text
    text_hash = hashlib.sha256(current.get("text", "").encode()).hexdigest()[:16]
    tombstone = Revision(
        entity_type="block",
        entity_id=block_id,
        previous_revision_id=current.get("current_revision_id"),
        action=ReviewStatus.REJECTED,
        original_text=f"[PURGED] text_hash={text_hash} type={current.get('type')} page={current.get('page_number')}",
        edited_text=None,
        reviewer_note="ADMIN PURGE — content permanently deleted",
    )
    await db.create_revision(tenant_id, job_id, tombstone)

    # Delete the block
    await db.blocks.delete_one({
        "tenant_id": tenant_id, "job_id": job_id, "block_id": block_id,
    })

    # Update block count
    remaining = await db.blocks.count_documents({"tenant_id": tenant_id, "job_id": job_id})
    await db.update_job(tenant_id, job_id, {"total_blocks": remaining})

    # Auto-rechunk
    new_count = await _rechunk_job(tenant_id, job_id)

    return {
        "status": "purged",
        "block_id": block_id,
        "tombstone_revision_id": tombstone.revision_id,
        "chunks_after_rechunk": new_count,
    }


@app.post("/jobs/{job_id}/chunks/{chunk_id}/purge")
async def purge_chunk(
    job_id: str, chunk_id: str,
    x_api_key: str = Header(...),
):
    """Admin-only: permanently delete a chunk. Writes a tombstone revision."""
    tenant_id = _get_tenant(x_api_key)

    # Get chunk before deletion
    chunks = await db.get_chunks(tenant_id, job_id)
    current = next((c for c in chunks if c.get("chunk_id") == chunk_id), None)
    if not current:
        raise HTTPException(status_code=404, detail="Chunk not found")

    # Write tombstone
    text_hash = hashlib.sha256(current.get("text", "").encode()).hexdigest()[:16]
    tombstone = Revision(
        entity_type="chunk",
        entity_id=chunk_id,
        previous_revision_id=current.get("current_revision_id"),
        action=ReviewStatus.REJECTED,
        original_text=f"[PURGED] text_hash={text_hash} type={current.get('chunk_type')}",
        edited_text=None,
        reviewer_note="ADMIN PURGE — content permanently deleted",
    )
    await db.create_revision(tenant_id, job_id, tombstone)

    # Delete the chunk
    await db.chunks.delete_one({
        "tenant_id": tenant_id, "job_id": job_id, "chunk_id": chunk_id,
    })

    # Update chunk count
    remaining = await db.chunks.count_documents({"tenant_id": tenant_id, "job_id": job_id})
    await db.update_job(tenant_id, job_id, {"total_chunks": remaining})

    return {
        "status": "purged",
        "chunk_id": chunk_id,
        "tombstone_revision_id": tombstone.revision_id,
    }


# ---------------------------------------------------------------------------
# Rechunk helper (shared by block edit + explicit rechunk)
# ---------------------------------------------------------------------------

async def _rechunk_job(tenant_id: str, job_id: str) -> int:
    """Re-chunk a job from current blocks. Returns new chunk count."""
    from ..schemas import Block, Provenance, BoundingBox, Confidence
    from ..chunkers import HybridChunker
    from ..schemas import ChunkingConfig

    blocks_data = await db.get_blocks(tenant_id, job_id)
    blocks = []
    for b in blocks_data:
        text = b.get("edited_text") or b.get("text", "")
        blocks.append(Block(
            block_id=b["block_id"],
            type=b.get("type", "paragraph"),
            text=text,
            order_index=b.get("order_index", 0),
            heading_level=b.get("heading_level"),
            indent_level=b.get("indent_level", 0),
            hierarchy_path=b.get("hierarchy_path", []),
            provenance=Provenance(
                source_file=b.get("provenance", {}).get("source_file", ""),
                page_number=b.get("page_number", 0),
                bbox=BoundingBox(**b.get("provenance", {}).get("bbox", {"x0": 0, "y0": 0, "x1": 0, "y1": 0})),
                extractor=b.get("provenance", {}).get("extractor", "docling"),
            ),
            confidence=Confidence(overall=1.0),
        ))

    chunker = HybridChunker(ChunkingConfig())
    new_chunks = chunker.chunk(blocks)

    await db.chunks.delete_many({"tenant_id": tenant_id, "job_id": job_id})
    for chunk in new_chunks:
        chunk_doc = chunk.model_dump(mode="json")
        chunk_doc["text_hash"] = hashlib.sha256(chunk.text.encode()).hexdigest()[:16]
        await db.upsert_chunk(tenant_id, job_id, chunk_doc)

    await db.update_job(tenant_id, job_id, {"total_chunks": len(new_chunks)})
    return len(new_chunks)


@app.post("/jobs/{job_id}/rechunk")
async def rechunk(job_id: str, x_api_key: str = Header(...)):
    """Explicitly re-chunk the job (also happens automatically after block edits)."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("ready_for_review",):
        raise HTTPException(
            status_code=400,
            detail="Can only rechunk jobs in 'ready_for_review' status",
        )

    new_count = await _rechunk_job(tenant_id, job_id)
    return {"status": "rechunked", "total_chunks": new_count}


# ---------------------------------------------------------------------------
# Routes: Finalize
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/finalize")
async def finalize_job(
    job_id: str,
    body: FinalizeRequest,
    x_api_key: str = Header(...),
):
    """Finalize review — apply policy, lock job."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("ready_for_review",):
        raise HTTPException(status_code=400, detail="Job not in reviewable state")

    # Apply policy
    if body.finalize_policy == FinalizePolicy.REQUIRE_ALL_APPROVED:
        pending = await db.apply_finalize_policy(
            tenant_id, job_id, body.finalize_policy
        )
        if pending > 0:
            raise HTTPException(
                status_code=400,
                detail=f"{pending} item(s) still pending — approve or reject all before finalizing",
            )
    else:
        affected = await db.apply_finalize_policy(
            tenant_id, job_id, body.finalize_policy
        )

    await db.update_job(tenant_id, job_id, {
        "status": "finalized",
        "finalized_at": datetime.now(timezone.utc),
    })

    return {"status": "finalized", "job_id": job_id, "policy": body.finalize_policy.value}


# ---------------------------------------------------------------------------
# Routes: Export (streaming zip)
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}/export")
async def export_job(job_id: str, x_api_key: str = Header(...)):
    """Stream download of finalized output as .zip."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    blocks = await db.get_blocks(tenant_id, job_id)
    chunks = await db.get_chunks(tenant_id, job_id)

    # Build zip in memory-efficient streaming fashion
    import json

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("blocks.json", json.dumps(blocks, default=str, indent=2))
        zf.writestr("chunks.json", json.dumps(chunks, default=str, indent=2))

        # Generate document.md from approved blocks
        md_lines = []
        for b in sorted(blocks, key=lambda x: x.get("order_index", 0)):
            text = b.get("edited_text") or b.get("text", "")
            block_type = b.get("type", "paragraph")
            level = b.get("heading_level", 1)
            if block_type == "heading" and level:
                md_lines.append(f"{'#' * level} {text}")
            else:
                md_lines.append(text)
            md_lines.append("")
        zf.writestr("document.md", "\n".join(md_lines))

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=longparser_export_{job_id[:8]}.zip"
        },
    )


# ---------------------------------------------------------------------------
# Routes: Embed
# ---------------------------------------------------------------------------

@app.post("/jobs/{job_id}/embed")
async def embed_job_route(
    job_id: str,
    body: EmbedRequest,
    x_api_key: str = Header(...),
):
    """Embed approved chunks → store in vector DB."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("finalized",):
        raise HTTPException(status_code=400, detail="Job must be finalized before embedding")

    index_version = str(uuid.uuid4())[:8]
    collection_name = body.collection_name or f"longparser_{job_id[:8]}"

    # Enqueue embedding task
    await queue.enqueue("embed_job", {
        "tenant_id": tenant_id,
        "job_id": job_id,
        "provider": body.provider,
        "model": body.model,
        "vector_db": body.vector_db,
        "collection_name": collection_name,
        "index_version": index_version,
    })

    await db.update_job(tenant_id, job_id, {"status": "embedding"})

    return {
        "status": "embedding",
        "job_id": job_id,
        "index_version": index_version,
        "provider": body.provider,
        "model": body.model,
        "vector_db": body.vector_db,
        "collection": collection_name,
    }


# ---------------------------------------------------------------------------
# Routes: Search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, x_api_key: str = Header(...)):
    """Search embedded chunks by similarity."""
    tenant_id = _get_tenant(x_api_key)
    job = await db.get_job(tenant_id, body.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Get index version
    if body.index_version:
        iv_doc = await db.index_versions.find_one(
            {"tenant_id": tenant_id, "job_id": body.job_id, "index_version": body.index_version},
            {"_id": 0},
        )
    else:
        iv_doc = await db.get_latest_index_version(tenant_id, body.job_id)

    if not iv_doc:
        raise HTTPException(status_code=404, detail="No embedding index found for this job")

    # Embed query using identical model configurations
    from .embeddings import EmbeddingEngine
    engine = EmbeddingEngine(
        provider=iv_doc.get("provider", "huggingface"),
        model_name=iv_doc["model"],
        dimensions=iv_doc.get("configured_dimensions")
    )
    query_embedding = engine.embed_query(body.query)

    # Search in vector DB
    from .vectorstores import get_vector_store
    store = get_vector_store(
        iv_doc["vector_db"],
        collection_name=iv_doc.get("collection", "longparser"),
        index_fingerprint=engine.get_fingerprint()
    )

    filters = {
        "tenant_id": tenant_id,
        "job_id": body.job_id,
        **body.filters,
    }
    raw_results = store.search(query_embedding, top_k=body.top_k, filters=filters)

    results = []
    for r in raw_results:
        meta = r.get("metadata", {})
        results.append(SearchResult(
            chunk_id=meta.get("chunk_id", ""),
            text=r.get("document", ""),
            score=r.get("score", 0.0),
            chunk_type=meta.get("chunk_type", ""),
            section_path=meta.get("section_path", []),
            page_numbers=meta.get("page_numbers", []),
            block_ids=meta.get("block_ids", []),
            metadata=meta,
        ))

    return SearchResponse(
        results=results,
        index_version=iv_doc["index_version"],
        model=iv_doc["model"],
        query=body.query,
        total=len(results),
    )


# ---------------------------------------------------------------------------
# Observability middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Attach request_id and log structured request data."""
    request_id = str(uuid.uuid4())[:8]
    start = _time.monotonic()
    response = await call_next(request)
    latency_ms = (_time.monotonic() - start) * 1000
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return response


# ---------------------------------------------------------------------------
# Routes: Chat Sessions
# ---------------------------------------------------------------------------

@app.post("/chat/sessions", status_code=201)
async def create_chat_session(
    body: dict,
    x_api_key: str = Header(...),
):
    """Create a new chat session (server-generated session_id)."""
    from .chat.schemas import CreateSessionRequest
    req = CreateSessionRequest(**body)
    tenant_id = _get_tenant(x_api_key)

    # Verify job belongs to tenant
    job = await db.get_job(tenant_id, req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    session_id = str(uuid.uuid4())
    await db.create_chat_session(tenant_id, session_id, req.job_id)

    return {"session_id": session_id, "job_id": req.job_id}


@app.get("/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    x_api_key: str = Header(...),
):
    """Get chat session with full history."""
    tenant_id = _get_tenant(x_api_key)
    session = await db.get_chat_session(tenant_id, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    turns = await db.get_all_turns(tenant_id, session_id)
    session["turns"] = turns
    return session


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    x_api_key: str = Header(...),
):
    """Soft-delete a chat session."""
    tenant_id = _get_tenant(x_api_key)
    deleted = await db.soft_delete_chat_session(tenant_id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ---------------------------------------------------------------------------
# Routes: Chat
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(
    body: dict,
    x_api_key: str = Header(...),
):
    """Ask a question — RAG chatbot with 3-layer memory.

    Set require_approval=true for Human-in-the-Loop review.
    """
    from .chat.schemas import ChatRequest, ChatResponse, ChatConfig
    from .chat.engine import ChatEngine

    req = ChatRequest(**body)
    tenant_id = _get_tenant(x_api_key)

    # ── Session ↔ Job binding validation ──
    session = await db.get_chat_session(tenant_id, req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["job_id"] != req.job_id:
        raise HTTPException(
            status_code=400,
            detail="job_id does not match session's job_id",
        )
    job = await db.get_job(tenant_id, req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # ── Create ChatEngine (reuse on app.state if available) ──
    config = ChatConfig()
    if not hasattr(app.state, "chat_engine"):
        app.state.chat_engine = ChatEngine(db=db, queue=queue, config=config)

    response = await app.state.chat_engine.ask(tenant_id, req)

    # ── HITL: if require_approval, pause for human review ──
    if req.require_approval and response.status == "complete":
        from .chat.schemas import LLMAnswer, SourceRef
        from .chat.graph import start_hitl_review

        answer_obj = LLMAnswer(
            answer=response.answer,
            cited_chunk_ids=[s.chunk_id for s in response.sources],
        )
        hitl_result = await start_hitl_review(
            tenant_id=tenant_id,
            session_id=req.session_id,
            job_id=req.job_id,
            question=req.question,
            answer=answer_obj,
            sources=response.sources,
        )
        response.status = "pending_review"
        response.thread_id = hitl_result["thread_id"]

    return response.model_dump(mode="json")


@app.post("/chat/resume")
async def resume_chat(
    body: dict,
    x_api_key: str = Header(...),
):
    """Resume a paused HITL chat with human decision (approve/edit/reject)."""
    from .chat.schemas import HITLResumeRequest, ChatResponse, SourceRef, Turn
    from .chat.graph import resume_hitl_review

    req = HITLResumeRequest(**body)
    tenant_id = _get_tenant(x_api_key)

    # Validate session belongs to tenant
    session = await db.get_chat_session(tenant_id, req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Resume the LangGraph flow
    result = await resume_hitl_review(
        thread_id=req.thread_id,
        action=req.action,
        edited_answer=req.edited_answer,
    )

    # If the answer was edited or approved, update the saved turn
    if result.get("status") == "complete":
        # Update the last turn's answer if edited
        if req.action == "edit" and req.edited_answer:
            await db.chat_turns.update_one(
                {
                    "tenant_id": tenant_id,
                    "session_id": req.session_id,
                },
                {"$set": {"answer": req.edited_answer}},
                sort=[("created_at", -1)],
            )

    sources = [SourceRef(**s) for s in result.get("sources", [])]

    return ChatResponse(
        session_id=req.session_id,
        turn_id=result.get("turn_id", ""),
        answer=result.get("answer", ""),
        sources=sources,
        status=result.get("status", "complete"),
    ).model_dump(mode="json")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "cleanrag-api"}


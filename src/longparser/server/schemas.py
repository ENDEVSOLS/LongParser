"""API request/response models for LongParser HITL review system."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..schemas import BlockType


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Job lifecycle states."""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    READY_FOR_REVIEW = "ready_for_review"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewStatus(str, Enum):
    """Block/chunk review states."""
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class FinalizePolicy(str, Enum):
    """What to do with pending items on finalize."""
    REJECT_PENDING = "reject_pending"
    APPROVE_PENDING = "approve_pending"
    REQUIRE_ALL_APPROVED = "require_all_approved"


class UserRole(str, Enum):
    """RBAC roles."""
    ADMIN = "admin"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


# ---------------------------------------------------------------------------
# Revision (append-only audit trail)
# ---------------------------------------------------------------------------

class Revision(BaseModel):
    """Immutable record of a block/chunk edit."""
    revision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str  # "block" | "chunk"
    entity_id: str
    previous_revision_id: Optional[str] = None
    action: ReviewStatus
    original_text: str
    edited_text: Optional[str] = None
    edited_type: Optional[BlockType] = None
    reviewer_id: str = ""
    reviewer_note: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# API Request Models
# ---------------------------------------------------------------------------

class BlockReviewUpdate(BaseModel):
    """Request body for PATCH /jobs/{id}/blocks/{bid}."""
    status: ReviewStatus
    edited_text: Optional[str] = None
    edited_type: Optional[BlockType] = None
    reviewer_note: str = ""
    version: int  # optimistic locking


class ChunkReviewUpdate(BaseModel):
    """Request body for PATCH /jobs/{id}/chunks/{cid}."""
    status: ReviewStatus
    edited_text: Optional[str] = None
    reviewer_note: str = ""
    version: int  # optimistic locking


class FinalizeRequest(BaseModel):
    """Request body for POST /jobs/{id}/finalize."""
    finalize_policy: FinalizePolicy = FinalizePolicy.REJECT_PENDING


class EmbedRequest(BaseModel):
    """Request body for POST /jobs/{id}/embed."""
    provider: str = Field(
        default_factory=lambda: os.getenv("LONGPARSER_EMBED_PROVIDER", "huggingface")
    )
    model: str = Field(
        default_factory=lambda: os.getenv("LONGPARSER_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    )
    vector_db: str = Field(
        default_factory=lambda: os.getenv("LONGPARSER_VECTOR_DB", "chroma")
    )  # "chroma" | "faiss" | "qdrant"
    collection_name: Optional[str] = None


class SearchRequest(BaseModel):
    """Request body for POST /search."""
    query: str
    job_id: str
    top_k: int = 5
    index_version: Optional[str] = None  # defaults to latest
    filters: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class ReviewProgress(BaseModel):
    """Review completion stats."""
    approved: int = 0
    edited: int = 0
    rejected: int = 0
    pending: int = 0


class JobResponse(BaseModel):
    """Response body for GET /jobs/{id}."""
    job_id: str
    tenant_id: str
    status: JobStatus
    source_file: str
    file_hash: str = ""
    total_pages: int = 0
    total_blocks: int = 0
    total_chunks: int = 0
    review_progress: ReviewProgress = Field(default_factory=ReviewProgress)
    created_at: datetime
    finalized_at: Optional[datetime] = None
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """Response body for GET /jobs."""
    jobs: list[JobResponse]
    total: int


class BlockResponse(BaseModel):
    """Block data for API responses (confidence excluded)."""
    block_id: str
    type: BlockType
    text: str
    order_index: int = 0
    heading_level: Optional[int] = None
    indent_level: int = 0
    hierarchy_path: list[str] = Field(default_factory=list)
    page_number: int = 0
    review_status: ReviewStatus = ReviewStatus.PENDING
    current_revision_id: Optional[str] = None
    version: int = 1


class ChunkResponse(BaseModel):
    """Chunk data for API responses."""
    chunk_id: str
    text: str
    token_count: int = 0
    chunk_type: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.PENDING
    current_revision_id: Optional[str] = None
    version: int = 1


class SearchResult(BaseModel):
    """Single search result."""
    chunk_id: str
    text: str
    score: float
    chunk_type: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response body for POST /search."""
    results: list[SearchResult]
    index_version: str
    model: str
    query: str
    total: int

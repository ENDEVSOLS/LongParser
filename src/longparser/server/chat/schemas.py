"""Pydantic models for LongParser Chat API."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactSourceType(str, Enum):
    """Allowed fact source types."""
    DOC = "doc"
    USER = "user"
    ASSISTANT_INFERENCE = "assistant_inference"  # ephemeral — never persisted


# ---------------------------------------------------------------------------
# Config (read from env with defaults)
# ---------------------------------------------------------------------------

class ChatConfig(BaseModel):
    """Chat configuration — all values from env with sensible defaults."""

    llm_provider: str = Field(
        default_factory=lambda: os.getenv("LONGPARSER_LLM_PROVIDER", "openai")
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LONGPARSER_LLM_MODEL", "gpt-5.3")
    )
    max_input_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_MAX_INPUT_TOKENS", "1000"))
    )
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_MAX_OUTPUT_TOKENS", "2000"))
    )
    max_prompt_tokens: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_MAX_PROMPT_TOKENS", "6000"))
    )
    max_top_k: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_MAX_TOP_K", "10"))
    )
    rate_limit: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_RATE_LIMIT", "20"))
    )
    short_term_turns: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_SHORT_TERM_TURNS", "8"))
    )
    summarize_every: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_SUMMARIZE_EVERY", "10"))
    )
    extract_facts_every: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_EXTRACT_FACTS_EVERY", "20"))
    )
    max_facts: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_MAX_FACTS", "20"))
    )
    llm_timeout: float = Field(
        default_factory=lambda: float(os.getenv("LONGPARSER_LLM_TIMEOUT", "30"))
    )
    llm_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_LLM_MAX_RETRIES", "3"))
    )
    ttl_days: int = Field(
        default_factory=lambda: int(os.getenv("LONGPARSER_CHAT_TTL_DAYS", "30"))
    )


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    """POST /chat/sessions — create a chat session."""
    job_id: str


class ChatRequest(BaseModel):
    """POST /chat — ask a question."""
    session_id: str
    job_id: str
    question: str
    llm_provider: Optional[str] = None   # override env default
    llm_model: Optional[str] = None      # override env default
    top_k: int = 5
    idempotency_key: Optional[str] = None
    require_approval: bool = False       # opt-in HITL review


class HITLResumeRequest(BaseModel):
    """POST /chat/resume — resume a paused HITL chat."""
    session_id: str
    thread_id: str                        # LangGraph thread ID
    action: str                           # "approve" | "edit" | "reject"
    edited_answer: Optional[str] = None   # only for action="edit"


class SourceRef(BaseModel):
    """A reference to a retrieved chunk used as evidence."""
    chunk_id: str
    score: float
    text: str = ""
    page_numbers: list[int] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    session_id: str
    turn_id: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    status: str = "complete"              # "complete" | "pending_review"
    thread_id: Optional[str] = None       # set when status="pending_review"


class LLMAnswer(BaseModel):
    """Structured LLM output — enforced via with_structured_output."""
    answer: str
    cited_chunk_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Turn & Fact Models (stored in MongoDB)
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single Q&A turn in a chat session."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    archived: bool = False
    idempotency_key: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Fact(BaseModel):
    """A long-term fact extracted from conversation."""
    type: str  # entities_from_doc | user_preferences | decisions
    source: FactSourceType
    fact: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionInfo(BaseModel):
    """Response for GET /chat/sessions/{id}."""
    session_id: str
    tenant_id: str
    job_id: str
    turn_count: int = 0
    rolling_summary: str = ""
    long_term_facts: list[Fact] = Field(default_factory=list)
    created_at: datetime
    updated_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None

"""LangGraph HITL workflow for LongParser Chat.

Implements Human-in-the-Loop using LangGraph's interrupt() primitive.
When require_approval=True, the graph pauses after LLM response and
waits for human review via Command(resume=...).

Flow:
  User Question → RAG Chain → interrupt() → Human Reviews Draft
    ↓ Approve → Save Turn + Return final answer
    ↓ Edit    → Save edited answer + Return
    ↓ Reject  → Return rejection
"""

from __future__ import annotations

import logging
import uuid
from typing import TypedDict, Optional, Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from .schemas import ChatConfig, ChatRequest, ChatResponse, SourceRef, Turn, LLMAnswer

logger = logging.getLogger(__name__)

# Shared checkpointer for all HITL flows
_checkpointer = InMemorySaver()


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class HITLState(TypedDict):
    """State flowing through the HITL graph."""
    tenant_id: str
    session_id: str
    job_id: str
    question: str
    answer: str
    cited_chunk_ids: list[str]
    sources: list[dict]
    turn_id: str
    status: str           # "pending_review" | "complete" | "rejected"
    human_decision: Optional[dict]


# ---------------------------------------------------------------------------
# Graph Nodes
# ---------------------------------------------------------------------------

async def generate_answer(state: HITLState) -> HITLState:
    """Run the RAG chain to generate a draft answer.

    This imports and uses ChatEngine.ask() internally.
    The answer is placed in state for human review.
    """
    # Already computed and injected by the caller
    return state


async def human_review(state: HITLState) -> HITLState:
    """Pause execution for human review.

    Uses LangGraph's interrupt() to pause and wait for
    a Command(resume={action, edited_answer}).
    """
    decision = interrupt({
        "type": "review_request",
        "session_id": state["session_id"],
        "draft_answer": state["answer"],
        "cited_chunk_ids": state["cited_chunk_ids"],
        "message": "Please review this answer before it is sent.",
    })

    state["human_decision"] = decision
    return state


async def process_decision(state: HITLState) -> HITLState:
    """Process the human's decision: approve, edit, or reject."""
    decision = state.get("human_decision", {})
    action = decision.get("action", "approve")

    if action == "approve":
        state["status"] = "complete"
    elif action == "edit":
        state["answer"] = decision.get("edited_answer", state["answer"])
        state["status"] = "complete"
    elif action == "reject":
        state["answer"] = "Answer rejected by reviewer."
        state["status"] = "rejected"
        state["cited_chunk_ids"] = []
    else:
        state["status"] = "complete"

    return state


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

def build_hitl_graph() -> Any:
    """Build and compile the HITL state graph."""
    graph = StateGraph(HITLState)

    graph.add_node("generate", generate_answer)
    graph.add_node("review", human_review)
    graph.add_node("decide", process_decision)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "review")
    graph.add_edge("review", "decide")
    graph.add_edge("decide", END)

    return graph.compile(checkpointer=_checkpointer)


# Module-level compiled graph
hitl_graph = build_hitl_graph()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def start_hitl_review(
    tenant_id: str,
    session_id: str,
    job_id: str,
    question: str,
    answer: LLMAnswer,
    sources: list[SourceRef],
) -> dict:
    """Start a HITL review flow. Returns thread_id + draft."""
    thread_id = str(uuid.uuid4())

    initial_state: HITLState = {
        "tenant_id": tenant_id,
        "session_id": session_id,
        "job_id": job_id,
        "question": question,
        "answer": answer.answer,
        "cited_chunk_ids": answer.cited_chunk_ids,
        "sources": [s.model_dump() for s in sources],
        "turn_id": "",
        "status": "pending_review",
        "human_decision": None,
    }

    config = {"configurable": {"thread_id": thread_id}}
    result = await hitl_graph.ainvoke(initial_state, config=config)

    return {
        "thread_id": thread_id,
        "status": "pending_review",
        "draft_answer": answer.answer,
        "cited_chunk_ids": answer.cited_chunk_ids,
    }


async def resume_hitl_review(
    thread_id: str,
    action: str,
    edited_answer: Optional[str] = None,
) -> HITLState:
    """Resume a paused HITL flow with the human's decision."""
    config = {"configurable": {"thread_id": thread_id}}

    result = await hitl_graph.ainvoke(
        Command(resume={"action": action, "edited_answer": edited_answer}),
        config=config,
    )

    return result

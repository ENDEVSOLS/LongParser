"""Generate summary chunks for document sections using an LLM."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Dict, List, Any
import uuid

logger = logging.getLogger(__name__)

async def generate_summary_chunks(
    db: Any,
    tenant_id: str,
    job_id: str,
    provider: str = "gemini",
    model: str | None = None,
    max_concurrent: int = 3,
) -> list[dict]:
    """Generate 1-2 sentence summaries for each document section.
    
    Loads chunks from DB, groups by section, queries LLM, and returns new chunks.
    """
    from ..server.chat.llm_chain import get_plain_chat_model
    from langchain_core.messages import SystemMessage, HumanMessage
    import tiktoken
    
    llm = get_plain_chat_model(provider=provider, model=model)
    sem = asyncio.Semaphore(max_concurrent)
    
    # Simple tokenizer just for rough truncation
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None

    def _truncate(text: str, max_tokens: int = 2000) -> str:
        if enc:
            tokens = enc.encode(text)
            if len(tokens) > max_tokens:
                return enc.decode(tokens[:max_tokens]) + "..."
        else:
            # Fallback: ~4 chars per token
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                return text[:max_chars] + "..."
        return text

    # 1. Load chunks from DB
    chunks = await db.get_chunks(tenant_id, job_id)
    if not chunks:
        return []

    # 2. Group by section_path
    section_groups: Dict[tuple, List[str]] = {}
    for chunk in chunks:
        # Only summarize main text sections, skip existing summaries or figures
        if chunk.get("chunk_type") not in ("section", "equation"):
            continue
            
        path = tuple(chunk.get("section_path", []))
        if path not in section_groups:
            section_groups[path] = []
        section_groups[path].append(chunk.get("text", ""))

    summary_chunks = []

    # 3. Define the LLM call task
    async def _summarize(path: tuple, texts: List[str]):
        async with sem:
            full_text = "\n\n".join(texts)
            truncated_text = _truncate(full_text, max_tokens=2000)
            
            section_title = path[-1] if path else "Document Start"
            
            messages = [
                SystemMessage(content=(
                    "You are a helpful assistant. Summarize the following document section "
                    "in 1-2 concise sentences. Focus on the key facts, findings, or arguments. "
                    "Do not use conversational filler, just output the summary."
                )),
                HumanMessage(content=f"Section: {section_title}\n---\n{truncated_text}"),
            ]
            
            try:
                response = await llm.ainvoke(messages)
                summary_text = response.content.strip()
                
                chunk_id = str(uuid.uuid4())
                chunk_doc = {
                    "chunk_id": chunk_id,
                    "text": summary_text,
                    "token_count": len(summary_text.split()) * 1.3, # rough estimate
                    "chunk_type": "summary",
                    "section_path": list(path),
                    "page_numbers": [], # Summaries span multiple pages potentially
                    "block_ids": [],
                    "metadata": {"generated_by": f"{provider}/{model or 'default'}"},
                    "text_hash": hashlib.sha256(summary_text.encode()).hexdigest()[:16],
                    "quality_score": 1.0, # LLM generated, assume high quality text
                }
                summary_chunks.append(chunk_doc)
            except Exception as e:
                logger.error(f"Failed to summarize section {path}: {e}")

    # 4. Run all summaries concurrently
    tasks = [
        _summarize(path, texts)
        for path, texts in section_groups.items()
        if texts # Ensure there is text to summarize
    ]
    
    if tasks:
        await asyncio.gather(*tasks)

    return summary_chunks

"""Cross-reference resolution for chunks using regex and dict lookup.

Supports:
  - Explicit numbered refs: Figure 3, Table 2, Eq. 1, Section 3.1, Appendix A
  - Implicit proximity refs: "the figure above", "the table below"
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from ..schemas import Document, Chunk, Block, BlockType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Explicit labels found in target blocks (figures, tables, etc.)
_TARGET_LABEL_RE = re.compile(
    r'((?:Figure|Fig\.|Table|Equation|Eq\.|Section|Sec\.|'
    r'Chart|Diagram|Exhibit|Appendix|App\.)\s*\d+(?:\.\d+)*[a-z]?)',
    re.IGNORECASE,
)

# Explicit references found in body text
_REFERENCE_RE = re.compile(
    r'(?:see\s+|refer\s+to\s+|shown\s+in\s+|described\s+in\s+|listed\s+in\s+)?'
    r'((?:Figure|Fig\.|Table|Equation|Eq\.|Section|Sec\.|'
    r'Chart|Diagram|Exhibit|Appendix|App\.)\s*\d+(?:\.\d+)*[a-z]?)',
    re.IGNORECASE,
)

# Implicit / proximity-based references
_IMPLICIT_FIGURE_RE = re.compile(
    r'\b(?:the\s+)?(?:figure|image|illustration|chart|diagram)\s+'
    r'(?:above|below|following|preceding|previous|next)\b',
    re.IGNORECASE,
)
_IMPLICIT_TABLE_RE = re.compile(
    r'\b(?:the\s+)?(?:table)\s+'
    r'(?:above|below|following|preceding|previous|next)\b',
    re.IGNORECASE,
)

# Direction words mapped to search direction
_BEFORE_WORDS = {"above", "preceding", "previous"}
_AFTER_WORDS = {"below", "following", "next"}

_DIRECTION_RE = re.compile(r'(above|below|following|preceding|previous|next)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Proximity index builder
# ---------------------------------------------------------------------------

def _build_proximity_index(
    blocks: List[Block],
) -> Dict[str, List[Block]]:
    """Build ordered lists of figure and table blocks for proximity lookups."""
    figures: List[Block] = []
    tables: List[Block] = []

    for block in blocks:
        if block.type == BlockType.FIGURE:
            figures.append(block)
        elif block.type == BlockType.TABLE:
            tables.append(block)

    return {"figure": figures, "table": tables}


def _find_nearest(
    target_blocks: List[Block],
    anchor_block_ids: List[str],
    all_blocks: List[Block],
    direction: str,
) -> Optional[str]:
    """Find the nearest target block relative to the anchor blocks.
    
    direction: 'before' = look backwards, 'after' = look forwards
    """
    if not target_blocks or not anchor_block_ids:
        return None

    # Build a position index for O(1) lookup
    pos_index = {b.block_id: i for i, b in enumerate(all_blocks)}

    # Find the anchor position (use the first block in the chunk)
    anchor_pos = pos_index.get(anchor_block_ids[0])
    if anchor_pos is None:
        return None

    best_block = None
    best_distance = float("inf")

    for tb in target_blocks:
        tb_pos = pos_index.get(tb.block_id)
        if tb_pos is None:
            continue

        if direction == "before" and tb_pos < anchor_pos:
            dist = anchor_pos - tb_pos
            if dist < best_distance:
                best_distance = dist
                best_block = tb
        elif direction == "after" and tb_pos > anchor_pos:
            dist = tb_pos - anchor_pos
            if dist < best_distance:
                best_distance = dist
                best_block = tb

    return best_block.block_id if best_block else None


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_cross_references(document: Document, chunks: list[Chunk]) -> list[Chunk]:
    """Link references in chunks to their target blocks.
    
    Populates chunk.metadata["cross_references"] with targets.
    Handles both explicit numbered refs and implicit proximity refs.
    """
    if not document or not chunks:
        return chunks

    all_blocks = list(document.all_blocks)

    # 1. Build explicit label index: O(n) pass over document blocks
    ref_index: Dict[str, str] = {}

    for block in all_blocks:
        if block.type in (BlockType.FIGURE, BlockType.TABLE, BlockType.CAPTION, BlockType.EQUATION):
            match = _TARGET_LABEL_RE.search(block.text.strip())
            if match:
                label = match.group(1).lower().strip()
                ref_index[label] = block.block_id

    # 2. Build proximity index for implicit references
    prox_index = _build_proximity_index(all_blocks)

    # 3. Resolve: O(n) pass over chunks
    links_added = 0
    for chunk in chunks:
        if "cross_references" not in chunk.metadata:
            chunk.metadata["cross_references"] = []

        # --- 3a. Explicit references ---
        refs_found = _REFERENCE_RE.findall(chunk.text)

        for ref_text in refs_found:
            target_id = ref_index.get(ref_text.lower().strip())
            if target_id:
                existing = [
                    cr for cr in chunk.metadata["cross_references"]
                    if cr.get("label") == ref_text
                ]
                if not existing:
                    chunk.metadata["cross_references"].append({
                        "label": ref_text,
                        "target_block_id": target_id,
                    })
                    links_added += 1

        # --- 3b. Implicit figure references ---
        for match in _IMPLICIT_FIGURE_RE.finditer(chunk.text):
            phrase = match.group(0)
            dir_match = _DIRECTION_RE.search(phrase)
            if not dir_match:
                continue

            direction_word = dir_match.group(1).lower()
            direction = "before" if direction_word in _BEFORE_WORDS else "after"

            target_id = _find_nearest(
                prox_index["figure"], chunk.block_ids, all_blocks, direction
            )
            if target_id:
                existing = [
                    cr for cr in chunk.metadata["cross_references"]
                    if cr.get("target_block_id") == target_id
                ]
                if not existing:
                    chunk.metadata["cross_references"].append({
                        "label": phrase.strip(),
                        "target_block_id": target_id,
                        "resolution": "proximity",
                    })
                    links_added += 1

        # --- 3c. Implicit table references ---
        for match in _IMPLICIT_TABLE_RE.finditer(chunk.text):
            phrase = match.group(0)
            dir_match = _DIRECTION_RE.search(phrase)
            if not dir_match:
                continue

            direction_word = dir_match.group(1).lower()
            direction = "before" if direction_word in _BEFORE_WORDS else "after"

            target_id = _find_nearest(
                prox_index["table"], chunk.block_ids, all_blocks, direction
            )
            if target_id:
                existing = [
                    cr for cr in chunk.metadata["cross_references"]
                    if cr.get("target_block_id") == target_id
                ]
                if not existing:
                    chunk.metadata["cross_references"].append({
                        "label": phrase.strip(),
                        "target_block_id": target_id,
                        "resolution": "proximity",
                    })
                    links_added += 1

        # Clean up empty cross_references
        if not chunk.metadata["cross_references"]:
            del chunk.metadata["cross_references"]

    logger.info(f"Resolved {links_added} cross-references across {len(chunks)} chunks")
    return chunks

"""Chunk quality scorer based on token-weighted confidence and noise penalties."""

from __future__ import annotations

import logging
import re
from typing import Dict, Set

from ..schemas import Block, Chunk

logger = logging.getLogger(__name__)

# --- Lazy-loaded resources ---
_english_words: Set[str] | None = None

def _get_english_words() -> Set[str]:
    """Load standard OS dictionary for word coverage checks."""
    global _english_words
    if _english_words is None:
        _english_words = set()
        # Try common unix dictionary path
        try:
            with open("/usr/share/dict/words", "r", encoding="utf-8") as f:
                _english_words = {line.strip().lower() for line in f}
            logger.info(f"Loaded {len(_english_words)} words for quality scoring")
        except Exception:
            logger.debug("System dictionary not found. Word coverage metric will be skipped.")
    return _english_words

def _get_lang_confidence(text: str) -> float:
    """Get fastText language detection confidence (0.0 to 1.0)."""
    text = text.strip().replace("\n", " ")
    if len(text) < 10:
        return 1.0  # Too short to reliably detect, assume okay
        
    try:
        from fast_langdetect import detect
        res = detect(text)
        return res.get("score", 1.0)
    except Exception:
        return 1.0



def score_chunks(chunks: list[Chunk], blocks: list[Block]) -> list[Chunk]:
    """Score chunks based on block confidence and text noise.

    Assigns a quality_score (0.0 to 1.0) to each chunk.
    """
    if not chunks or not blocks:
        return chunks

    # Build block lookup for fast access
    block_lookup: Dict[str, Block] = {b.block_id: b for b in blocks}

    for chunk in chunks:
        chunk_blocks = [
            block_lookup[bid] for bid in chunk.block_ids if bid in block_lookup
        ]

        if not chunk_blocks:
            chunk.quality_score = 0.5  # Fallback
            continue

        # 1. Base score: token-weighted average of block confidence
        weighted_sum = sum(
            (b.confidence.overall if b.confidence else 1.0) * len(b.text)
            for b in chunk_blocks
        )
        total_weight = sum(len(b.text) for b in chunk_blocks)
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # 2. Noise penalty: density of garbled characters
        text = chunk.text
        noise_chars = sum(
            1 for c in text if not (c.isalnum() or c in ' .,;:!?()-"\'\n\t')
        )
        noise_ratio = noise_chars / max(len(text), 1)
        # Cap penalty at 50%
        penalty = min(noise_ratio * 2.0, 0.5)

        # 3. Dictionary Word Coverage penalty
        words = _get_english_words()
        if words:
            # Extract alphabetic tokens
            tokens = [t.lower() for t in re.findall(r'\b[a-zA-Z]{2,}\b', text)]
            if tokens:
                coverage = sum(1 for t in tokens if t in words) / len(tokens)
                # If less than 60% of tokens are real words, apply up to 30% penalty
                if coverage < 0.6:
                    penalty += min((0.6 - coverage), 0.3)

        # 4. FastText Language Confidence penalty
        # Garbled text often confuses the language ID model, resulting in low confidence
        lang_score = _get_lang_confidence(text)
        if lang_score < 0.8:
            # Scale penalty: 0.8 confidence = 0 penalty, 0.0 confidence = 0.4 penalty
            penalty += (0.8 - lang_score) * 0.5

        # 5. Completeness bonus: full sentences score higher
        ends_properly = text.rstrip().endswith(('.', '!', '?', ':', '"'))
        bonus = 0.05 if ends_properly else 0.0

        # Calculate final score (cap penalty before applying it)
        total_penalty = min(penalty, 0.8) # Max penalty is 80% to avoid dropping to 0 for weird formatting
        final_score = max(0.0, min(1.0, base_score - total_penalty + bonus))
        chunk.quality_score = final_score

    return chunks

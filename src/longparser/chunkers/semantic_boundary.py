"""Semantic boundary detection using SentenceTransformers."""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

_models: dict = {}


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazily load the SentenceTransformer model (cached by name)."""
    if model_name not in _models:
        try:
            from sentence_transformers import SentenceTransformer
            _models[model_name] = SentenceTransformer(model_name)
            logger.info("Loaded semantic chunking model: %s", model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. Semantic chunking disabled.")
            return None
    return _models[model_name]


def find_semantic_boundaries(
    texts: List[str],
    threshold: float = 0.3,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[int]:
    """Find semantic boundaries in a list of texts.
    
    Args:
        texts: List of block texts in reading order.
        threshold: Cosine similarity threshold. Drops below this indicate a shift.
        
    Returns:
        List of block indices where a semantic shift occurs (the boundary is *before* the index).
    """
    if not texts or len(texts) < 2:
        return []

    model = _get_model(model_name)
    if not model:
        return []

    # Batch encode all texts (fast on CPU)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    import numpy as np

    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    boundaries = []
    
    for i in range(len(embeddings) - 1):
        sim = cosine_sim(embeddings[i], embeddings[i+1])
        if sim < threshold:
            # Shift occurs before block i+1
            boundaries.append(i + 1)

    return boundaries

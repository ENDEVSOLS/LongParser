"""Pinned real-model integration and a small, fixed labelled boundary evaluation."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from longparser.chunkers import HybridChunker, quality_scorer, semantic_boundary
from longparser.schemas import ChunkingConfig
from tests.conftest import make_block

pytestmark = pytest.mark.real_embeddings
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
THRESHOLD = 0.3
DOCUMENTS = json.loads((Path(__file__).parents[1] / "fixtures/semantic_documents.json").read_text())


@pytest.fixture(scope="module")
def real_model_path():
    # Explicit opt-in must fail clearly if prerequisites are absent, not silently skip.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HUB_OFFLINE", "1")
        patch.setenv("TRANSFORMERS_OFFLINE", "1")

        def deny_network(*args, **kwargs):
            raise AssertionError("Real-model tests require local assets and must stay offline")

        patch.setattr(socket.socket, "connect", deny_network)
        try:
            import torch
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer  # noqa: F401

            path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        except (ImportError, OSError) as exc:
            pytest.fail(f"Real embedding prerequisites unavailable: {exc}. See docs/chunking-tests.md.")
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        patch.setattr(semantic_boundary, "_models", {})
        patch.setattr(quality_scorer, "_english_words", set())
        # Language scoring is separate from embedding accuracy and must not download fastText.
        import sys
        from types import ModuleType
        detector = ModuleType("fast_langdetect")
        detector.detect = lambda text: {"score": 1.0}
        patch.setitem(sys.modules, "fast_langdetect", detector)
        try:
            yield path
        finally:
            semantic_boundary._models.clear()
            torch.set_num_threads(previous_threads)


def test_real_embeddings_are_finite_and_have_expected_dimension(real_model_path):
    import numpy as np

    texts = DOCUMENTS[0]["texts"]
    assert semantic_boundary.find_semantic_boundaries(texts, THRESHOLD, real_model_path) == []
    from sentence_transformers import SentenceTransformer

    embeddings = SentenceTransformer(real_model_path, device="cpu").encode(
        texts, show_progress_bar=False
    )
    assert embeddings.shape == (3, 384)
    assert np.isfinite(embeddings).all()
    assert (np.linalg.norm(embeddings, axis=1) > 0).all()


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda doc: doc["id"])
def test_real_model_matches_fixed_boundary_labels(real_model_path, document):
    actual = semantic_boundary.find_semantic_boundaries(document["texts"], THRESHOLD, real_model_path)
    assert actual == document["boundaries"], (
        f"{document['id']}: expected {document['boundaries']}, got {actual}; "
        f"threshold={THRESHOLD}, revision={MODEL_REVISION}"
    )


def test_real_semantic_boundaries_control_chunk_packing(real_model_path):
    document = DOCUMENTS[1]
    blocks = [make_block(text, page=index + 1) for index, text in enumerate(document["texts"])]
    config = ChunkingConfig(max_tokens=512, min_tokens=0, overlap_blocks=0,
                            detect_equations=False, use_semantic_chunking=True,
                            semantic_threshold=THRESHOLD, semantic_model=real_model_path)
    chunks = HybridChunker(config).chunk(blocks)
    assert [c.text for c in chunks] == ["\n\n".join(document["texts"][:2]), document["texts"][2]]
    assert [c.block_ids for c in chunks] == [[b.block_id for b in blocks[:2]], [blocks[2].block_id]]
    assert [c.page_numbers for c in chunks] == [[1, 2], [3]]

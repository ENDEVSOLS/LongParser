"""Shared pytest fixtures for LongParser test suite."""

from __future__ import annotations

import pytest

# Import directly from submodules to avoid pulling in heavy optional deps
# (docling, motor, etc.) that live behind the top-level __init__ lazy imports.
from longparser.schemas import (
    Block,
    BlockFlags,
    BlockType,
    BoundingBox,
    Chunk,
    ChunkingConfig,
    Confidence,
    ProcessingConfig,
    Provenance,
    ExtractorType,
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def make_provenance(source="test.pdf", page=1) -> Provenance:
    """Create a minimal Provenance for tests."""
    return Provenance(
        source_file=source,
        page_number=page,
        bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
        extractor=ExtractorType.DOCLING,
    )


def make_confidence(score=1.0) -> Confidence:
    """Create a Confidence with the given overall score."""
    return Confidence(overall=score)


def make_block(
    text: str = "Hello world",
    block_type: BlockType = BlockType.PARAGRAPH,
    order_index: int = 0,
    heading_level: int | None = None,
    page: int = 1,
) -> Block:
    """Create a minimal Block suitable for chunker tests."""
    return Block(
        type=block_type,
        text=text,
        order_index=order_index,
        heading_level=heading_level,
        provenance=make_provenance(page=page),
        confidence=make_confidence(),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> ChunkingConfig:
    """Default chunking configuration."""
    return ChunkingConfig()


@pytest.fixture
def small_config() -> ChunkingConfig:
    """Tight token-budget chunking config for overflow tests."""
    return ChunkingConfig(max_tokens=50, min_tokens=5, overlap_blocks=0)


@pytest.fixture
def paragraph_block() -> Block:
    """Single paragraph block fixture."""
    return make_block("This is a test paragraph.", BlockType.PARAGRAPH)


@pytest.fixture
def heading_block() -> Block:
    """H1 heading block fixture."""
    return make_block("Introduction", BlockType.HEADING, heading_level=1)


@pytest.fixture
def simple_block_list() -> list[Block]:
    """H1 → paragraph → H2 → paragraph sequence."""
    return [
        make_block("Introduction", BlockType.HEADING, order_index=0, heading_level=1),
        make_block("This is the intro text. " * 5, BlockType.PARAGRAPH, order_index=1),
        make_block("Background", BlockType.HEADING, order_index=2, heading_level=2),
        make_block("Background details. " * 5, BlockType.PARAGRAPH, order_index=3),
    ]


@pytest.fixture
def processing_config() -> ProcessingConfig:
    """Default processing config."""
    return ProcessingConfig()

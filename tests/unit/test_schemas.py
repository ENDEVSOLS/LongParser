"""Unit tests for LongParser Pydantic schemas."""

from __future__ import annotations

import uuid
import pytest
from pydantic import ValidationError

from longparser.schemas import (
    Block,
    BlockFlags,
    BlockType,
    BoundingBox,
    Chunk,
    ChunkingConfig,
    Confidence,
    Document,
    DocumentMetadata,
    ExtractorType,
    JobRequest,
    JobResult,
    Page,
    ProcessingConfig,
    Provenance,
    Table,
    TableCell,
)
from tests.conftest import make_block, make_provenance, make_confidence


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def test_valid(self):
        bb = BoundingBox(x0=0, y0=0, x1=100, y1=50)
        assert bb.x1 == 100

    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            BoundingBox(x0=0, y0=0)  # type: ignore


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_defaults(self):
        c = Confidence(overall=0.95)
        assert c.text_confidence == 1.0
        assert c.layout_confidence == 1.0
        assert c.table_confidence is None

    def test_range_validation_below(self):
        with pytest.raises(ValidationError):
            Confidence(overall=-0.1)

    def test_range_validation_above(self):
        with pytest.raises(ValidationError):
            Confidence(overall=1.1)


# ---------------------------------------------------------------------------
# BlockFlags
# ---------------------------------------------------------------------------

class TestBlockFlags:
    def test_defaults_all_false(self):
        f = BlockFlags()
        assert f.needs_review is False
        assert f.repaired is False
        assert f.excluded_from_rag is False


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class TestBlock:
    def test_auto_block_id(self):
        b = make_block()
        assert b.block_id  # non-empty UUID string
        assert len(b.block_id) == 36  # UUID4 format

    def test_table_block_has_no_table_by_default(self):
        b = make_block(block_type=BlockType.TABLE)
        assert b.table is None

    def test_heading_level_none_for_paragraph(self):
        b = make_block(block_type=BlockType.PARAGRAPH)
        assert b.heading_level is None

    def test_hierarchy_path_defaults_empty(self):
        b = make_block()
        assert b.hierarchy_path == []


# ---------------------------------------------------------------------------
# TableCell — alias fields
# ---------------------------------------------------------------------------

class TestTableCell:
    def test_alias_fields(self):
        cell = TableCell(**{"r0": 0, "c0": 1, "text": "hello"})
        assert cell.row_index == 0
        assert cell.col_index == 1

    def test_populate_by_name(self):
        cell = TableCell(row_index=2, col_index=3, text="world")
        assert cell.text == "world"


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_all_blocks_across_pages(self):
        page1 = Page(
            page_number=1, width=595, height=842,
            blocks=[make_block("A"), make_block("B")]
        )
        page2 = Page(
            page_number=2, width=595, height=842,
            blocks=[make_block("C")]
        )
        doc = Document(
            metadata=DocumentMetadata(source_file="test.pdf"),
            pages=[page1, page2],
        )
        assert len(doc.all_blocks) == 3

    def test_all_tables_empty_if_no_tables(self):
        doc = Document(
            metadata=DocumentMetadata(source_file="test.pdf"),
            pages=[],
        )
        assert doc.all_tables == []


# ---------------------------------------------------------------------------
# ProcessingConfig
# ---------------------------------------------------------------------------

class TestProcessingConfig:
    def test_defaults(self):
        cfg = ProcessingConfig()
        assert cfg.do_ocr is True
        assert cfg.formula_mode == "smart"
        assert cfg.ocr_backend == "easyocr"

    def test_custom_values(self):
        cfg = ProcessingConfig(do_ocr=False, force_full_page_ocr=True)
        assert cfg.do_ocr is False
        assert cfg.force_full_page_ocr is True


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------

class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.max_tokens == 512
        assert cfg.table_rows_per_chunk == 15
        assert cfg.table_chunk_format == "row_record"

    def test_custom(self):
        cfg = ChunkingConfig(max_tokens=256, overlap_blocks=2)
        assert cfg.max_tokens == 256
        assert cfg.overlap_blocks == 2


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_auto_chunk_id(self):
        chunk = Chunk(text="Test", token_count=1, chunk_type="section")
        assert chunk.chunk_id
        assert len(chunk.chunk_id) == 36

    def test_defaults(self):
        chunk = Chunk(text="Content", token_count=5, chunk_type="table")
        assert chunk.section_path == []
        assert chunk.page_numbers == []
        assert chunk.overlap_with_previous is False


# ---------------------------------------------------------------------------
# JobRequest / JobResult
# ---------------------------------------------------------------------------

class TestJobModels:
    def test_job_request_defaults(self):
        req = JobRequest(file_path="/tmp/test.pdf")
        assert req.job_id
        assert req.config.do_ocr is True

    def test_job_result_success_default(self):
        doc = Document(metadata=DocumentMetadata(source_file="x.pdf"))
        result = JobResult(job_id="abc", document=doc)
        assert result.success is True
        assert result.errors == []

"""Backward compatibility tests for v0.2.x changes.

Ensures that users who wrote code against v0.1.3 can upgrade to v0.2.x
without changing a single line of their code. Every new field must have
a default that matches the v0.1.3 behavior.
"""

import pytest


class TestProcessingConfigCompat:
    """ProcessingConfig() with no args must behave exactly like v0.1.3."""

    def test_default_values_match_v013(self):
        from longparser.schemas import ProcessingConfig
        config = ProcessingConfig()

        # v0.1.3 defaults — these must NEVER change
        assert config.academic_mode is False
        assert config.rtl_hint is False
        assert config.do_ocr is True
        assert config.formula_ocr is True
        assert config.do_table_structure is True
        assert config.export_images is True
        assert config.formula_mode == "smart"
        assert config.smart_max_equations == 25
        assert config.smart_max_ocr_seconds == 300.0
        assert config.exclude_page_headers_footers is True

    def test_new_fields_have_safe_defaults(self):
        """New v0.2.x fields must default to values that don't change behavior."""
        from longparser.schemas import ProcessingConfig
        config = ProcessingConfig()

        # backend must default to docling (existing behavior)
        backend = getattr(config, "backend", "docling")
        assert backend == "docling"

        # auto_detect_language defaults to True but only runs if languages=None
        auto_detect = getattr(config, "auto_detect_language", True)
        assert auto_detect is True

        # languages=None means "use existing tesseract_lang param"
        languages = getattr(config, "languages", None)
        assert languages is None


class TestDocumentMetadataCompat:
    """DocumentMetadata must keep all v0.1.3 fields."""

    def test_v013_fields_exist(self):
        from longparser.schemas import DocumentMetadata
        meta = DocumentMetadata(source_file="test.pdf")

        assert meta.source_file == "test.pdf"
        assert meta.file_hash == ""
        assert meta.language is None
        assert meta.total_pages == 0
        assert meta.academic_mode is False
        assert meta.rtl_hint is False


class TestBlockCompat:
    """Block schema must keep all v0.1.3 fields and types."""

    def test_block_type_values_unchanged(self):
        from longparser.schemas import BlockType

        # All v0.1.3 values must still exist
        assert BlockType.HEADING == "heading"
        assert BlockType.PARAGRAPH == "paragraph"
        assert BlockType.LIST_ITEM == "list_item"
        assert BlockType.TABLE == "table"
        assert BlockType.FIGURE == "figure"
        assert BlockType.CAPTION == "caption"
        assert BlockType.FOOTER == "footer"
        assert BlockType.HEADER == "header"
        assert BlockType.EQUATION == "equation"
        assert BlockType.CODE == "code"

    def test_extractor_type_values_unchanged(self):
        from longparser.schemas import ExtractorType

        # All v0.1.3 values must still exist
        assert ExtractorType.DOCLING == "docling"
        assert ExtractorType.SURYA == "surya"
        assert ExtractorType.MARKER == "marker"
        assert ExtractorType.NATIVE_PDF == "native_pdf"
        assert ExtractorType.PADDLE == "paddle"


class TestChunkCompat:
    """Chunk schema must keep all v0.1.3 fields."""

    def test_chunk_fields_exist(self):
        from longparser.schemas import Chunk
        chunk = Chunk(text="test", token_count=1, chunk_type="section")

        assert chunk.text == "test"
        assert chunk.token_count == 1
        assert chunk.chunk_type == "section"
        assert chunk.section_path == []
        assert chunk.page_numbers == []
        assert chunk.block_ids == []
        assert chunk.overlap_with_previous is False
        assert chunk.equation_detected is False


class TestPublicAPICompat:
    """All v0.1.3 public names must still be importable."""

    def test_all_v013_exports_available(self):
        from longparser import (  # noqa: F401
            __version__,
            Document,
            Page,
            Block,
            Table,
            TableCell,
            BlockType,
            ExtractorType,
            ProcessingConfig,
            BoundingBox,
            Provenance,
            Confidence,
            BlockFlags,
            DocumentMetadata,
            PageProfile,
            ExtractionMetadata,
            ChunkingConfig,
            Chunk,
            JobRequest,
            JobResult,
        )

    def test_lazy_imports_still_work(self):
        """Lazy imports from v0.1.3 must still resolve."""
        from longparser import DocumentPipeline  # noqa: F401
        from longparser import PipelineOrchestrator  # noqa: F401
        from longparser import PipelineResult  # noqa: F401
        from longparser import HybridChunker  # noqa: F401
        from longparser import DoclingExtractor  # noqa: F401

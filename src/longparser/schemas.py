"""Pydantic schemas for LongParser document processing pipeline."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Types of document blocks."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    EQUATION = "equation"
    CODE = "code"


class ExtractorType(str, Enum):
    """Document extraction engines."""
    DOCLING = "docling"
    SURYA = "surya"
    MARKER = "marker"
    NATIVE_PDF = "native_pdf"
    PADDLE = "paddle"


class BoundingBox(BaseModel):
    """Bounding box coordinates (PDF coordinate system)."""
    x0: float
    y0: float
    x1: float
    y1: float


class Provenance(BaseModel):
    """Traceability information for a block."""
    source_file: str
    page_number: int
    bbox: BoundingBox
    extractor: ExtractorType
    extractor_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"


class Confidence(BaseModel):
    """Confidence scores for extraction quality."""
    overall: float = Field(ge=0.0, le=1.0)
    text_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    layout_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    table_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class BlockFlags(BaseModel):
    """Flags indicating block status."""
    needs_review: bool = False
    repaired: bool = False
    fallback_used: bool = False
    excluded_from_rag: bool = False


class TableCell(BaseModel):
    """Individual cell in a table."""
    row_index: int = Field(alias="r0")
    col_index: int = Field(alias="c0")
    row_span: int = Field(default=1, alias="rspan")
    col_span: int = Field(default=1, alias="cspan")
    text: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


class Table(BaseModel):
    """Table structure with cells and metadata."""
    table_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    n_rows: int
    n_cols: int
    cells: list[TableCell] = Field(default_factory=list)
    table_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    csv_path: Optional[str] = None
    html_path: Optional[str] = None


class Block(BaseModel):
    """Core document element with full provenance."""
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType
    text: str = ""
    order_index: int = 0
    heading_level: Optional[int] = Field(default=None, description="Heading level (1-6) for heading blocks, inferred by Docling")
    indent_level: int = Field(default=0, description="Bullet nesting depth (0=top, 1=sub, 2=sub-sub). Used for PPTX list items.")
    hierarchy_path: list[str] = Field(default_factory=list)
    refs_out: list[str] = Field(default_factory=list)
    refs_in: list[str] = Field(default_factory=list)
    provenance: Provenance
    confidence: Confidence
    flags: BlockFlags = Field(default_factory=BlockFlags)
    table: Optional[Table] = None
    image_path: Optional[str] = None
    bbox_px: Optional[tuple] = Field(default=None, description="Pixel-space bounding box (x0,y0,x1,y1) for MFD dedup")


class PageProfile(BaseModel):
    """Validation profile for a single page."""
    page_number: int
    needs_reprocess: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    layout_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    table_confidence: Optional[float] = None
    has_rtl: bool = False
    has_math: bool = False
    detected_columns: int = Field(default=1, description="Number of text columns detected on page")
    reading_order_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence of reading-order reconstruction")


class Page(BaseModel):
    """Single page with blocks and metadata."""
    page_number: int
    width: float
    height: float
    blocks: list[Block] = Field(default_factory=list)
    rendered_image_path: Optional[str] = None
    profile: Optional[PageProfile] = None


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    source_file: str
    file_hash: str = ""
    language: Optional[str] = None
    detected_language: Optional[str] = Field(default=None, description="Auto-detected language code (ISO 639-1) via fast-langdetect")
    language_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence of auto-detected language")
    total_pages: int = 0
    academic_mode: bool = False
    rtl_hint: bool = False


class Document(BaseModel):
    """Complete processed document."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata
    pages: list[Page] = Field(default_factory=list)

    @property
    def all_blocks(self) -> list[Block]:
        """Get all blocks across all pages."""
        return [block for page in self.pages for block in page.blocks]

    @property
    def all_tables(self) -> list[Table]:
        """Get all tables from table blocks."""
        return [
            block.table
            for block in self.all_blocks
            if block.type == BlockType.TABLE and block.table
        ]


class ProcessingConfig(BaseModel):
    """Configuration for pipeline execution."""
    # --- v0.1.4: Backend selection ---
    backend: str = Field(default="docling", description="Extraction backend: 'docling' | 'pymupdf' | 'auto'")

    # --- v0.1.4: Language detection ---
    languages: Optional[list[str]] = Field(default=None, description="Explicit Tesseract language codes, e.g. ['eng','ara']. Overrides auto-detect.")
    auto_detect_language: bool = Field(default=True, description="Auto-detect document language before OCR (uses fast-langdetect)")

    # --- v0.1.4: Multi-column layout ---
    column_count_hint: Optional[int] = Field(default=None, description="Manual column count hint. None = auto-detect by Docling")
    force_left_to_right: bool = Field(default=False, description="Force left-to-right top-to-bottom reading order")

    academic_mode: bool = False
    rtl_hint: bool = False
    do_ocr: bool = True
    formula_ocr: bool = True  # Independent from do_ocr — runs pix2tex even when text OCR is off
    do_table_structure: bool = True
    export_images: bool = True
    images_output_dir: Optional[str] = None
    layout_confidence_threshold: float = 0.7
    table_confidence_threshold: float = 0.75
    ocr_noise_threshold: float = 0.15
    enable_fallback: bool = True
    # Smart extraction fallback options
    prefer_initial_on_degradation: bool = True  # Keep original if fallback degrades quality
    ocr_backend: str = "easyocr"  # easyocr | tesseract | rapidocr
    ocr_use_gpu: bool = True
    # Force full page OCR for better text segmentation in two-column layouts
    force_full_page_ocr: bool = False
    # Exclude page headers and footers from extraction output
    exclude_page_headers_footers: bool = True
    
    # Formula extraction mode: "full" (slow, best LaTeX), "fast" (Unicode text), "smart" (hybrid)
    formula_mode: str = "smart"
    # Smart mode safety caps
    smart_max_pages: int = 5      # Max pages to enrich in smart mode before fallback to fast
    smart_max_ratio: float = 0.2  # Max ratio of pages (enriched/total) before fallback
    smart_max_equations: int = 25       # Max equations to OCR per document (circuit breaker)
    smart_max_ocr_seconds: float = 300.0  # Total OCR time budget (must fit within arq timeout)


class ExtractionMetadata(BaseModel):
    """Metadata from smart extraction strategy."""
    strategy_used: str = "standard"  # standard | force_full_page_ocr
    initial_low_grade: Optional[str] = None
    fallback_low_grade: Optional[str] = None
    improved: bool = False
    fallback_degraded: bool = False
    reprocessed_pages: list[int] = Field(default_factory=list)
    ocr_backend_used: Optional[str] = None
    reasons: list[str] = Field(default_factory=list)
    # --- v0.1.4: OCR routing metadata ---
    ocr_strategy: str = Field(default="standard", description="OCR strategy used: 'standard' | 'math' | 'full_ocr'")
    is_scanned: bool = Field(default=False, description="Whether the document was detected as scanned (no text layer)")
    page_complexity_scores: dict[int, int] = Field(default_factory=dict, description="Per-page complexity scores used for OCR routing")


class ChunkingConfig(BaseModel):
    """Configuration for hybrid chunking."""
    max_tokens: int = 512
    min_tokens: int = 100
    overlap_blocks: int = 1
    table_rows_per_chunk: int = 15
    exclude_headers_footers: bool = True
    detect_equations: bool = True
    table_chunk_format: str = "row_record"  # "row_record" | "pipe"
    generate_schema_chunks: bool = True
    wide_table_col_threshold: int = 25


class Chunk(BaseModel):
    """A RAG-optimized chunk with full provenance."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    token_count: int
    chunk_type: str  # "section" | "table" | "table_schema" | "list" | "equation" | "figure" | "continuation"
    section_path: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    overlap_with_previous: bool = False
    equation_detected: bool = False
    image_path: Optional[str] = Field(default=None, description="Path to figure image if chunk_type == 'figure'")
    metadata: dict = Field(default_factory=dict)  # row_start, row_end, sheet, col_band


class JobRequest(BaseModel):
    """Request to process a document."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    config: ProcessingConfig = Field(default_factory=ProcessingConfig)


class JobResult(BaseModel):
    """Result of document processing."""
    job_id: str
    document: Document
    success: bool = True
    errors: list[str] = Field(default_factory=list)
    processing_time_seconds: float = 0.0

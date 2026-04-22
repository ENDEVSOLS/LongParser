"""PyMuPDF4LLM-based extractor for fast, CPU-native PDF extraction.

⚠️  LICENSE NOTICE — AGPL-3.0
    pymupdf4llm is dual-licensed under AGPL-3.0 or Artifex Commercial License.
    By using this backend, you agree to the terms of the AGPL-3.0 license
    unless you have purchased a commercial license from Artifex Software, Inc.

    This module is NOT imported by default — users must explicitly opt in
    via ``pip install longparser[pymupdf]`` and ``backend='pymupdf'``.

⚠️  ISOLATION RULES (do NOT violate)
    1. This file must NEVER be imported by ``extractors/__init__.py``
    2. This file must NEVER be imported at module level by ``orchestrator.py``
    3. This file must ONLY be imported behind ``if backend == "pymupdf":``
    4. ``import longparser`` must NEVER trigger loading this file

Best for:
    - Native PDFs with embedded text (not scanned)
    - Speed-critical pipelines (10-50× faster than Docling)
    - CPU-only environments (no GPU, no ML models)

NOT suitable for:
    - Scanned PDFs (no OCR capability)
    - Complex tables with merged cells
    - Documents needing deep heading hierarchy detection

Usage::

    from longparser import ProcessingConfig, DocumentPipeline

    pipeline = DocumentPipeline(
        config=ProcessingConfig(backend="pymupdf")
    )
    result = pipeline.process_file("report.pdf")
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

from ..schemas import (
    Document, Page, Block, Table, TableCell,
    BlockType, ExtractorType, ProcessingConfig,
    BoundingBox, Provenance, Confidence, BlockFlags,
    DocumentMetadata, PageProfile, ExtractionMetadata,
)
from .base import BaseExtractor

logger = logging.getLogger(__name__)


def _require_pymupdf():
    """Check that pymupdf4llm is installed; raise clear error if not.

    Returns the ``pymupdf4llm`` module on success.
    """
    try:
        import pymupdf4llm
        return pymupdf4llm
    except ImportError:
        raise ImportError(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  pymupdf4llm is not installed.                         ║\n"
            "║                                                        ║\n"
            "║  Install:  pip install 'longparser[pymupdf]'           ║\n"
            "║                                                        ║\n"
            "║  ⚠️  pymupdf4llm is licensed under AGPL-3.0.           ║\n"
            "║  By installing it, you agree to AGPL terms for that    ║\n"
            "║  component. LongParser core remains MIT-licensed.      ║\n"
            "║                                                        ║\n"
            "║  For commercial use without AGPL obligations, purchase ║\n"
            "║  a license from https://artifex.com                    ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )


def _require_pymupdf_fitz():
    """Import the fitz (PyMuPDF) module for page-level operations."""
    try:
        import pymupdf as fitz
        return fitz
    except ImportError:
        try:
            import fitz
            return fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for the pymupdf backend. "
                "Install with: pip install 'longparser[pymupdf]'"
            )


class PyMuPDFExtractor(BaseExtractor):
    """Fast, CPU-native PDF extractor using PyMuPDF4LLM.

    Converts PDFs to structured Markdown and maps the output to
    LongParser's ``Document`` / ``Block`` model. Uses no ML models,
    no GPU — pure C-based PDF parsing via MuPDF.

    Attributes
    ----------
    extractor_type : ExtractorType
        Always ``ExtractorType.NATIVE_PDF``.
    version : str
        Extractor version string.
    """

    extractor_type = ExtractorType.NATIVE_PDF
    version = "1.0.0"

    def __init__(self):
        """Initialize and verify pymupdf4llm is available."""
        _require_pymupdf()
        self._images: list = []
        logger.info(
            "PyMuPDF4LLM backend initialized (CPU-native, no OCR, no GPU)"
        )

    def extract(
        self,
        file_path: Path,
        config: ProcessingConfig,
        page_numbers: Optional[List[int]] = None,
    ) -> Tuple[Document, ExtractionMetadata]:
        """Extract a PDF using PyMuPDF4LLM.

        Parameters
        ----------
        file_path:
            Path to the PDF file.
        config:
            Processing configuration.
        page_numbers:
            Optional list of 0-indexed page numbers to extract.

        Returns
        -------
        tuple[Document, ExtractionMetadata]
            Extracted document and metadata.
        """
        import pymupdf4llm

        file_path = Path(file_path)
        logger.info("Extracting with PyMuPDF4LLM: %s", file_path.name)

        # Validate file type
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"PyMuPDF4LLM backend only supports PDF files, got: {file_path.suffix}"
            )

        # File hash
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]

        # Extract with pymupdf4llm
        kwargs = {"show_progress": False}
        if page_numbers is not None:
            kwargs["pages"] = page_numbers

        md_text = pymupdf4llm.to_markdown(str(file_path), **kwargs)

        # Get page-level info using PyMuPDF directly
        fitz = _require_pymupdf_fitz()
        pdf_doc = fitz.open(str(file_path))
        total_pages = len(pdf_doc)

        # Extract images if config.export_images
        self._images = []
        if config.export_images:
            self._extract_images(pdf_doc, config)

        # Build Document from Markdown
        document = self._markdown_to_document(
            md_text=md_text,
            pdf_doc=pdf_doc,
            file_path=file_path,
            file_hash=file_hash,
            total_pages=total_pages,
            config=config,
        )

        pdf_doc.close()

        meta = ExtractionMetadata(
            strategy_used="pymupdf4llm",
            ocr_backend_used="none (native text)",
        )

        logger.info(
            "PyMuPDF4LLM extraction complete: %d pages, %d blocks",
            total_pages, len(document.all_blocks),
        )

        return document, meta

    def _markdown_to_document(
        self,
        md_text: str,
        pdf_doc,
        file_path: Path,
        file_hash: str,
        total_pages: int,
        config: ProcessingConfig,
    ) -> Document:
        """Convert Markdown text to a LongParser Document model."""
        metadata = DocumentMetadata(
            source_file=str(file_path),
            file_hash=file_hash,
            total_pages=total_pages,
        )

        pages: list[Page] = []

        # Split markdown by page breaks (pymupdf4llm uses "---" or form feeds)
        page_chunks = self._split_by_pages(md_text, total_pages)

        for page_idx, page_md in enumerate(page_chunks):
            page_no = page_idx + 1

            # Get page dimensions from PyMuPDF
            if page_idx < len(pdf_doc):
                rect = pdf_doc[page_idx].rect
                width, height = rect.width, rect.height
            else:
                width, height = 612.0, 792.0  # Letter default

            # Parse markdown blocks
            blocks = self._parse_markdown_blocks(page_md, page_no, file_path)

            # Build page profile
            profile = PageProfile(
                page_number=page_no,
                layout_confidence=0.9,  # PyMuPDF is reliable for native PDFs
            )

            pages.append(Page(
                page_number=page_no,
                width=width,
                height=height,
                blocks=blocks,
                profile=profile,
            ))

        return Document(metadata=metadata, pages=pages)

    def _split_by_pages(self, md_text: str, total_pages: int) -> list[str]:
        """Split markdown text into per-page chunks."""
        import re

        # pymupdf4llm inserts page separators
        # Common patterns: "-----" (5+ dashes), or form feed characters
        parts = re.split(r'\n-{3,}\n|\f', md_text)

        # If splitting didn't work, put everything on page 1
        if len(parts) <= 1:
            return [md_text]

        # Pad to total_pages if needed
        while len(parts) < total_pages:
            parts.append("")

        return parts[:total_pages]

    def _parse_markdown_blocks(
        self,
        page_md: str,
        page_no: int,
        file_path: Path,
    ) -> list[Block]:
        """Parse markdown text into Block objects."""
        blocks: list[Block] = []
        lines = page_md.strip().split("\n")
        order_idx = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                i += 1
                continue

            # Detect block type
            if stripped.startswith("#"):
                # Heading
                level = len(stripped) - len(stripped.lstrip("#"))
                text = stripped.lstrip("#").strip()
                block = self._make_block(
                    BlockType.HEADING, text, order_idx, page_no,
                    file_path, heading_level=min(level, 6),
                )
                blocks.append(block)

            elif stripped.startswith("|") and "|" in stripped[1:]:
                # Table — collect all table lines
                table_lines = [stripped]
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                    i += 1
                table_md = "\n".join(table_lines)
                table_obj = self._parse_table(table_lines)
                block = self._make_block(
                    BlockType.TABLE, table_md, order_idx, page_no,
                    file_path, table=table_obj,
                )
                blocks.append(block)
                order_idx += 1
                continue  # Already incremented i

            elif stripped.startswith(("- ", "* ", "+ ")) or (
                len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)"
            ):
                # List item
                text = stripped.lstrip("-*+ ").lstrip("0123456789.)").strip()
                block = self._make_block(
                    BlockType.LIST_ITEM, text, order_idx, page_no, file_path,
                )
                blocks.append(block)

            elif stripped.startswith("```"):
                # Code block
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                code_text = "\n".join(code_lines)
                block = self._make_block(
                    BlockType.CODE, code_text, order_idx, page_no, file_path,
                )
                blocks.append(block)
                i += 1  # Skip closing ```
                order_idx += 1
                continue

            elif stripped.startswith("$$") or stripped.startswith("\\["):
                # Equation block
                eq_lines = [stripped]
                if not (stripped.endswith("$$") and len(stripped) > 2):
                    i += 1
                    while i < len(lines):
                        eq_line = lines[i].strip()
                        eq_lines.append(eq_line)
                        if eq_line.endswith("$$") or eq_line.endswith("\\]"):
                            break
                        i += 1
                eq_text = "\n".join(eq_lines)
                block = self._make_block(
                    BlockType.EQUATION, eq_text, order_idx, page_no, file_path,
                )
                blocks.append(block)

            else:
                # Regular paragraph
                block = self._make_block(
                    BlockType.PARAGRAPH, stripped, order_idx, page_no, file_path,
                )
                blocks.append(block)

            order_idx += 1
            i += 1

        return blocks

    def _make_block(
        self,
        block_type: BlockType,
        text: str,
        order_index: int,
        page_no: int,
        file_path: Path,
        heading_level: Optional[int] = None,
        table: Optional[Table] = None,
    ) -> Block:
        """Create a Block with standard provenance."""
        return Block(
            type=block_type,
            text=text,
            order_index=order_index,
            heading_level=heading_level,
            provenance=Provenance(
                source_file=str(file_path),
                page_number=page_no,
                bbox=BoundingBox(x0=0, y0=0, x1=0, y1=0),
                extractor=self.extractor_type,
                extractor_version=self.version,
            ),
            confidence=Confidence(overall=0.9),
            table=table,
        )

    def _parse_table(self, table_lines: list[str]) -> Table:
        """Parse a Markdown table into a Table object."""
        # Filter out separator lines (|---|---|)
        data_lines = [
            line for line in table_lines
            if line.strip() and not all(c in "|-: " for c in line.strip())
        ]

        if not data_lines:
            return Table(n_rows=0, n_cols=0)

        cells: list[TableCell] = []
        n_cols = 0

        for row_idx, line in enumerate(data_lines):
            parts = [p.strip() for p in line.strip("|").split("|")]
            n_cols = max(n_cols, len(parts))
            for col_idx, cell_text in enumerate(parts):
                cells.append(TableCell(
                    r0=row_idx, c0=col_idx, text=cell_text
                ))

        return Table(
            n_rows=len(data_lines),
            n_cols=n_cols,
            cells=cells,
            table_confidence=0.85,
        )

    def _extract_images(self, pdf_doc, config: ProcessingConfig):
        """Extract images from PDF pages."""
        for page_idx in range(len(pdf_doc)):
            page = pdf_doc[page_idx]
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_doc.extract_image(xref)
                    if base_image:
                        self._images.append({
                            "page": page_idx + 1,
                            "index": img_idx,
                            "data": base_image["image"],
                            "ext": base_image.get("ext", "png"),
                        })
                except Exception as e:
                    logger.debug("Failed to extract image on page %d: %s", page_idx + 1, e)

    def save_images(self, output_dir: Path) -> list[Path]:
        """Save extracted images to disk.

        Parameters
        ----------
        output_dir:
            Directory to save images to.

        Returns
        -------
        list[Path]
            Paths to saved image files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []

        for img_info in self._images:
            fname = f"page_{img_info['page']:03d}_img_{img_info['index']:02d}.{img_info['ext']}"
            fpath = output_dir / fname
            with open(fpath, "wb") as f:
                f.write(img_info["data"])
            saved.append(fpath)

        logger.info("Saved %d images to %s", len(saved), output_dir)
        return saved

    def to_markdown(self, document: Document) -> str:
        """Convert Document back to Markdown."""
        parts = []
        for page in document.pages:
            for block in page.blocks:
                if block.type == BlockType.HEADING:
                    level = block.heading_level or 1
                    parts.append(f"{'#' * level} {block.text}")
                elif block.type == BlockType.TABLE:
                    parts.append(block.text)
                elif block.type == BlockType.LIST_ITEM:
                    parts.append(f"- {block.text}")
                elif block.type == BlockType.CODE:
                    parts.append(f"```\n{block.text}\n```")
                elif block.type == BlockType.EQUATION:
                    parts.append(f"$$\n{block.text}\n$$")
                else:
                    parts.append(block.text)
                parts.append("")
        return "\n".join(parts)

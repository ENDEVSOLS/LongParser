"""Marker-based extractor for high-fidelity extraction on complex PDFs.

⚠️  LICENSE NOTICE — GPL-3.0
    marker-pdf is licensed under GPL-3.0.
    By using this backend, you agree to the terms of the GPL-3.0 license.

    This module is NOT imported by default — users must explicitly opt in
    via ``pip install longparser[marker]`` and ``backend='marker'``.

⚠️  ISOLATION RULES (do NOT violate)
    1. This file must NEVER be imported by ``extractors/__init__.py``
    2. This file must NEVER be imported at module level by ``orchestrator.py``
    3. This file must ONLY be imported behind ``if backend == "marker":``
    4. ``import longparser`` must NEVER trigger loading this file
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Tuple

from ..schemas import (
    Document, Page, Block, BlockType, ExtractorType, ProcessingConfig,
    BoundingBox, Provenance, Confidence, DocumentMetadata, PageProfile, ExtractionMetadata
)
from .base import BaseExtractor

logger = logging.getLogger(__name__)


def _require_marker():
    """Check that marker-pdf is installed; raise clear error if not."""
    try:
        import marker
        return marker
    except ImportError:
        raise ImportError(
            "\n"
            "╔══════════════════════════════════════════════════════════╗\n"
            "║  marker-pdf is not installed.                          ║\n"
            "║                                                        ║\n"
            "║  Install:  pip install 'longparser[marker]'            ║\n"
            "║                                                        ║\n"
            "║  ⚠️  marker-pdf is licensed under GPL-3.0.             ║\n"
            "║  By installing it, you agree to GPL terms for that     ║\n"
            "║  component. LongParser core remains MIT-licensed.      ║\n"
            "╚══════════════════════════════════════════════════════════╝\n"
        )


class MarkerExtractor(BaseExtractor):
    """Extractor using Marker for high-fidelity output.
    
    Includes soft-cap logic for running on CPU to prevent infinite hangs.
    """

    extractor_type = ExtractorType.MARKER
    version = "1.0.0"

    def __init__(self):
        """Initialize and verify marker-pdf is available."""
        _require_marker()
        
        # Check for GPU
        try:
            import torch
            if not torch.cuda.is_available() and not torch.backends.mps.is_available():
                logger.warning(
                    "⚠️  Marker is running on CPU — expect 5-10× slower extraction. "
                    "A soft cap of 10 pages is enforced by default. "
                    "Set `force_marker_cpu=True` to bypass this."
                )
        except ImportError:
            pass
            
        logger.info("Marker backend initialized")

    def extract(
        self,
        file_path: Path,
        config: ProcessingConfig,
        page_numbers: Optional[List[int]] = None,
    ) -> Tuple[Document, ExtractionMetadata]:
        """Extract a PDF using Marker."""
        from marker.convert import convert_single_pdf
        from marker.models import load_all_models
        from marker.settings import settings
        import fitz  # PyMuPDF is a marker dependency anyway

        file_path = Path(file_path)
        logger.info("Extracting with Marker: %s", file_path.name)

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Marker backend only supports PDF files, got: {file_path.suffix}")

        pdf_doc = fitz.open(str(file_path))
        total_pages = len(pdf_doc)
        pdf_doc.close()

        # Soft cap logic for CPU
        try:
            import torch
            is_cpu = not torch.cuda.is_available() and not torch.backends.mps.is_available()
        except ImportError:
            is_cpu = True

        if is_cpu and not config.force_marker_cpu and total_pages > 10:
            if page_numbers is None or len(page_numbers) > 10:
                raise RuntimeError(
                    f"Marker CPU Soft Cap exceeded. Document has {total_pages} pages "
                    f"(limit: 10). Extraction will take too long on CPU. "
                    f"Set config.force_marker_cpu=True to override."
                )

        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]

        # Load models (cached internally by Marker)
        model_lst = load_all_models()
        
        # Convert
        full_text, images, out_meta = convert_single_pdf(
            str(file_path),
            model_lst,
            max_pages=settings.MAX_PAGES if not page_numbers else len(page_numbers),
            langs=config.languages if config.languages else None,
            batch_multiplier=settings.BATCH_MULTIPLIER,
            start_page=page_numbers[0] if page_numbers else None
        )

        # Map to LongParser Document
        # Note: Marker's output is flat markdown, so we do a fast mapping
        # similar to PyMuPDFExtractor.
        document = self._markdown_to_document(
            md_text=full_text,
            file_path=file_path,
            file_hash=file_hash,
            total_pages=total_pages,
        )

        meta = ExtractionMetadata(
            strategy_used="marker",
            ocr_backend_used="surya (marker)",
        )

        return document, meta

    def _markdown_to_document(
        self,
        md_text: str,
        file_path: Path,
        file_hash: str,
        total_pages: int,
    ) -> Document:
        """Convert Marker's markdown into a LongParser Document."""
        metadata = DocumentMetadata(
            source_file=str(file_path),
            file_hash=file_hash,
            total_pages=total_pages,
        )

        pages: list[Page] = []
        blocks: list[Block] = []
        
        lines = md_text.strip().split("\n")
        order_idx = 0
        
        # Fast parse
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
                
            block_type = BlockType.PARAGRAPH
            heading_level = None
            
            if stripped.startswith("#"):
                block_type = BlockType.HEADING
                heading_level = min(len(stripped) - len(stripped.lstrip("#")), 6)
                stripped = stripped.lstrip("#").strip()
            elif stripped.startswith(("- ", "* ")):
                block_type = BlockType.LIST_ITEM
                stripped = stripped.lstrip("-* ").strip()
                
            blocks.append(Block(
                type=block_type,
                text=stripped,
                order_index=order_idx,
                heading_level=heading_level,
                provenance=Provenance(
                    source_file=str(file_path),
                    page_number=1, # Marker loses page boundaries in its markdown string
                    bbox=BoundingBox(x0=0, y0=0, x1=0, y1=0),
                    extractor=self.extractor_type,
                    extractor_version=self.version,
                ),
                confidence=Confidence(overall=0.9),
            ))
            order_idx += 1

        pages.append(Page(
            page_number=1,
            width=612.0,
            height=792.0,
            blocks=blocks,
            profile=PageProfile(page_number=1, layout_confidence=0.9)
        ))

        return Document(metadata=metadata, pages=pages)

    def extract_page(
        self,
        file_path: Path,
        page_number: int,
        config: ProcessingConfig,
    ) -> Page:
        doc, _ = self.extract(file_path, config, page_numbers=[page_number])
        return doc.pages[0] if doc.pages else Page(page_number=page_number, width=0, height=0)

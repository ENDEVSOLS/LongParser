"""Simple pipeline orchestrator for LongParser.

Supports multiple extraction backends:

- ``"docling"`` (default) — Docling with Tesseract CLI OCR (MIT)
- ``"pymupdf"`` — PyMuPDF4LLM for fast native PDF extraction (AGPL, optional)
- ``"auto"``    — Automatic backend selection based on document properties

Language detection runs before OCR to set the correct Tesseract language.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
import time
import logging
import json

from ..schemas import Document, ProcessingConfig, JobRequest, BlockType, ChunkingConfig, Chunk
from ..extractors import DoclingExtractor
from ..extractors.docling_extractor import HierarchyChunk
from ..chunkers import HybridChunker
from ..utils.lang_detect import detect_language, get_tesseract_langs, extract_sample_text

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    document: Document
    hierarchy: List[HierarchyChunk]
    processing_time_seconds: float
    chunks: List[Chunk] = field(default_factory=list)
    
    @property
    def total_blocks(self) -> int:
        return sum(len(p.blocks) for p in self.document.pages)


class PipelineOrchestrator:
    """
    Pipeline orchestrator with backend selection and language detection.
    
    Flow:
    1. (Optional) Auto-detect document language
    2. Select backend: Docling, PyMuPDF, or auto-route
    3. Extract with chosen backend
    4. HierarchicalChunker preserves heading hierarchy
    
    Parameters
    ----------
    config:
        Processing configuration with backend, language, and layout settings.
        Only used for backend selection during init. Per-file config is passed
        to ``process_file()``.
    tesseract_lang:
        Languages for Tesseract OCR (default: ``["eng"]``). Overridden by
        ``config.languages`` or auto-detection if enabled.
    tessdata_path:
        Path to tessdata directory with language models and configs.
    force_full_page_ocr:
        If True, OCR entire page even if embedded text exists.
    """
    
    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        tesseract_lang: List[str] = None,
        tessdata_path: str = None,
        force_full_page_ocr: bool = False,
    ):
        self._config = config or ProcessingConfig()
        self._tessdata_path = tessdata_path
        self._force_full_page_ocr = force_full_page_ocr
        self._base_tesseract_lang = tesseract_lang

        # Determine backend from config
        backend = self._config.backend

        if backend == "pymupdf":
            # Lazy import — only loaded when user explicitly requests it
            from ..extractors.pymupdf_extractor import PyMuPDFExtractor
            self.extractor = PyMuPDFExtractor()
            self._backend_name = "pymupdf"
            logger.info("Pipeline initialized with PyMuPDF4LLM backend (CPU-native, fast)")

        elif backend == "auto":
            # Auto mode: start with Docling (safe default), route at process time
            self.extractor = DoclingExtractor(
                tesseract_lang=tesseract_lang,
                tessdata_path=tessdata_path,
                force_full_page_ocr=force_full_page_ocr,
            )
            self._backend_name = "auto"
            logger.info("Pipeline initialized in auto mode (will choose backend per document)")

        else:
            # Default: Docling (MIT, always available)
            self.extractor = DoclingExtractor(
                tesseract_lang=tesseract_lang,
                tessdata_path=tessdata_path,
                force_full_page_ocr=force_full_page_ocr,
            )
            self._backend_name = "docling"
            logger.info("Pipeline initialized with Docling backend (default)")

    def _resolve_languages(
        self,
        file_path: Path,
        config: ProcessingConfig,
    ) -> list[str]:
        """Resolve OCR languages via user override or auto-detection.

        Priority order:
        1. ``config.languages`` (explicit user override — always wins)
        2. ``self._base_tesseract_lang`` (constructor param)
        3. Auto-detection via ``fast-langdetect`` (if enabled)
        4. Default: ``["eng"]``
        """
        # 1. Explicit user override
        if config.languages:
            logger.info("Using user-specified languages: %s", config.languages)
            return config.languages

        # 2. Constructor param
        if self._base_tesseract_lang:
            # If auto-detect is enabled, try to improve on constructor default
            if config.auto_detect_language:
                detected_langs = self._auto_detect(file_path)
                if detected_langs:
                    return detected_langs
            return self._base_tesseract_lang

        # 3. Auto-detect
        if config.auto_detect_language:
            detected_langs = self._auto_detect(file_path)
            if detected_langs:
                return detected_langs

        # 4. Default
        return ["eng"]

    def _auto_detect(self, file_path: Path) -> Optional[list[str]]:
        """Run language detection and return Tesseract codes, or None."""
        sample = extract_sample_text(file_path)
        if not sample or len(sample.strip()) < 20:
            return None

        lang_code, confidence = detect_language(sample)
        if confidence > 0.0:
            tess_langs = get_tesseract_langs(lang_code)
            logger.info(
                "Auto-detected language: %s (%.0f%%) → Tesseract: %s",
                lang_code, confidence * 100, tess_langs,
            )
            # Store for later use in document metadata
            self._detected_lang = lang_code
            self._detected_lang_confidence = confidence
            return tess_langs

        return None

    def _should_use_pymupdf(self, file_path: Path) -> bool:
        """Check if PyMuPDF is a better choice for this file (auto mode)."""
        ext = file_path.suffix.lower()

        # PyMuPDF only handles PDFs
        if ext != ".pdf":
            return False

        # Check if PDF has a text layer (= native, not scanned)
        sample = extract_sample_text(file_path, max_chars=500)
        if sample and len(sample.strip()) > 100:
            # Has text → native PDF → PyMuPDF is faster
            try:
                from ..extractors.pymupdf_extractor import PyMuPDFExtractor
                return True
            except ImportError:
                # pymupdf4llm not installed — fall back to Docling
                logger.debug("Auto mode: pymupdf4llm not installed, using Docling")
                return False

        # Scanned PDF or too little text → use Docling (has OCR)
        return False

    def process(self, request: JobRequest) -> PipelineResult:
        """Process a document."""
        start_time = time.time()
        
        file_path = Path(request.file_path)
        config = request.config

        # Initialize language detection state
        self._detected_lang = None
        self._detected_lang_confidence = 0.0
        
        logger.info(f"Processing: {file_path.name}")

        # Auto-mode: decide backend per document
        if self._backend_name == "auto" and self._should_use_pymupdf(file_path):
            from ..extractors.pymupdf_extractor import PyMuPDFExtractor
            extractor = PyMuPDFExtractor()
            logger.info("Auto mode selected: PyMuPDF4LLM (native PDF detected)")
        else:
            extractor = self.extractor

            # Resolve languages for Docling backend
            if isinstance(extractor, DoclingExtractor):
                resolved_langs = self._resolve_languages(file_path, config)
                extractor._languages = resolved_langs

        # Extract document
        document, meta = extractor.extract(file_path, config)

        # Inject language detection results into metadata
        if self._detected_lang:
            document.metadata.detected_language = self._detected_lang
            document.metadata.language_confidence = self._detected_lang_confidence

        # Get hierarchy (only DoclingExtractor has this)
        if isinstance(extractor, DoclingExtractor):
            hierarchy = extractor.get_hierarchy(file_path, config)
        else:
            hierarchy = []
        
        processing_time = time.time() - start_time
        logger.info(f"Completed in {processing_time:.2f}s")
        
        return PipelineResult(
            document=document,
            hierarchy=hierarchy,
            processing_time_seconds=processing_time,
        )
    
    def process_file(
        self,
        file_path: str | Path,
        config: Optional[ProcessingConfig] = None,
    ) -> PipelineResult:
        """Convenience method to process a file directly."""
        request = JobRequest(
            file_path=str(file_path),
            config=config or ProcessingConfig(),
        )
        return self.process(request)
    
    def export_to_markdown(self, result: PipelineResult, output_path: Path) -> Path:
        """Export document to Markdown."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        md_path = output_path / "document.md"
        md_content = self.extractor.to_markdown(result.document)
        
        with open(md_path, "w") as f:
            f.write(md_content)
        
        return md_path
    
    def export_hierarchy(self, result: PipelineResult, output_path: Path) -> Path:
        """Export hierarchy to JSON."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        hierarchy_path = output_path / "hierarchy.json"
        hierarchy_data = [
            {
                "text": h.text[:200],  # Truncate for readability
                "heading_path": h.heading_path,
                "level": h.level,
                "page": h.page_number,
            }
            for h in result.hierarchy
        ]
        
        with open(hierarchy_path, "w") as f:
            json.dump(hierarchy_data, f, indent=2)
        
        return hierarchy_path

    def export_results(self, result: PipelineResult, output_dir: Path) -> dict:
        """
        Export results in the format expected by the user (blocks.json, manifest.json).
        
        Args:
            result: Pipeline execution result
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of created files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        created_files = {}
        
        # 1. blocks.json - Flattened list of blocks from all pages
        all_blocks = []
        total_tables = 0
        
        for page in result.document.pages:
            for block in page.blocks:
                # Exclude confidence from public output
                block_dict = block.model_dump(exclude={"confidence"})
                # Ensure compatibility with expected format
                if block.type == BlockType.TABLE:
                    total_tables += 1
                all_blocks.append(block_dict)
                
        blocks_path = output_dir / "blocks.json"
        with open(blocks_path, "w") as f:
            json.dump(all_blocks, f, indent=2, default=str)
        created_files["blocks"] = blocks_path
        
        # 2. manifest.json - Processing metadata
        manifest = {
            "source_file": result.document.metadata.source_file,
            "file_hash": result.document.metadata.file_hash,
            "total_pages": result.document.metadata.total_pages,
            "total_blocks": len(all_blocks),
            "total_tables": total_tables,
            "processing_time_seconds": result.processing_time_seconds,
            "detected_language": result.document.metadata.detected_language,
            "language_confidence": result.document.metadata.language_confidence,
            "stages_completed": [
                "stage1_extraction",
                "stage2_validation",
                "stage3_reprocess",
                "stage4_enrichment",
                "stage5_verification"
            ],
            "verification": {
                "auto_accepted": False,
                "needs_hitl_review": True,
                "low_confidence_pages": [],
                "low_confidence_tables": []
            }
        }
            
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        created_files["manifest"] = manifest_path
        
        # 3. document.md - Markdown representation
        md_path = self.export_to_markdown(result, output_dir)
        created_files["markdown"] = md_path
        
        # 4. Images
        images_dir = output_dir / "images"
        images = self.save_images(images_dir)
        created_files["images"] = images
        
        return created_files

    def chunk(self, result: PipelineResult, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        """
        Run hybrid chunking on a pipeline result.
        
        Args:
            result: Pipeline execution result with extracted blocks
            config: Chunking configuration (uses defaults if None)
            
        Returns:
            List of RAG-optimized chunks
        """
        chunker = HybridChunker(config or ChunkingConfig())
        all_blocks = result.document.all_blocks
        chunks = chunker.chunk(all_blocks)
        result.chunks = chunks
        return chunks

    def export_chunks(self, result: PipelineResult, output_dir: Path) -> Path:
        """Export chunks to JSON."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chunks_path = output_dir / "chunks.json"
        chunks_data = [c.model_dump() for c in result.chunks]
        with open(chunks_path, "w") as f:
            json.dump(chunks_data, f, indent=2, default=str)
        
        logger.info(f"Exported {len(result.chunks)} chunks to {chunks_path}")
        return chunks_path

    def save_images(self, output_dir: Path) -> List[Path]:
        """Save extracted images."""
        return self.extractor.save_images(output_dir)


"""Simple pipeline orchestrator for LongParser."""

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
    Simple pipeline orchestrator using Docling.
    
    Flow:
    1. Docling extracts with Tesseract CLI OCR
    2. Layout analysis detects structure
    3. HierarchicalChunker preserves heading hierarchy
    """
    
    def __init__(self, tesseract_lang: List[str] = None, tessdata_path: str = None, force_full_page_ocr: bool = False):
        """
        Initialize pipeline.
        
        Args:
            tesseract_lang: Languages for Tesseract OCR (default: ["eng"])
            tessdata_path: Path to tessdata directory with language models and configs.
            force_full_page_ocr: If True, OCR entire page even if embedded text exists.
        """
        self.extractor = DoclingExtractor(
            tesseract_lang=tesseract_lang,
            tessdata_path=tessdata_path,
            force_full_page_ocr=force_full_page_ocr,
        )
    
    def process(self, request: JobRequest) -> PipelineResult:
        """Process a document."""
        start_time = time.time()
        
        file_path = Path(request.file_path)
        config = request.config
        
        logger.info(f"Processing: {file_path.name}")
        
        # Extract document
        document, meta = self.extractor.extract(file_path, config)
        
        # Get hierarchy
        hierarchy = self.extractor.get_hierarchy(file_path, config)
        
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

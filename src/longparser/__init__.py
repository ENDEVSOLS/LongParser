"""LongParser — Privacy-first document intelligence engine for RAG.

LongParser converts complex documents (PDFs, DOCX, PPTX, XLSX, CSV) into
AI-ready structured output via a 5-stage extraction pipeline::

    Extract → Validate → HITL Review → Chunk → Embed → Index

Built by ENDEVSOLS for production RAG pipelines.

Quick start::

    from longparser import DocumentPipeline, ProcessingConfig

    pipeline = DocumentPipeline(ProcessingConfig())
    result = pipeline.process_file("document.pdf")
    print(result.chunks[0].text)

For the full REST API server::

    uv run uvicorn longparser.server.app:app --reload --port 8000

See :class:`~longparser.pipeline.DocumentPipeline` for the main SDK entry
point and :mod:`longparser.server` for the REST API layer.
"""

from __future__ import annotations

__version__ = "0.1.3"
__author__ = "ENDEVSOLS Team"
__license__ = "MIT"

from .schemas import (
    Block,
    BlockFlags,
    BlockType,
    BoundingBox,
    Chunk,
    ChunkingConfig,
    Confidence,
    Document,
    DocumentMetadata,
    ExtractionMetadata,
    ExtractorType,
    JobRequest,
    JobResult,
    Page,
    PageProfile,
    ProcessingConfig,
    Provenance,
    Table,
    TableCell,
)

# Heavy dependencies (docling, motor, etc.) are imported lazily so that
# ``import longparser`` and ``from longparser.schemas import ...`` work
# in environments where optional extras are not installed.
def __getattr__(name: str):
    """Lazy import shim for optional heavy dependencies."""
    if name == "DoclingExtractor":
        from .extractors import DoclingExtractor
        return DoclingExtractor
    if name == "PipelineOrchestrator":
        from .pipeline import PipelineOrchestrator
        return PipelineOrchestrator
    if name == "DocumentPipeline":
        from .pipeline import DocumentPipeline
        return DocumentPipeline
    if name == "PipelineResult":
        from .pipeline import PipelineResult
        return PipelineResult
    if name == "HybridChunker":
        from .chunkers import HybridChunker
        return HybridChunker
    raise AttributeError(f"module 'longparser' has no attribute {name!r}")


__all__ = [
    # Meta
    "__version__",
    "__author__",
    "__license__",
    # Schemas — always available (no heavy deps)
    "Document",
    "Page",
    "Block",
    "Table",
    "TableCell",
    "BlockType",
    "ExtractorType",
    "ProcessingConfig",
    "BoundingBox",
    "Provenance",
    "Confidence",
    "BlockFlags",
    "DocumentMetadata",
    "PageProfile",
    "ExtractionMetadata",
    "ChunkingConfig",
    "Chunk",
    "JobRequest",
    "JobResult",
    # Lazily imported (require extras)
    "DoclingExtractor",
    "PipelineOrchestrator",
    "DocumentPipeline",
    "PipelineResult",
    "HybridChunker",
]

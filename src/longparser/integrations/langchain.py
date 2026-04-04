"""LangChain integration adapter for LongParser.

Provides ``LongParserLoader``, a LangChain-compatible document loader that
wraps the LongParser extraction pipeline.

Install the extra to use this adapter::

    pip install clean_rag[langchain]

Usage::

    from longparser.integrations.langchain import LongParserLoader

    loader = LongParserLoader("report.pdf")
    docs = loader.load()        # list[langchain_core.documents.Document]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document as LCDocument

from ..schemas import ProcessingConfig, ChunkingConfig

_INSTALL_MSG = (
    "langchain-core is required for the LangChain adapter. "
    "Install it with:  pip install clean_rag[langchain]"
)


def _import_langchain():
    """Lazy import langchain-core with a clear install message."""
    try:
        from langchain_core.document_loaders import BaseLoader
        from langchain_core.documents import Document as LCDocument
        return BaseLoader, LCDocument
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


class LongParserLoader:
    """LangChain document loader backed by the LongParser pipeline.

    Converts a file (PDF, DOCX, PPTX, XLSX, CSV) into a list of
    LangChain ``Document`` objects — one per chunk (if chunking is
    enabled) or one per block.

    Parameters
    ----------
    file_path:
        Path to the input file.
    config:
        LongParser ``ProcessingConfig``. Uses defaults if ``None``.
    chunking_config:
        LongParser ``ChunkingConfig``. If provided, the loader yields
        one ``Document`` per chunk; otherwise one per block.
    tesseract_lang:
        Languages for Tesseract OCR (e.g. ``["eng", "urd"]``).
    tessdata_path:
        Path to tessdata directory.
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        config: Optional[ProcessingConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        tesseract_lang: list[str] | None = None,
        tessdata_path: str | None = None,
    ) -> None:
        # Validate langchain is available at construction time
        BaseLoader, _ = _import_langchain()
        self._BaseLoader = BaseLoader

        self.file_path = Path(file_path)
        self.config = config or ProcessingConfig()
        self.chunking_config = chunking_config
        self.tesseract_lang = tesseract_lang
        self.tessdata_path = tessdata_path

    # ---- LangChain interface -------------------------------------------------

    def load(self) -> list["LCDocument"]:
        """Load and return all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator["LCDocument"]:
        """Lazily yield LangChain ``Document`` objects."""
        _, LCDocument = _import_langchain()

        from ..pipeline import PipelineOrchestrator

        pipeline = PipelineOrchestrator(
            tesseract_lang=self.tesseract_lang,
            tessdata_path=self.tessdata_path,
        )
        result = pipeline.process_file(self.file_path, config=self.config)

        # If chunking is requested, yield one doc per chunk
        if self.chunking_config is not None:
            chunks = pipeline.chunk(result, config=self.chunking_config)
            for chunk in chunks:
                yield LCDocument(
                    page_content=chunk.text,
                    metadata={
                        "source": str(self.file_path),
                        "chunk_id": chunk.chunk_id,
                        "chunk_type": chunk.chunk_type,
                        "section_path": chunk.section_path,
                        "page_numbers": chunk.page_numbers,
                        "token_count": chunk.token_count,
                        "equation_detected": chunk.equation_detected,
                    },
                )
            return

        # Otherwise, yield one doc per block
        for page in result.document.pages:
            for block in page.blocks:
                yield LCDocument(
                    page_content=block.text,
                    metadata={
                        "source": str(self.file_path),
                        "block_id": block.block_id,
                        "block_type": block.type.value,
                        "heading_level": block.heading_level,
                        "hierarchy_path": block.hierarchy_path,
                        "page_number": page.page_number,
                        "confidence": block.confidence.overall,
                    },
                )


__all__ = ["LongParserLoader"]

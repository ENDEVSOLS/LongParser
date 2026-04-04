"""LlamaIndex integration adapter for LongParser.

Provides ``LongParserReader``, a LlamaIndex-compatible document reader
that wraps the LongParser extraction pipeline.

Install the extra to use this adapter::

    pip install clean_rag[llamaindex]

Usage::

    from longparser.integrations.llamaindex import LongParserReader

    reader = LongParserReader()
    docs = reader.load_data("report.pdf")  # list[llama_index.core.Document]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core import Document as LIDocument

from ..schemas import ProcessingConfig, ChunkingConfig

_INSTALL_MSG = (
    "llama-index-core is required for the LlamaIndex adapter. "
    "Install it with:  pip install clean_rag[llamaindex]"
)


def _import_llamaindex():
    """Lazy import llama-index-core with a clear install message."""
    try:
        from llama_index.core.readers.base import BaseReader
        from llama_index.core import Document as LIDocument
        return BaseReader, LIDocument
    except ImportError as exc:
        raise ImportError(_INSTALL_MSG) from exc


class LongParserReader:
    """LlamaIndex document reader backed by the LongParser pipeline.

    Converts a file (PDF, DOCX, PPTX, XLSX, CSV) into a list of
    LlamaIndex ``Document`` objects — one per chunk (if chunking is
    enabled) or one per block.

    Parameters
    ----------
    config:
        LongParser ``ProcessingConfig``. Uses defaults if ``None``.
    chunking_config:
        LongParser ``ChunkingConfig``. If provided, the reader yields
        one ``Document`` per chunk; otherwise one per block.
    tesseract_lang:
        Languages for Tesseract OCR (e.g. ``["eng", "urd"]``).
    tessdata_path:
        Path to tessdata directory.
    """

    def __init__(
        self,
        *,
        config: Optional[ProcessingConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        tesseract_lang: list[str] | None = None,
        tessdata_path: str | None = None,
    ) -> None:
        # Validate llama-index is available at construction time
        BaseReader, _ = _import_llamaindex()
        self._BaseReader = BaseReader

        self.config = config or ProcessingConfig()
        self.chunking_config = chunking_config
        self.tesseract_lang = tesseract_lang
        self.tessdata_path = tessdata_path

    # ---- LlamaIndex interface ------------------------------------------------

    def load_data(
        self,
        file: str | Path,
        extra_info: dict | None = None,
    ) -> list["LIDocument"]:
        """Load data from a file and return LlamaIndex ``Document`` objects.

        Parameters
        ----------
        file:
            Path to the input file.
        extra_info:
            Additional metadata to merge into every document.

        Returns
        -------
        list[Document]
            One ``Document`` per chunk or per block.
        """
        _, LIDocument = _import_llamaindex()

        from ..pipeline import PipelineOrchestrator

        file = Path(file)
        pipeline = PipelineOrchestrator(
            tesseract_lang=self.tesseract_lang,
            tessdata_path=self.tessdata_path,
        )
        result = pipeline.process_file(file, config=self.config)

        docs: list[LIDocument] = []
        base_meta = {"source": str(file), **(extra_info or {})}

        # If chunking is requested, yield one doc per chunk
        if self.chunking_config is not None:
            chunks = pipeline.chunk(result, config=self.chunking_config)
            for chunk in chunks:
                docs.append(
                    LIDocument(
                        text=chunk.text,
                        extra_info={
                            **base_meta,
                            "chunk_id": chunk.chunk_id,
                            "chunk_type": chunk.chunk_type,
                            "section_path": chunk.section_path,
                            "page_numbers": chunk.page_numbers,
                            "token_count": chunk.token_count,
                            "equation_detected": chunk.equation_detected,
                        },
                    )
                )
            return docs

        # Otherwise, yield one doc per block
        for page in result.document.pages:
            for block in page.blocks:
                docs.append(
                    LIDocument(
                        text=block.text,
                        extra_info={
                            **base_meta,
                            "block_id": block.block_id,
                            "block_type": block.type.value,
                            "heading_level": block.heading_level,
                            "hierarchy_path": block.hierarchy_path,
                            "page_number": page.page_number,
                            "confidence": block.confidence.overall,
                        },
                    )
                )

        return docs


__all__ = ["LongParserReader"]

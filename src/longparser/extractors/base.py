"""Abstract base class for document extractors."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..schemas import Document, Page, ProcessingConfig, ExtractorType


class BaseExtractor(ABC):
    """Base class for all document extractors."""

    extractor_type: ExtractorType
    version: str = "1.0.0"

    @abstractmethod
    def extract(
        self,
        file_path: Path,
        config: ProcessingConfig,
        page_numbers: Optional[list[int]] = None,
    ) -> Document:
        """
        Extract content from a document.

        Args:
            file_path: Path to the input file (PDF/image)
            config: Processing configuration
            page_numbers: Optional list of specific pages to process.
                         If None, process all pages.

        Returns:
            Document with extracted pages and blocks
        """
        pass

    @abstractmethod
    def extract_page(
        self,
        file_path: Path,
        page_number: int,
        config: ProcessingConfig,
    ) -> Page:
        """
        Extract a single page from a document.

        Args:
            file_path: Path to the input file
            page_number: 0-indexed page number
            config: Processing configuration

        Returns:
            Page with extracted blocks
        """
        pass

    def get_provenance_info(self) -> dict:
        """Get extractor provenance information."""
        return {
            "extractor": self.extractor_type,
            "extractor_version": self.version,
        }

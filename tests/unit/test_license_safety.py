"""License safety tests — ensure GPL/AGPL packages are never loaded by default.

These tests verify that importing ``longparser`` and using its default
pipeline does NOT load any GPL/AGPL-licensed package (pymupdf4llm, marker,
surya). This is critical to maintain LongParser's MIT license.
"""

import sys
import pytest


# Packages that must NEVER appear in sys.modules after a default import
_BLOCKED_MODULES = [
    "pymupdf4llm",
    "pymupdf",
    "fitz",           # PyMuPDF's internal module name
    "marker",
    "marker.converters",
    "surya",
    "surya.ocr",
]


def _clear_blocked_modules():
    """Remove any pre-loaded blocked modules from sys.modules."""
    for mod_name in list(sys.modules):
        for blocked in _BLOCKED_MODULES:
            if mod_name == blocked or mod_name.startswith(blocked + "."):
                del sys.modules[mod_name]


class TestLicenseSafety:
    """Verify that core imports do not load GPL/AGPL dependencies."""

    def test_import_longparser_does_not_load_agpl(self):
        """``import longparser`` must not load any GPL/AGPL module."""
        _clear_blocked_modules()

        import longparser  # noqa: F401

        for mod_name in _BLOCKED_MODULES:
            assert mod_name not in sys.modules, (
                f"GPL/AGPL module '{mod_name}' was loaded by 'import longparser'. "
                f"This violates the MIT license isolation. "
                f"Check __init__.py and extractors/__init__.py for stray imports."
            )

    def test_import_schemas_does_not_load_agpl(self):
        """``from longparser.schemas import ...`` must not load GPL/AGPL."""
        _clear_blocked_modules()

        from longparser.schemas import (  # noqa: F401
            ProcessingConfig, Document, Block, Chunk
        )

        for mod_name in _BLOCKED_MODULES:
            assert mod_name not in sys.modules, (
                f"GPL/AGPL module '{mod_name}' was loaded by schema import."
            )

    def test_processing_config_default_backend_is_docling(self):
        """Default backend must be 'docling' (MIT), not a GPL/AGPL backend."""
        from longparser.schemas import ProcessingConfig
        config = ProcessingConfig()

        # If backend field exists, it must default to docling
        backend = getattr(config, "backend", "docling")
        assert backend == "docling", (
            f"Default backend is '{backend}', expected 'docling'. "
            f"Defaulting to a GPL/AGPL backend would violate MIT license."
        )

    def test_pymupdf_extractor_not_in_extractors_init(self):
        """PyMuPDFExtractor must NOT be exported from extractors/__init__.py."""
        from longparser import extractors

        public_names = getattr(extractors, "__all__", dir(extractors))

        assert "PyMuPDFExtractor" not in public_names, (
            "PyMuPDFExtractor must NOT be in extractors/__init__.py. "
            "It must only be imported lazily when backend='pymupdf' is set."
        )

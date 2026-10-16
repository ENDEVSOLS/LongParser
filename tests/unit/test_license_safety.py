"""License safety tests — ensure GPL/AGPL packages are never loaded by default.

These tests verify that importing ``longparser`` and using its default
pipeline does NOT load any GPL/AGPL-licensed package (pymupdf4llm, marker,
surya). This is critical to maintain LongParser's MIT license.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _license_check_script():
    """Read the executable check from CI so tests cannot drift from its rules."""
    workflow = (_REPO_ROOT / ".github/workflows/license-check.yml").read_text()
    _, script = workflow.split("        run: |\n", maxsplit=1)
    return textwrap.dedent(script)


@pytest.mark.parametrize(
    "source,blocked",
    [
        ("import marker", True),
        ("import marker as backend", True),
        ("from marker import converters", True),
        ("import marker.converters", True),
        ("from marker.converters import pdf", True),
        ("import pymupdf", True),
        ("from pymupdf import open", True),
        ("import surya", True),
        ("from surya.ocr import run_ocr", True),
        ("import marker_utils", False),
        ("from markerish import parser", False),
        ("import document_marker", False),
        ("from longparser.schemas import ProcessingConfig", False),
    ],
)
def test_license_workflow_checks_core_imports(tmp_path, source, blocked):
    core = tmp_path / "src/longparser"
    core.mkdir(parents=True)
    (core / "core.py").write_text(source + "\n")

    result = subprocess.run(
        ["bash", "-c", _license_check_script()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == (1 if blocked else 0), result.stdout + result.stderr
    assert ("core.py" in result.stdout) == blocked


@pytest.mark.parametrize("filename", ["marker_extractor.py", "pymupdf_extractor.py"])
def test_license_workflow_allows_only_isolated_extractors(tmp_path, filename):
    core = tmp_path / "src/longparser"
    core.mkdir(parents=True)
    imports = "import marker\nfrom marker.converters import pdf\nimport pymupdf\nimport surya\n"
    (core / filename).write_text(imports)

    result = subprocess.run(
        ["bash", "-c", _license_check_script()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The same imports must still fail when another core file contains them.
    (core / "ordinary_extractor.py").write_text(imports)
    result = subprocess.run(
        ["bash", "-c", _license_check_script()],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ordinary_extractor.py" in result.stdout


def test_fresh_core_import_is_lightweight_and_license_safe():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT / "src")
    script = """
import sys
import longparser
from longparser.schemas import Block, Chunk, Document, ProcessingConfig

blocked = ("pymupdf4llm", "pymupdf", "fitz", "marker", "surya",
           "docling", "torch", "motor", "fastapi")
loaded = sorted(name for name in sys.modules
                if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked))
assert not loaded, f"Unexpected dependencies loaded: {loaded}"
assert ProcessingConfig().backend == "docling"
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

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

        from longparser.schemas import Block, Chunk, Document, ProcessingConfig  # noqa: F401

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

"""Optional PDF adapters against controlled external SDKs; no optional model weights."""

import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest

from longparser.schemas import ProcessingConfig


def sdk_module(monkeypatch, name, **attrs):
    sdk = ModuleType(name)
    sdk.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, sdk)


@pytest.fixture
def marker(monkeypatch, tmp_path):
    from longparser.extractors.marker_extractor import MarkerExtractor
    sdk_module(monkeypatch, "marker")
    torch = NS(cuda=NS(is_available=lambda: False), backends=NS(mps=NS(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    convert = Mock(return_value=("# Heading\n\nParagraph\n- List\n* Another\n", {}, {}))
    models = Mock(return_value=["controlled-model"])
    sdk_module(monkeypatch, "marker.convert", convert_single_pdf=convert)
    sdk_module(monkeypatch, "marker.models", load_all_models=models)
    sdk_module(monkeypatch, "marker.settings", settings=NS(MAX_PAGES=100, BATCH_MULTIPLIER=2))
    opened = MagicMock()
    opened.__len__ = Mock(return_value=2)
    sdk_module(monkeypatch, "fitz", open=Mock(return_value=opened))
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"controlled PDF boundary")
    return NS(extractor=MarkerExtractor(), path=path, convert=convert, models=models, opened=opened)


def test_marker_extract_mapping_and_page_selection(marker):
    doc, metadata = marker.extractor.extract(marker.path, ProcessingConfig(languages=["en"]), page_numbers=[1])
    assert [block.type.value for block in doc.pages[0].blocks] == ["heading", "paragraph", "list_item", "list_item"]
    assert doc.pages[0].blocks[0].heading_level == 1
    assert doc.metadata.total_pages == 2 and len(doc.metadata.file_hash) == 16
    assert metadata.strategy_used == "marker"
    assert marker.convert.call_args.kwargs == {"max_pages": 1, "langs": ["en"], "batch_multiplier": 2, "start_page": 1}
    assert marker.extractor.extract_page(marker.path, 1, ProcessingConfig()).blocks[0].text == "Heading"
    marker.opened.close.assert_called()


def test_marker_cpu_limit_override_and_nonpdf_rejection(marker, monkeypatch):
    marker.opened.__len__.return_value = 11
    with pytest.raises(RuntimeError, match="Soft Cap"):
        marker.extractor.extract(marker.path, ProcessingConfig())
    marker.models.assert_not_called()
    assert marker.extractor.extract(marker.path, ProcessingConfig(force_marker_cpu=True))[0].metadata.total_pages == 11
    with pytest.raises(ValueError, match="only supports PDF"):
        marker.extractor.extract(Path("sample.docx"), ProcessingConfig())
    monkeypatch.setitem(sys.modules, "torch", None)
    from longparser.extractors.marker_extractor import MarkerExtractor
    extractor = MarkerExtractor()
    with pytest.raises(RuntimeError, match="Soft Cap"):
        extractor.extract(marker.path, ProcessingConfig())


@pytest.fixture
def pymupdf(monkeypatch):
    sdk_module(monkeypatch, "pymupdf")
    sdk_module(monkeypatch, "pymupdf4llm")
    from longparser.extractors.pymupdf_extractor import PyMuPDFExtractor
    return PyMuPDFExtractor()


def test_pymupdf_markdown_roundtrip_tables_code_math_and_page_separator(pymupdf):
    text = "# Heading\n\n| A | B |\n|---|---|\n|one|two|\n- List\n1. Numbered\n```python\nprint(1)\n```\n$$x+y$$\n$$\nx-y\n$$\nParagraph"
    blocks = pymupdf._parse_markdown_blocks(text, 2, Path("sample.pdf"))
    assert [block.type.value for block in blocks] == ["heading", "table", "list_item", "list_item", "code", "equation", "equation", "paragraph"]
    assert [cell.text for cell in blocks[1].table.cells] == ["A", "B", "one", "two"]
    assert blocks[4].text == "print(1)"
    assert all(block.provenance.page_number == 2 for block in blocks)
    markdown = pymupdf.to_markdown(NS(pages=[NS(blocks=blocks)]))
    assert "# Heading" in markdown and "```\nprint(1)\n```" in markdown
    assert pymupdf._split_by_pages("first\fsecond", 3) == ["first", "second", ""]
    assert pymupdf._split_by_pages("first", 2) == ["first"]
    assert pymupdf._parse_table(["|---|---|"]).n_rows == 0


def test_pymupdf_images_write_bytes_and_ignore_failed_extraction(pymupdf, tmp_path):
    pdf = MagicMock()
    pdf.__len__ = Mock(return_value=1)
    pdf.__getitem__ = Mock(return_value=NS(get_images=lambda full: [(1,), (2,), (3,)]))
    pdf.extract_image.side_effect = [{"image": b"image-content", "ext": "png"}, None, ValueError("broken image")]
    pymupdf._extract_images(pdf, ProcessingConfig())
    paths = pymupdf.save_images(tmp_path)
    assert len(paths) == 1 and paths[0].read_bytes() == b"image-content"
    assert paths[0].name == "page_001_img_00.png"

"""Adapters execute real extraction/chunking and map external document types."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from longparser.schemas import ChunkingConfig, ProcessingConfig


@pytest.fixture
def converter_fixture(monkeypatch, tmp_path):
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    extractor = importlib.import_module("longparser.extractors.docling_extractor")
    document = DoclingDocument(name="fixture")
    document.add_heading(text="Return Policy", level=1)
    document.add_text(label=DocItemLabel.TEXT, text="Unused items may be returned within thirty days.")
    converter = Mock()
    converter.convert.return_value = SimpleNamespace(document=document)
    monkeypatch.setattr(extractor, "DocumentConverter", Mock(return_value=converter))
    detector = ModuleType("fast_langdetect")
    detector.detect = lambda text: {"lang": "en", "score": 1.0}
    monkeypatch.setitem(sys.modules, "fast_langdetect", detector)
    filename = tmp_path / "report.docx"
    filename.write_bytes(b"synthetic SDK boundary input")
    core = ModuleType("llama_index.core")

    class LlamaDocument:
        def __init__(self, text, extra_info):
            self.text, self.extra_info = text, extra_info

    core.Document = LlamaDocument
    readers = ModuleType("llama_index.core.readers.base")
    readers.BaseReader = object
    parent = ModuleType("llama_index")
    parent.core = core
    monkeypatch.setitem(sys.modules, "llama_index", parent)
    monkeypatch.setitem(sys.modules, "llama_index.core", core)
    monkeypatch.setitem(sys.modules, "llama_index.core.readers.base", readers)
    return filename


@pytest.mark.parametrize("adapter", ["langchain", "llamaindex"])
@pytest.mark.parametrize("chunking", [False, True])
def test_document_adapters_preserve_content_and_provenance(converter_fixture, adapter, chunking):
    filename = converter_fixture
    config = ProcessingConfig(do_ocr=False, formula_ocr=False, formula_mode="fast", auto_detect_language=False)
    chunk_config = ChunkingConfig(overlap_blocks=0) if chunking else None
    if adapter == "langchain":
        from longparser.integrations.langchain import LongParserLoader

        documents = LongParserLoader(filename, config=config, chunking_config=chunk_config).load()
        texts = [document.page_content for document in documents]
        metadata = [document.metadata for document in documents]
    else:
        from longparser.integrations.llamaindex import LongParserReader

        documents = LongParserReader(config=config, chunking_config=chunk_config).load_data(filename, extra_info={"source_type": "policy"})
        texts = [document.text for document in documents]
        metadata = [document.extra_info for document in documents]
        assert all(item["source_type"] == "policy" for item in metadata)
    assert "Unused items may be returned within thirty days." in "\n".join(texts)
    assert all(item["source"] == str(filename) for item in metadata)
    if chunking:
        assert len(documents) == 1
        assert metadata[0]["chunk_id"] and metadata[0]["page_numbers"] == [0]
        assert metadata[0]["token_count"] > 0
    else:
        assert len(documents) == 2
        assert metadata[0]["block_type"] == "heading"
        assert metadata[1]["confidence"] == 1.0


@pytest.mark.parametrize("adapter,module", [("langchain", "langchain_core"), ("llamaindex", "llama_index")])
def test_missing_adapter_sdk_gives_install_guidance(monkeypatch, adapter, module):
    real_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith(module):
            raise ImportError("external adapter unavailable")
        return real_import(name, *args, **kwargs)

    integrations = importlib.import_module("longparser.integrations")
    target = importlib.import_module(f"longparser.integrations.{adapter}")
    monkeypatch.setattr(builtins, "__import__", missing)
    assert not getattr(integrations, f"_has_{adapter}")()
    with pytest.raises(ImportError, match="Install it with"):
        getattr(target, f"_import_{adapter}")()


def test_integration_sdk_availability_paths(converter_fixture):
    from longparser.integrations import _has_langchain, _has_llamaindex

    assert _has_langchain() and _has_llamaindex()

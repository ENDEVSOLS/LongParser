"""Controlled extractor helper/guard tests; no converters, weights or inference."""

from __future__ import annotations

import builtins
import socket
import sys
from types import SimpleNamespace

import pytest

from longparser.schemas import BlockType, Document, DocumentMetadata, ExtractorType, Page
from tests.pipeline_doubles import install_docling_sdk_stubs


@pytest.fixture(autouse=True)
def offline_import_boundary(monkeypatch):
    install_docling_sdk_stubs(monkeypatch)

    def deny_network(*args, **kwargs):
        raise AssertionError("Extractor unit tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", deny_network)


@pytest.fixture
def docling_env(monkeypatch):
    from longparser.extractors import docling_extractor

    # Match cached-module bindings to the controlled external SDK types too.
    sdk = sys.modules["docling_core.types.doc"]
    for name in ["SectionHeaderItem", "TitleItem", "TableItem", "ListItem", "PictureItem"]:
        monkeypatch.setattr(docling_extractor, name, getattr(sdk, name))
    extractor = docling_extractor.DoclingExtractor()
    return SimpleNamespace(module=docling_extractor, extractor=extractor)


def sdk_item(cls, **fields):
    item = cls()
    for name, value in fields.items():
        setattr(item, name, value)
    return item


def test_base_extractor_rejects_incomplete_subclass():
    from longparser.extractors.base import BaseExtractor

    class Incomplete(BaseExtractor):
        def extract(self, *args, **kwargs):
            return Document(metadata=DocumentMetadata(source_file="synthetic.txt"))

    with pytest.raises(TypeError, match="extract_page"):
        Incomplete()


def test_minimal_extractor_contract_and_provenance():
    from longparser.extractors.base import BaseExtractor

    class Complete(BaseExtractor):
        extractor_type = ExtractorType.DOCLING
        version = "test-version"

        def extract(self, *args, **kwargs):
            return Document(metadata=DocumentMetadata(source_file="synthetic.txt"))

        def extract_page(self, *args, **kwargs):
            return Page(page_number=1, width=100, height=200)

    extractor = Complete()
    assert extractor.extract(None, None).metadata.source_file == "synthetic.txt"
    assert extractor.extract_page(None, 0, None).page_number == 1
    assert extractor.get_provenance_info() == {"extractor": ExtractorType.DOCLING,
                                               "extractor_version": "test-version"}


@pytest.mark.parametrize("sdk_name,expected", [("SectionHeaderItem", BlockType.HEADING),
    ("TitleItem", BlockType.HEADING), ("TableItem", BlockType.TABLE),
    ("ListItem", BlockType.LIST_ITEM), ("PictureItem", BlockType.FIGURE)])
def test_sdk_item_classification(docling_env, sdk_name, expected):
    env = docling_env
    item = sdk_item(getattr(env.module, sdk_name), text="Synthetic item")
    assert env.extractor._determine_block_type(item, level=2) == (
        expected, 2 if expected == BlockType.HEADING else None)


@pytest.mark.parametrize("label,expected", [("paragraph", BlockType.PARAGRAPH),
    ("caption", BlockType.CAPTION), ("page_footer", BlockType.FOOTER),
    ("footnote", BlockType.FOOTER), ("page_header", BlockType.HEADER),
    ("section_header", BlockType.PARAGRAPH), ("formula", BlockType.EQUATION),
    ("code", BlockType.CODE), ("title", BlockType.HEADING), (None, BlockType.PARAGRAPH)])
def test_label_fallback_classification(docling_env, label, expected):
    result = docling_env.extractor._determine_block_type(SimpleNamespace(label=label), level=0)
    assert result == (expected, 1 if expected == BlockType.HEADING else None)


@pytest.mark.parametrize("sdk_name", ["SectionHeaderItem", "TitleItem"])
@pytest.mark.parametrize("inferred,expected", [(3, (BlockType.HEADING, 3)),
                                           (-1, (BlockType.PARAGRAPH, None))])
def test_heading_inferred_level_or_demotion(docling_env, sdk_name, inferred, expected):
    env = docling_env
    item = sdk_item(getattr(env.module, sdk_name), text="Example heading")
    assert env.extractor._determine_block_type(item, 1, {"Example heading": inferred}) == expected


@pytest.mark.parametrize("text,marker,kind", [("1. Introduction", "1.", "numeric"),
    ("2.3 Methods", "2.3", "numeric"), ("A. Details", "A.", "alpha"),
    ("IV. Results", "IV.", "roman"), ("", None, "other"),
    ("— Unmarked", None, "other"), ("Introduction", None, "other"),
    ("Plain heading", "Plain", "other")])
def test_heading_marker_recognition_records_existing_behavior(docling_env, text, marker, kind):
    extractor = docling_env.extractor
    assert extractor._extract_marker(text) == marker
    assert extractor._classify_marker_type(marker) == kind


@pytest.mark.parametrize("marker,kind", [("A.1", "alpha"), ("1.1.1", "numeric"),
    ("II", "roman"), ("a)", "other"), ("•", "other")])
def test_marker_classification_patterns(docling_env, marker, kind):
    assert docling_env.extractor._classify_marker_type(marker) == kind


@pytest.mark.parametrize("heights,tolerance,expected", [
    ([], 0.15, []), ([20, 19, 10, 9, 20], 0.15, [[20, 19], [10, 9]]),
    ([20, 17], 0.15, [[20, 17]]), ([20, 16.9], 0.15, [[20], [16.9]]),
    ([10, 10, 9], 0, [[10], [9]]),
])
def test_font_clusters_and_inclusive_tolerance(docling_env, heights, tolerance, expected):
    assert docling_env.extractor._cluster_font_sizes(heights, tolerance) == expected


def heading_item(env, text, height, ref):
    return sdk_item(env.module.SectionHeaderItem, text=text, self_ref=ref,
                    prov=[SimpleNamespace(bbox=SimpleNamespace(t=height, b=0))])


def test_hierarchy_maps_external_paths_and_infers_font_levels(docling_env):
    env = docling_env
    descriptions = [("Introduction", 30, "h1"), ("1. Methods", 20, "h2"),
                    ("A. Setup", 10, "h3"), ("B. Run", 10, "h4"),
                    ("2. Results", 20, "h5"), ("Unexpected prose", 20, "h6"),
                    ("References", 20, "h7")]
    items = [heading_item(env, *description) for description in descriptions]
    document = SimpleNamespace(iterate_items=lambda: iter((item, 1) for item in items))
    paths = [["Introduction"], ["Introduction", "1. Methods"],
             ["Introduction", "1. Methods", "A. Setup"],
             ["Introduction", "1. Methods", "B. Run"], ["Introduction", "2. Results"],
             [], ["References"]]
    sdk_chunks = [SimpleNamespace(meta=SimpleNamespace(headings=path, doc_items=[item]))
                  for item, path in zip(items, paths, strict=True)]
    env.extractor._chunker = SimpleNamespace(chunk=lambda doc: sdk_chunks)
    refs, levels = env.extractor._build_hierarchy_map(document)
    assert refs == {"h1": ["Introduction"], "h2": ["Introduction", "1. Methods"],
                    "h3": ["Introduction", "1. Methods", "A. Setup"],
                    "h4": ["Introduction", "1. Methods", "B. Run"],
                    "h5": ["Introduction", "2. Results"], "h6": [], "h7": ["References"]}
    assert levels == {"Introduction": 1, "1. Methods": 2, "A. Setup": 3, "B. Run": 3,
                      "2. Results": 2, "Unexpected prose": -1, "References": 2}


def test_hierarchy_same_font_uses_marker_spans(docling_env):
    env = docling_env
    texts = ["I. Main", "A. Child", "B. Child", "II. Main", "C. Child", "D. Child"]
    items = [heading_item(env, text, 20, f"h{index}") for index, text in enumerate(texts)]
    document = SimpleNamespace(iterate_items=lambda: iter((item, 1) for item in items))
    env.extractor._chunker = SimpleNamespace(chunk=lambda doc: [])
    _, levels = env.extractor._build_hierarchy_map(document)
    assert levels == {"I. Main": 1, "A. Child": 2, "B. Child": 2,
                      "II. Main": 1, "C. Child": 2, "D. Child": 2}


def test_hierarchy_without_headings_is_empty(docling_env):
    document = SimpleNamespace(iterate_items=lambda: iter([(SimpleNamespace(text="Body text"), 0)]))
    assert docling_env.extractor._build_hierarchy_map(document) == ({}, {})


def test_hierarchy_sdk_failure_keeps_inferred_levels(docling_env, caplog):
    env = docling_env
    item = heading_item(env, "Introduction", 20, "heading")
    document = SimpleNamespace(iterate_items=lambda: iter([(item, 1)]))

    def failure(doc):
        raise RuntimeError("Synthetic hierarchy SDK failure")

    env.extractor._chunker = SimpleNamespace(chunk=failure)
    assert env.extractor._build_hierarchy_map(document) == ({}, {"Introduction": 1})
    assert "hierarchy paths will be empty" in caplog.text


@pytest.mark.parametrize("texts,valid", [([], True), (["Ordinary text."], True),
    ([""], True), (["Text /C0 garble"], False), (["Text /C1 garble"], False)])
def test_enriched_page_validity_and_page_selection(docling_env, texts, valid):
    requested = []

    def iterate_items(page_no):
        requested.append(page_no)
        return iter((SimpleNamespace(text=text), 0) for text in texts)

    document = SimpleNamespace(iterate_items=iterate_items)
    assert docling_env.extractor._is_enriched_page_valid(document, page_no=2) is valid
    assert requested == [2]


def missing_imports(monkeypatch, names):
    original = builtins.__import__
    failures = {}

    def guarded(name, *args, **kwargs):
        if name in names:
            error = ImportError(f"Deliberately missing {name}")
            failures[name] = error
            raise error
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    return failures


def test_missing_pymupdf4llm_has_actionable_message_and_cause(monkeypatch):
    from longparser.extractors.pymupdf_extractor import PyMuPDFExtractor

    failures = missing_imports(monkeypatch, {"pymupdf4llm"})
    with pytest.raises(ImportError, match=r"longparser\[pymupdf\]") as caught:
        PyMuPDFExtractor()
    assert caught.value.__cause__ is failures["pymupdf4llm"]
    assert "pymupdf4llm is not installed" in str(caught.value)


def test_both_pymupdf_aliases_missing_has_actionable_message_and_cause(monkeypatch):
    from longparser.extractors.pymupdf_extractor import _require_pymupdf_fitz

    failures = missing_imports(monkeypatch, {"pymupdf", "fitz"})
    with pytest.raises(ImportError, match=r"longparser\[pymupdf\]") as caught:
        _require_pymupdf_fitz()
    assert set(failures) == {"pymupdf", "fitz"}
    assert caught.value.__cause__ is failures["fitz"]


def test_missing_marker_has_actionable_message_and_cause(monkeypatch):
    from longparser.extractors.marker_extractor import MarkerExtractor

    failures = missing_imports(monkeypatch, {"marker"})
    with pytest.raises(ImportError, match=r"longparser\[marker\]") as caught:
        MarkerExtractor()
    assert caught.value.__cause__ is failures["marker"]
    assert "marker-pdf is not installed" in str(caught.value)


@pytest.fixture
def latex_env(monkeypatch):
    from types import ModuleType

    from longparser.extractors import latex_ocr

    monkeypatch.setattr(latex_ocr.LaTeXOCR, "_instances", {})
    monkeypatch.setattr(latex_ocr.MFDBackend, "_instance", None)
    monkeypatch.delenv("LONGPARSER_MFD_MODEL_DIR", raising=False)
    monkeypatch.delenv("LONGPARSER_UNIMERNET_MODEL_DIR", raising=False)
    monkeypatch.setenv("LONGPARSER_LATEX_OCR_THREADS", "2")
    threads = []
    torch = ModuleType("torch")
    torch.set_num_threads = threads.append
    monkeypatch.setitem(sys.modules, "torch", torch)
    return SimpleNamespace(module=latex_ocr, threads=threads)


def test_latex_singleton_is_lazy_and_separate_per_backend(latex_env):
    cls = latex_env.module.LaTeXOCR
    first = cls("pix2tex")
    assert cls("pix2tex") is first
    assert cls("unimernet") is not first
    assert latex_env.threads == []


def test_missing_pix2tex_is_unavailable_without_loading_or_inference(latex_env, monkeypatch, caplog):
    failures = missing_imports(monkeypatch, {"pix2tex.cli"})
    ocr = latex_env.module.LaTeXOCR("pix2tex")
    assert ocr.available is False
    assert ocr.available is False
    assert ocr.recognize(object()) is None
    assert set(failures) == {"pix2tex.cli"}
    assert latex_env.threads == [2]  # External setup attempted once, then cached.
    assert "pix2tex>=0.1.4" in caplog.text


def test_pix2tex_availability_and_thread_configuration_with_sdk_stub(latex_env, monkeypatch):
    from types import ModuleType

    model_creations = []
    inference_calls = []

    class Model:
        def __call__(self, image):
            inference_calls.append(image)
            raise AssertionError("No OCR inference may run in setup tests")

    def create_model():
        model_creations.append("external SDK constructor")
        return Model()

    package = ModuleType("pix2tex")
    package.__path__ = []
    sdk = ModuleType("pix2tex.cli")
    sdk.LatexOCR = create_model
    monkeypatch.setitem(sys.modules, "pix2tex", package)
    monkeypatch.setitem(sys.modules, "pix2tex.cli", sdk)
    # Disable the adapter's optional pre-warm before any inference can be called.
    missing_imports(monkeypatch, {"PIL"})
    monkeypatch.setenv("LONGPARSER_LATEX_OCR_THREADS", "3")
    ocr = latex_env.module.LaTeXOCR("pix2tex")
    assert model_creations == []
    assert ocr.available is True
    assert ocr.available is True
    assert model_creations == ["external SDK constructor"]
    assert latex_env.threads == [3]
    assert inference_calls == []


def test_unknown_latex_backend_is_unavailable(latex_env, caplog):
    ocr = latex_env.module.LaTeXOCR("unknown-backend")
    assert ocr.available is False
    assert ocr.recognize(object()) is None
    assert latex_env.threads == []
    assert "Unknown LaTeX OCR backend" in caplog.text


def test_missing_unimernet_is_unavailable(latex_env, monkeypatch, caplog):
    missing_imports(monkeypatch, {"unimernet.common.config"})
    assert latex_env.module.LaTeXOCR("unimernet").available is False
    assert "unimernet>=0.2.0" in caplog.text


def test_unimernet_missing_model_directory_never_calls_weight_loader(latex_env, monkeypatch, tmp_path):
    from types import ModuleType

    def forbidden(*args, **kwargs):
        raise AssertionError("Weight loading must not execute")

    for name, symbols in {"unimernet.common.config": {"Config": forbidden},
                          "unimernet.models": {"load_model": forbidden},
                          "unimernet.processors": {"load_processor": forbidden}}.items():
        module = ModuleType(name)
        for symbol, value in symbols.items():
            setattr(module, symbol, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("LONGPARSER_UNIMERNET_MODEL_DIR", str(tmp_path / "missing-model-dir"))
    assert latex_env.module.LaTeXOCR("unimernet").available is False
    assert latex_env.threads == [2]


@pytest.mark.parametrize("directory", ["unset", "missing", "empty"])
def test_mfd_without_local_weights_is_unavailable(latex_env, monkeypatch, tmp_path, directory):
    if directory == "missing":
        monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path / "missing"))
    elif directory == "empty":
        monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path))
    mfd = latex_env.module.MFDBackend.get()
    assert latex_env.module.MFDBackend.get() is mfd
    assert mfd.available is False
    assert mfd.detect(object()) == []


def test_mfd_local_path_missing_sdk_stays_unavailable(latex_env, monkeypatch, tmp_path):
    (tmp_path / "mfd-stub.onnx").write_bytes(b"Path-selection fixture, not model weights")
    monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path))
    failures = missing_imports(monkeypatch, {"pix2text.formula_detector"})
    assert latex_env.module.MFDBackend.get().available is False
    assert set(failures) == {"pix2text.formula_detector"}


def test_mfd_passes_explicit_local_model_path_and_cpu_to_sdk(latex_env, monkeypatch, tmp_path):
    from types import ModuleType

    path = tmp_path / "mfd-stub.onnx"
    path.write_bytes(b"Path-selection fixture, not model weights")
    monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path))
    calls = []
    sdk = ModuleType("pix2text.formula_detector")
    sdk.MathFormulaDetector = lambda **kwargs: (calls.append(kwargs) or object())
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    assert latex_env.module.MFDBackend.get().available is True
    assert calls == [{"model_path": path, "device": "cpu"}]
    # Do not invoke detect(): only local-path/configuration availability is tested.

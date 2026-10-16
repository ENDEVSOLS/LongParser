"""Equation validation and OCR orchestration with controlled external model SDKs."""

import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from longparser.extractors import latex_ocr as module


@pytest.mark.parametrize("text,valid", [
    ("", False), ("  ", False), ("x" * 2001, False), ("}{", False),
    ("{x", False), (r"\left(x", False), (r"\frac" * 6, False),
    (r"\frac{x}{y}", True), (r"\left(x\right)", True),
    (r"\alpha\beta\gamma\delta\epsilon\zeta", True),
])
def test_latex_validation(text, valid):
    assert module.validate_latex(text) is valid


@pytest.mark.parametrize("raw", [" $$ x+y $$ ", "$x+y$", r"\[x+y\]", r"\(x+y\)", "x+y"])
def test_delimiters(raw):
    assert module.strip_delimiters(raw) == "x+y"


def install(monkeypatch, name, **symbols):
    sdk = ModuleType(name)
    for key, value in symbols.items():
        setattr(sdk, key, value)
    monkeypatch.setitem(sys.modules, name, sdk)
    return sdk


@pytest.fixture
def models(monkeypatch):
    monkeypatch.setattr(module.LaTeXOCR, "_instances", {})
    monkeypatch.setattr(module.MFDBackend, "_instance", None)
    monkeypatch.delenv("LONGPARSER_MFD_MODEL_DIR", raising=False)
    torch = install(monkeypatch, "torch", set_num_threads=Mock())
    return torch


def test_pix2tex_loading_warmup_and_inference(models, monkeypatch):
    image = Image.new("RGB", (70, 70))
    model = Mock(return_value="x+y")
    factory = Mock(return_value=model)
    install(monkeypatch, "pix2tex.cli", LatexOCR=factory)
    backend = module.Pix2TexBackend()
    assert backend.recognize(image) is None
    assert backend.load()
    models.set_num_threads.assert_called_once_with(2)
    assert model.call_args.args[0].size == (64, 64)
    assert backend.recognize(image) == "x+y"
    model.return_value = 3
    assert backend.recognize(image) == "3"
    model.side_effect = RuntimeError("inference failed")
    assert backend.recognize(image) is None
    assert backend.load()  # warmup failure does not prevent availability
    factory.side_effect = RuntimeError("broken weights")
    assert not module.Pix2TexBackend().load()
    monkeypatch.setitem(sys.modules, "pix2tex.cli", None)
    assert not module.Pix2TexBackend().load()


def test_unimernet_configuration_and_inference(models, monkeypatch, tmp_path):
    cfg = Mock(side_effect=lambda value: value)
    model = Mock()
    processor = Mock(return_value="tensor")
    install(monkeypatch, "unimernet.common.config", Config=cfg)
    factory = Mock(return_value=model)
    install(monkeypatch, "unimernet.models", load_model=factory)
    install(monkeypatch, "unimernet.processors", load_processor=Mock(return_value=processor))
    backend = module.UniMERNetBackend()
    assert backend.recognize("image") is None
    monkeypatch.delenv("LONGPARSER_UNIMERNET_MODEL_DIR", raising=False)
    assert not backend.load()
    monkeypatch.setenv("LONGPARSER_UNIMERNET_MODEL_DIR", str(tmp_path))
    assert backend.load()
    cfg.assert_called_with({"model": {"arch": "unimernet_tiny", "model_path": str(tmp_path)}})
    model.generate.return_value = ["x+y"]
    assert backend.recognize("image") == "x+y"
    model.generate.assert_called_with("tensor")
    model.generate.return_value = []
    assert backend.recognize("image") is None
    processor.side_effect = ValueError("bad image")
    assert backend.recognize("image") is None
    factory.side_effect = RuntimeError("bad weights")
    assert not backend.load()
    monkeypatch.setitem(sys.modules, "unimernet.models", None)
    assert not backend.load()


def test_singleton_lazy_validation_and_failure(models, monkeypatch):
    sdk_model = Mock(return_value="$$x+y$$")
    install(monkeypatch, "pix2tex.cli", LatexOCR=Mock(return_value=sdk_model))
    ocr = module.LaTeXOCR()
    assert ocr is module.LaTeXOCR()
    assert not ocr._initialized
    assert ocr.available
    assert ocr.recognize("image") == "x+y"
    sdk_model.return_value = "{"
    assert ocr.recognize("image") is None
    sdk_model.return_value = None  # backend coerces non-string output before validation
    assert ocr.recognize("image") == "None"
    sdk_model.side_effect = RuntimeError("failed")
    assert ocr.recognize("image") is None
    unknown = module.LaTeXOCR("unknown")
    assert not unknown.available
    assert unknown.recognize("image") is None
    monkeypatch.setitem(sys.modules, "unimernet.common.config", None)
    assert not module.LaTeXOCR("unimernet").available


def test_detector_configuration_and_failure(models, monkeypatch, tmp_path):
    assert not module.MFDBackend.get().available
    assert module.MFDBackend.get() is module.MFDBackend.get()
    monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path / "missing"))
    assert not module.MFDBackend._load().available
    monkeypatch.setenv("LONGPARSER_MFD_MODEL_DIR", str(tmp_path))
    assert not module.MFDBackend._load().available
    checkpoint = tmp_path / "mfd-1.5.onnx"
    checkpoint.write_bytes(b"controlled SDK fixture")
    factory = Mock(return_value=Mock())
    install(monkeypatch, "pix2text.formula_detector", MathFormulaDetector=factory)
    backend = module.MFDBackend._load()
    assert backend.available
    factory.assert_called_once_with(model_path=checkpoint, device="cpu")
    factory.side_effect = RuntimeError("broken weights")
    assert not module.MFDBackend._load().available
    monkeypatch.setitem(sys.modules, "pix2text.formula_detector", None)
    assert not module.MFDBackend._load().available


def test_detector_filters_sorts_caps_and_recovers(models):
    backend = module.MFDBackend()
    assert backend.detect("image") == []
    backend.available = True
    backend._mfd = Mock()
    def box(size, kind, score):
        return {"box": np.array([[0, 0], [size, 0], [size, size], [0, size]]),
                "type": kind, "score": score}
    backend._mfd.detect.return_value = [box(10, "isolated", 1), box(100, "embedding", .9),
                                       box(60, "isolated", .7), box(60, "isolated", .8)]
    result = backend.detect("image", threshold=.3, max_boxes=2)
    assert [(b["type"], b["score"]) for b in result] == [("isolated", .8), ("isolated", .7)]
    assert result[0]["x1"] == result[0]["y1"] == 60
    backend._mfd.detect.assert_called_with("image", threshold=.3)
    backend._mfd.detect.side_effect = RuntimeError("inference failed")
    assert backend.detect("image") == []

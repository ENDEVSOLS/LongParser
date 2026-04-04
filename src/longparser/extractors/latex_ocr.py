"""LaTeX OCR module — surgical equation-image → LaTeX conversion.

Production-hardened with:
- Pluggable backends (pix2tex / UniMERNet)
- Thread-safe singleton with lazy loading
- LaTeX validation (braces, left/right parity, length, repeated tokens)
- Forced CPU inference (no GPU surprise)
- Graceful degradation when weights unavailable
"""

import os
import re
import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LaTeX validation utilities
# ---------------------------------------------------------------------------

def validate_latex(s: str) -> bool:
    """Check if a LaTeX string is well-formed enough to use."""
    if not s or not s.strip():
        return False
    s = s.strip()

    # Max length
    if len(s) > 2000:
        logger.debug(f"LaTeX too long ({len(s)} chars), rejecting")
        return False

    # Balanced braces
    depth = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if depth < 0:
            return False
    if depth != 0:
        logger.debug("LaTeX has unbalanced braces")
        return False

    # \left / \right parity
    lefts = len(re.findall(r'\\left[^a-zA-Z]', s))
    rights = len(re.findall(r'\\right[^a-zA-Z]', s))
    if lefts != rights:
        logger.debug(f"LaTeX \\left/{lefts} != \\right/{rights}")
        return False

    # Repeated token check (e.g., "\frac\frac\frac" junk)
    tokens = re.findall(r'\\[a-zA-Z]+', s)
    if len(tokens) > 5:
        from collections import Counter
        counts = Counter(tokens)
        most_common_count = counts.most_common(1)[0][1]
        if most_common_count > len(tokens) * 0.6:
            logger.debug("LaTeX has repeated junk tokens")
            return False

    return True


def strip_delimiters(s: str) -> str:
    """Remove existing LaTeX delimiters from raw OCR output."""
    s = s.strip()
    # Remove wrapping $$ or $
    if s.startswith("$$") and s.endswith("$$"):
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove \[ \] or \( \)
    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()
    elif s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class LaTeXOCRBackend(ABC):
    """Abstract base for LaTeX OCR backends."""

    @abstractmethod
    def load(self) -> bool:
        """Load model. Returns True if successful."""
        ...

    @abstractmethod
    def recognize(self, image) -> Optional[str]:
        """Run inference on a PIL Image. Returns raw LaTeX or None."""
        ...


# ---------------------------------------------------------------------------
# pix2tex backend (CC BY-NC-SA weights — non-commercial only)
# ---------------------------------------------------------------------------

class Pix2TexBackend(LaTeXOCRBackend):
    """pix2tex / LaTeX-OCR backend (~30MB, ~20ms/eq on CPU)."""

    def __init__(self):
        self._model = None

    def load(self) -> bool:
        try:
            import torch
            torch.set_num_threads(int(os.getenv("LONGPARSER_LATEX_OCR_THREADS", "2")))

            from pix2tex.cli import LatexOCR
            self._model = LatexOCR()

            try:
                # Pre-warm with dummy inference (safe to fail)
                from PIL import Image
                dummy = Image.new("RGB", (64, 64), color="white")
                self._model(dummy)
            except Exception as e:
                logger.debug(f"Pix2Tex pre-warm dummy inference skipped: {e}")

            logger.info("Pix2TexBackend loaded and pre-warmed")
            return True
        except ImportError:
            logger.warning("pix2tex not installed. Install: pip install 'pix2tex>=0.1.4'")
            return False
        except Exception as e:
            logger.warning(f"Pix2TexBackend failed to load: {e}")
            return False

    def recognize(self, image) -> Optional[str]:
        if self._model is None:
            return None
        try:
            result = self._model(image)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.debug(f"pix2tex inference failed: {e}")
            return None


# ---------------------------------------------------------------------------
# UniMERNet backend (Apache 2.0 — commercial safe)
# ---------------------------------------------------------------------------

class UniMERNetBackend(LaTeXOCRBackend):
    """UniMERNet-tiny backend (~441MB, Apache 2.0)."""

    def __init__(self):
        self._model = None

    def load(self) -> bool:
        try:
            import torch
            torch.set_num_threads(int(os.getenv("LONGPARSER_LATEX_OCR_THREADS", "2")))

            from unimernet.common.config import Config
            from unimernet.processors import load_processor
            from unimernet.models import load_model

            model_dir = os.getenv("LONGPARSER_UNIMERNET_MODEL_DIR", "")
            if not model_dir or not os.path.isdir(model_dir):
                logger.warning(
                    "UniMERNet model dir not found. "
                    "Set LONGPARSER_UNIMERNET_MODEL_DIR to the checkpoint directory."
                )
                return False

            cfg = Config({"model": {"arch": "unimernet_tiny", "model_path": model_dir}})
            self._model = load_model(cfg)
            self._processor = load_processor(cfg)

            logger.info("UniMERNetBackend loaded")
            return True
        except ImportError:
            logger.warning("unimernet not installed. Install: pip install 'unimernet>=0.2.0'")
            return False
        except Exception as e:
            logger.warning(f"UniMERNetBackend failed to load: {e}")
            return False

    def recognize(self, image) -> Optional[str]:
        if self._model is None:
            return None
        try:
            inputs = self._processor(image)
            result = self._model.generate(inputs)
            return result[0] if result else None
        except Exception as e:
            logger.debug(f"UniMERNet inference failed: {e}")
            return None


# ---------------------------------------------------------------------------
# Main singleton
# ---------------------------------------------------------------------------

class LaTeXOCR:
    """Thread-safe singleton LaTeX OCR with pluggable backend.

    Usage:
        ocr = LaTeXOCR(backend="pix2tex")
        if ocr.available:
            latex = ocr.recognize(pil_image)
    """

    _instances: dict = {}
    _lock = threading.Lock()

    def __new__(cls, backend: str = "pix2tex"):
        with cls._lock:
            if backend not in cls._instances:
                instance = super().__new__(cls)
                instance._backend_name = backend
                instance._backend: Optional[LaTeXOCRBackend] = None
                instance._available = False
                instance._initialized = False
                cls._instances[backend] = instance
            return cls._instances[backend]

    def _ensure_loaded(self):
        """Lazy-load backend on first use."""
        if self._initialized:
            return
        self._initialized = True

        if self._backend_name == "pix2tex":
            self._backend = Pix2TexBackend()
        elif self._backend_name == "unimernet":
            self._backend = UniMERNetBackend()
        else:
            logger.error(f"Unknown LaTeX OCR backend: {self._backend_name}")
            return

        self._available = self._backend.load()
        if not self._available:
            logger.warning(
                f"LaTeX OCR backend '{self._backend_name}' not available. "
                "Formula OCR will be skipped."
            )

    @property
    def available(self) -> bool:
        """Whether the backend is loaded and ready."""
        self._ensure_loaded()
        return self._available

    def recognize(self, image) -> Optional[str]:
        """Recognize a formula image → validated LaTeX string.

        Returns None if backend unavailable, inference fails, or validation fails.
        """
        if not self.available:
            return None

        with self._lock:
            raw = self._backend.recognize(image)

        if raw is None:
            return None

        # Strip existing delimiters
        latex = strip_delimiters(raw)

        # Validate
        if not validate_latex(latex):
            logger.debug(f"LaTeX validation failed for: {latex[:100]}...")
            return None

        return latex


# ---------------------------------------------------------------------------
# MFD: Math Formula Detector (page-level, pix2text YOLO-based)
# ---------------------------------------------------------------------------

class MFDBackend:
    """Thread-safe singleton for page-level math formula detection.

    Uses pix2text's MathFormulaDetector (YOLO-nano, MIT).
    Requires LONGPARSER_MFD_MODEL_DIR pointing to a directory containing
    a *mfd*.onnx file. If missing, available=False and no network calls
    are ever made.

    Usage:
        mfd = MFDBackend.get()
        if mfd.available:
            boxes = mfd.detect(page_pil_image)
    """

    _instance: Optional["MFDBackend"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._mfd = None
        self.available: bool = False

    @classmethod
    def get(cls) -> "MFDBackend":
        """Return the singleton, initialising it on first call."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls._load()
        return cls._instance

    @classmethod
    def _load(cls) -> "MFDBackend":
        inst = cls()
        model_dir = os.getenv("LONGPARSER_MFD_MODEL_DIR", "").strip()
        if not model_dir:
            logger.debug("MFD disabled: LONGPARSER_MFD_MODEL_DIR not set")
            return inst

        from pathlib import Path as _Path
        model_dir_path = _Path(model_dir)
        if not model_dir_path.exists():
            logger.warning(f"MFD model dir not found: {model_dir}. MFD disabled.")
            return inst

        # Scan for *mfd*.onnx — same glob pattern as pix2text's own find_files()
        candidates = sorted(model_dir_path.rglob("*mfd*.onnx"))
        if not candidates:
            logger.warning(
                f"No *mfd*.onnx found in {model_dir}. MFD disabled. "
                "Download the mfd-1.5.onnx from the pix2text model hub and place it here."
            )
            return inst

        model_path = candidates[0]
        try:
            from pix2text.formula_detector import MathFormulaDetector
            # Pass model_path directly → prepare_model_files() is never called
            inst._mfd = MathFormulaDetector(
                model_path=model_path,
                device="cpu",
            )
            inst.available = True
            logger.info(f"MFDBackend loaded from: {model_path}")
        except ImportError:
            logger.warning(
                "pix2text not installed. Install: pip install 'pix2text>=1.1.1,<1.2'. MFD disabled."
            )
        except Exception as e:
            logger.warning(f"MFDBackend failed to load: {e}. MFD disabled.")

        return inst

    def detect(
        self,
        page_img,
        threshold: float = 0.45,
        max_boxes: int = 10,
        min_area_px: int = 2048,
    ) -> list[dict]:
        """Detect math formula regions in a PIL page image.

        Args:
            page_img: PIL.Image of a document page.
            threshold: Detection confidence threshold.
            max_boxes: Maximum boxes to return (after sorting).
            min_area_px: Minimum pixel area to keep a detection.

        Returns:
            List of dicts: {x0, y0, x1, y1, type:'isolated'|'embedding', score}
            Sorted: isolated first → larger area first → higher score first.
            Returns [] on error or unavailability.
        """
        if not self.available:
            return []
        try:
            import numpy as np
            raw = self._mfd.detect(page_img, threshold=threshold)
            boxes = []
            for r in raw:
                pts = r["box"]  # np.ndarray shape (4, 2): [[x,y], ...]
                x0 = int(np.min(pts[:, 0]))
                y0 = int(np.min(pts[:, 1]))
                x1 = int(np.max(pts[:, 0]))
                y1 = int(np.max(pts[:, 1]))
                area = (x1 - x0) * (y1 - y0)
                if area < min_area_px:
                    continue
                boxes.append({
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "type": r.get("type", "isolated"),
                    "score": float(r.get("score", 1.0)),
                })
            # Priority: isolated > larger area > higher confidence
            boxes.sort(key=lambda b: (
                0 if b["type"] == "isolated" else 1,
                -((b["x1"] - b["x0"]) * (b["y1"] - b["y0"])),
                -b["score"],
            ))
            return boxes[:max_boxes]
        except Exception as e:
            logger.warning(f"MFD detect error: {e}")
            return []

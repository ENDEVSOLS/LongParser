"""Language detection for document text samples.

Uses ``fast-langdetect`` (Apache-2.0, Facebook FastText model) to detect
the primary language of a text sample and map it to Tesseract language codes.

This module is designed for zero-failure operation:
- Falls back to English if ``fast-langdetect`` is not installed
- Falls back to English if detection confidence is too low
- Falls back to English on any unexpected error
- Never raises exceptions that would break the pipeline

Usage::

    from longparser.utils.lang_detect import detect_language, get_tesseract_langs

    lang, confidence = detect_language("هذا نص عربي")  # ("ar", 0.99)
    tess_codes = get_tesseract_langs("ar")             # ["ara"]
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: ISO 639-1 code (fast-langdetect) → Tesseract language code(s)
# ---------------------------------------------------------------------------
_LANG_TO_TESSERACT: dict[str, list[str]] = {
    "af": ["afr"],   "am": ["amh"],   "ar": ["ara"],   "az": ["aze"],
    "be": ["bel"],   "bg": ["bul"],   "bn": ["ben"],   "bs": ["bos"],
    "ca": ["cat"],   "cs": ["ces"],   "cy": ["cym"],   "da": ["dan"],
    "de": ["deu"],   "el": ["ell"],   "en": ["eng"],   "es": ["spa"],
    "et": ["est"],   "eu": ["eus"],   "fa": ["fas"],   "fi": ["fin"],
    "fr": ["fra"],   "ga": ["gle"],   "gl": ["glg"],   "gu": ["guj"],
    "ha": ["hau"],   "he": ["heb"],   "hi": ["hin"],   "hr": ["hrv"],
    "hu": ["hun"],   "hy": ["hye"],   "id": ["ind"],   "is": ["isl"],
    "it": ["ita"],   "ja": ["jpn"],   "jv": ["jav"],   "ka": ["kat"],
    "kk": ["kaz"],   "km": ["khm"],   "kn": ["kan"],   "ko": ["kor"],
    "la": ["lat"],   "lt": ["lit"],   "lv": ["lav"],   "mk": ["mkd"],
    "ml": ["mal"],   "mn": ["mon"],   "mr": ["mar"],   "ms": ["msa"],
    "my": ["mya"],   "ne": ["nep"],   "nl": ["nld"],   "no": ["nor"],
    "pa": ["pan"],   "pl": ["pol"],   "pt": ["por"],   "ro": ["ron"],
    "ru": ["rus"],   "si": ["sin"],   "sk": ["slk"],   "sl": ["slv"],
    "sq": ["sqi"],   "sr": ["srp"],   "sv": ["swe"],   "sw": ["swa"],
    "ta": ["tam"],   "te": ["tel"],   "th": ["tha"],   "tl": ["tgl"],
    "tr": ["tur"],   "uk": ["ukr"],   "ur": ["urd"],   "uz": ["uzb"],
    "vi": ["vie"],   "yo": ["yor"],
    # Chinese variants
    "zh": ["chi_sim", "chi_tra"],
}


def detect_language(
    text: str,
    min_confidence: float = 0.5,
) -> tuple[str, float]:
    """Detect the primary language of a text sample.

    Parameters
    ----------
    text:
        Text sample to analyze. At least 20 characters recommended.
    min_confidence:
        Minimum confidence threshold. Below this, falls back to ``"en"``.

    Returns
    -------
    tuple[str, float]
        ``(language_code, confidence)`` — e.g. ``("ar", 0.99)``.
        Falls back to ``("en", 0.0)`` on any failure.
    """
    if not text or len(text.strip()) < 20:
        logger.debug("Text too short for language detection, defaulting to English")
        return "en", 0.0

    try:
        from fast_langdetect import detect
        result = detect(text)
        lang = result.get("lang", "en")
        score = result.get("score", 0.0)

        if score < min_confidence:
            logger.info(
                "Language detection low confidence (%.2f for '%s'), "
                "defaulting to English", score, lang
            )
            return "en", score

        logger.info("Detected language: %s (confidence: %.2f)", lang, score)
        return lang, score

    except ImportError:
        logger.warning(
            "fast-langdetect is not installed. Language detection disabled. "
            "Install with: pip install fast-langdetect"
        )
        return "en", 0.0
    except Exception as e:
        logger.warning("Language detection failed: %s — defaulting to English", e)
        return "en", 0.0


def get_tesseract_langs(lang_code: str) -> list[str]:
    """Map a detected language code to Tesseract language code(s).

    Parameters
    ----------
    lang_code:
        ISO 639-1 language code (e.g. ``"ar"``, ``"en"``).

    Returns
    -------
    list[str]
        Tesseract language codes (e.g. ``["ara"]``, ``["eng"]``).
    """
    return _LANG_TO_TESSERACT.get(lang_code, ["eng"])


def extract_sample_text(file_path, max_chars: int = 2000) -> str:
    """Extract a sample of text from a document for language detection.

    Uses a lightweight approach: reads first few KB of the file and
    extracts printable text. For PDFs, attempts to use PyMuPDF if
    available, otherwise falls back to reading raw bytes.

    Parameters
    ----------
    file_path:
        Path to the document file.
    max_chars:
        Maximum characters to extract.

    Returns
    -------
    str
        Extracted text sample, or empty string if extraction fails.
    """
    from pathlib import Path
    file_path = Path(file_path)

    if not file_path.exists():
        return ""

    ext = file_path.suffix.lower()

    # For PDFs: try lightweight text extraction
    if ext == ".pdf":
        return _extract_pdf_sample(file_path, max_chars)

    # For text-like files: read directly
    if ext in (".csv", ".txt", ".md"):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars)
        except Exception:
            return ""

    # For other formats: return empty (language detection will use
    # text extracted by Docling later)
    return ""


def _extract_pdf_sample(file_path, max_chars: int) -> str:
    """Extract text sample from a PDF using the lightest method available."""
    # Try pdfplumber (lightweight, often available)
    try:
        import pdfplumber
        with pdfplumber.open(str(file_path)) as pdf:
            text = ""
            for page in pdf.pages[:3]:  # First 3 pages
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                if len(text) >= max_chars:
                    break
            return text[:max_chars]
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: read raw bytes and extract printable chars
    try:
        with open(file_path, "rb") as f:
            raw = f.read(max_chars * 4)  # Read more bytes since not all are text
        # Extract ASCII/Unicode text from raw bytes
        text = raw.decode("utf-8", errors="ignore")
        # Filter to printable characters
        printable = "".join(c for c in text if c.isprintable() or c in "\n\t ")
        return printable[:max_chars]
    except Exception:
        return ""

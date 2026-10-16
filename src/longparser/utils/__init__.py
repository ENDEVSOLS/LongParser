"""Utility modules for LongParser."""

from .lang_detect import detect_language, get_tesseract_langs
from .ocr_router import get_ocr_strategy, is_page_scanned, score_page_complexity
from .rtl_detector import detect_rtl_language

__all__ = [
    "detect_rtl_language",
    "detect_language",
    "get_tesseract_langs",
    "is_page_scanned",
    "score_page_complexity",
    "get_ocr_strategy",
]

"""Smart OCR routing for scanned PDFs.

Routes pages to the best OCR strategy based on content complexity:

- **standard** — Tesseract with default settings (fast, CPU-native)
- **math** — Tesseract for text + pix2tex for equations
- **full_ocr** — Tesseract with ``force_full_page_ocr=True``

All strategies are CPU-friendly. No GPU-dependent engines (Surya, Marker)
are used in the routing — those are available as separate optional backends.

Usage::

    from longparser.utils.ocr_router import (
        is_page_scanned, score_page_complexity, get_ocr_strategy,
    )

    if is_page_scanned(page_text):
        score = score_page_complexity(page_text, num_blocks=15, has_tables=True)
        strategy = get_ocr_strategy(score)
        # strategy = "full_ocr" for score >= 5
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pattern to detect math symbols and simple equations in text.
# Matches Unicode math symbols and simple algebraic patterns like "x = 5".
_MATH_RE = re.compile(
    r'[\u2211\u220F\u222B\u221A\u00B1\u2264\u2265\u2248\u2260\u03B1-\u03C9\u03A3]'
    r'|[a-z]\s*=\s*[a-z0-9]',
    re.IGNORECASE,
)


def is_page_scanned(page_text: str, min_chars: int = 30) -> bool:
    """Check if a page is likely scanned (no usable text layer).

    Parameters
    ----------
    page_text:
        Extracted text from the page.
    min_chars:
        Minimum character count to consider the page as having a text layer.

    Returns
    -------
    bool
        ``True`` if the page has fewer than ``min_chars`` printable characters
        (indicating it's likely a scanned image with no embedded text).
    """
    clean = page_text.strip()
    return len(clean) < min_chars


def has_math_content(text: str) -> bool:
    """Check if text contains mathematical symbols or equation patterns.

    Parameters
    ----------
    text:
        Text to check for math content.

    Returns
    -------
    bool
        ``True`` if math symbols or equation patterns are found.
    """
    return bool(_MATH_RE.search(text))


def score_page_complexity(
    page_text: str,
    num_blocks: int = 0,
    has_tables: bool = False,
) -> int:
    """Score page complexity on a scale of 0-10.

    Used to decide which OCR strategy to apply:

    - **0-2** → ``"standard"`` — Simple page, Tesseract is enough
    - **3-4** → ``"math"`` — Has equations, add pix2tex
    - **5+** → ``"full_ocr"`` — Complex layout, use full-page OCR

    Parameters
    ----------
    page_text:
        Extracted text from the page.
    num_blocks:
        Number of content blocks on the page.
    has_tables:
        Whether the page contains tables.

    Returns
    -------
    int
        Complexity score from 0 to 10.
    """
    score = 0

    # Tables add significant complexity
    if has_tables:
        score += 3

    # Math content needs pix2tex
    if has_math_content(page_text):
        score += 2

    # Many blocks suggest a dense/complex layout
    if num_blocks > 20:
        score += 2
    elif num_blocks > 10:
        score += 1

    # Very short text on a page with blocks = likely OCR issues
    if page_text and len(page_text.strip()) < 100 and num_blocks > 5:
        score += 1

    return min(score, 10)


def get_ocr_strategy(complexity_score: int) -> str:
    """Pick OCR strategy based on page complexity score.

    Parameters
    ----------
    complexity_score:
        Score from :func:`score_page_complexity` (0-10).

    Returns
    -------
    str
        One of:

        - ``"standard"`` — Tesseract with default settings
        - ``"math"`` — Tesseract + pix2tex for equations
        - ``"full_ocr"`` — Tesseract with ``force_full_page_ocr=True``
    """
    if complexity_score <= 2:
        return "standard"
    elif complexity_score <= 4:
        return "math"
    else:
        return "full_ocr"

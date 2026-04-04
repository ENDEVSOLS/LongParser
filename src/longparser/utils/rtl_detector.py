"""RTL (Right-to-Left) language detection utility."""

import re
from typing import Optional


# Unicode ranges for RTL scripts
RTL_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    (0x0590, 0x05FF),  # Hebrew
    (0xFB00, 0xFB4F),  # Hebrew Presentation Forms
    (0x0700, 0x074F),  # Syriac
    (0x0780, 0x07BF),  # Thaana (Maldivian)
    (0x0840, 0x085F),  # Mandaic
]

# Compile regex pattern for RTL detection
RTL_PATTERN = re.compile(
    '[' + ''.join(
        f'\\u{start:04x}-\\u{end:04x}'
        for start, end in RTL_RANGES
    ) + ']'
)


def detect_rtl_language(text: str, threshold: float = 0.1) -> bool:
    """
    Detect if text contains significant RTL content.
    
    Args:
        text: Text to analyze
        threshold: Minimum ratio of RTL characters to consider text as RTL
        
    Returns:
        True if text is predominantly RTL
    """
    if not text:
        return False
    
    # Count RTL characters
    rtl_chars = len(RTL_PATTERN.findall(text))
    
    # Count total alphabetic characters (excluding spaces, numbers, punctuation)
    alpha_chars = sum(1 for c in text if c.isalpha())
    
    if alpha_chars == 0:
        return False
    
    rtl_ratio = rtl_chars / alpha_chars
    return rtl_ratio >= threshold


def detect_rtl_script(text: str) -> Optional[str]:
    """
    Detect the specific RTL script in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Script name ('arabic', 'hebrew', 'urdu', etc.) or None
    """
    if not text:
        return None
    
    # Arabic/Urdu detection (same script, different languages)
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]')
    arabic_count = len(arabic_pattern.findall(text))
    
    # Hebrew detection
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    hebrew_count = len(hebrew_pattern.findall(text))
    
    if arabic_count > hebrew_count and arabic_count > 0:
        # Check for Urdu-specific characters (some unique to Urdu)
        urdu_specific = re.compile(r'[\u0679\u067E\u0686\u0688\u0691\u0698\u06A9\u06AF\u06BA\u06BE\u06C1\u06C3\u06CC\u06D2]')
        if urdu_specific.search(text):
            return 'urdu'
        return 'arabic'
    
    if hebrew_count > 0:
        return 'hebrew'
    
    return None


def get_rtl_languages() -> list[str]:
    """Get list of supported RTL language codes."""
    return ['ar', 'he', 'ur', 'fa', 'ps', 'sd', 'yi', 'dv']

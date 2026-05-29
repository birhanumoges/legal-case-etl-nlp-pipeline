"""
utils/text_utils.py
--------------------
Shared text helpers reused across extractors, preprocessing, and modeling.
"""

import re
from typing import List, Optional


def truncate(text: str, max_chars: int = 10_000) -> str:
    """Truncate text to max_chars, breaking on a word boundary."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_page_markers(text: str) -> str:
    """Remove * page-number markers common in legal OCR output."""
    return re.sub(r"\*\d+", "", text)


def extract_dollar_amount(text: str) -> Optional[str]:
    """
    Return the first dollar amount found in text, e.g. '$1,200.00'.
    Returns None if no amount found.
    """
    m = re.search(r"\$([\d,]+(?:\.\d{2})?)", text)
    return m.group(0) if m else None


def extract_year(text: str) -> Optional[str]:
    """Return the first 4-digit year (1700-2099) found in text."""
    m = re.search(r"\b(1[7-9]\d{2}|20\d{2})\b", text)
    return m.group(0) if m else None


def count_words(text: str) -> int:
    """Return approximate word count."""
    return len(text.split()) if text else 0


def is_empty(text) -> bool:
    """Return True if text is None, empty, or whitespace-only."""
    return not text or not str(text).strip()


def clean_case_name(name: str) -> str:
    """
    Normalise a case name string.
    e.g. 'smith_v_jones_0001' → 'smith_v_jones_0001'  (kept as-is for IDs)
    """
    return str(name).strip() if name else ""
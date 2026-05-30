"""
extractors/text_extractor.py
-----------------------------
Reads an HTML file, strips boilerplate tags (script / style / nav / footer /
header) with BeautifulSoup, then normalises whitespace.

Returns (cleaned_text, original_char_count).
"""

import re
import logging
from pathlib import Path
from typing import Tuple

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags whose content we discard entirely
_STRIP_TAGS = ["script", "style", "nav", "footer", "header"]


def extract_text_from_html(html_file: Path) -> Tuple[str, int]:
    """
    Parse *html_file* and return ``(cleaned_text, original_length)``.

    Falls back to raw file content if BeautifulSoup fails.
    Returns ``("", 0)`` on any read error.
    """
    try:
        raw = html_file.read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Cannot read %s: %s", html_file, exc)
        return "", 0

    try:
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:
        text = raw

    original_length = len(text)
    text = _clean(text)
    return text, original_length


# ── Internal helpers ──────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"\*\d+", "", text)          # remove page markers like *12
    text = re.sub(r"\s+", " ", text)            # collapse whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)     # normalise blank lines
    return text.strip()
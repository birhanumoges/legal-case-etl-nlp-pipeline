"""
extractors/citation_extractor.py
----------------------------------
Combines regex-based citation extraction from raw case text with citations
already parsed from the JSON sidecar.

Returns a sorted, deduplicated list of citation strings.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Regex patterns for common legal citation formats
_CITATION_PATTERNS = [
    # Volume Reporter Page  e.g. "12 Ind. 345"
    r"\d+\s+[A-Z][a-z]+\.?\s+\d+",
    # Volume Two-word Reporter Page
    r"\d+\s+[A-Z][a-z]+\.?\s+[A-Z]?\.?\s+\d+",
    # U.S. Reports
    r"\d+\s+U\.S\.\s+\d+",
    # Blackford
    r"\d+\s+Blackf\.\s+\d+",
    # Binney
    r"\d+\s+Binn\.\s+\d+",
    # Named case citation  X v. Y, Vol Rep Page
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,"
    r"\s+\d+\s+[A-Za-z\.]+\s+\d+",
]

_COMPILED = [re.compile(p) for p in _CITATION_PATTERNS]


def extract_citations(text: str, json_metadata: Dict) -> List[str]:
    """
    Return a sorted, deduplicated list of legal citation strings.

    Sources
    -------
    1. Regex scan of *text*.
    2. ``citations`` list already parsed from *json_metadata*.
    """
    found: set = set()

    for pattern in _COMPILED:
        for m in pattern.finditer(text):
            found.add(m.group(0).strip())

    # Add pre-parsed citations from JSON
    if json_metadata:
        for cite in json_metadata.get("citations", []):
            if cite:
                found.add(str(cite).strip())

    return sorted(found)
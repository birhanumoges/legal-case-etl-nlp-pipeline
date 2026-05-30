"""
extractors/metadata_extractor.py
----------------------------------
Reads a CAP-style JSON sidecar file and returns a normalised metadata dict:

    {
        "case_name":     str,
        "decision_date": str,
        "docket_number": str,
        "first_page":    str,
        "last_page":     str,
        "court_name":    str,
        "citations":     [str, ...],
    }
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_metadata_from_json(json_file: Path) -> Dict:
    """
    Parse the JSON sidecar and return a normalised metadata dict.
    Returns an empty dict on any error.
    """
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Cannot parse %s: %s", json_file, exc)
        return {}

    meta: Dict = {
        "case_name":     data.get("name", ""),
        "decision_date": data.get("decision_date", ""),
        "docket_number": data.get("docket_number", ""),
        "first_page":    data.get("first_page", ""),
        "last_page":     data.get("last_page", ""),
        "court_name":    _extract_court_name(data),
        "citations":     _extract_citations(data),
    }
    return meta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_court_name(data: Dict) -> str:
    court = data.get("court")
    if isinstance(court, dict):
        return court.get("name", "")
    if court:
        return str(court)
    return ""


def _extract_citations(data: Dict) -> List[str]:
    """Collect citation strings from 'citations' and 'cites_to' fields."""
    cites: List[str] = []

    for cite in data.get("citations", []):
        if isinstance(cite, dict) and "cite" in cite:
            cites.append(cite["cite"])

    for cite in data.get("cites_to", []):
        if isinstance(cite, dict) and "cite" in cite:
            cites.append(cite["cite"])
        elif isinstance(cite, str):
            cites.append(cite)

    return cites


def extract_year(json_metadata: Dict, text: str) -> str:
    """
    Pull a 4-digit year from the decision_date field first, then fall back
    to scanning the raw text.
    """
    import re
    if json_metadata:
        m = re.search(r"(\d{4})", str(json_metadata.get("decision_date", "")))
        if m:
            return m.group(1)
    m = re.search(r"\b(18|19|20)\d{2}\b", text)
    return m.group(0) if m else "Unknown"


def extract_court(json_metadata: Dict, text: str) -> str:
    """Return court name from metadata or text heuristic."""
    import re
    if json_metadata and json_metadata.get("court_name"):
        return json_metadata["court_name"]
    m = re.search(r"Supreme Court of ([A-Za-z]+)", text)
    if m:
        return f"Supreme Court of {m.group(1)}"
    return "Supreme Court of Indiana"


def extract_case_name(json_metadata: Dict, text: str, file_stem: str) -> str:
    """Return case name from metadata, text header, or file stem."""
    import re
    if json_metadata and json_metadata.get("case_name"):
        return json_metadata["case_name"]
    m = re.search(
        r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+and\s+[A-Z][a-z]+)?)"
        r"\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text[:500],
    )
    if m:
        return f"{m.group(1)} v. {m.group(2)}"
    return file_stem
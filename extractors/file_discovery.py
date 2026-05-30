"""
extractors/file_discovery.py
-----------------------------
Walks the nested folder structure:
    <root>/<source_folder>/html/*.html
    <root>/<source_folder>/json/*.json
    <root>/<source_folder>/metadata/all_metadata.json

Returns a dict keyed by source_folder name, each value containing
a list of matched case dicts ready for processing.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def discover_and_match_files(root_path: str) -> Dict:
    """
    Scan *root_path* for sub-folders that contain html/ and json/ directories.
    Match files by base-name (stem) and attach any available metadata.

    Returns
    -------
    dict  {source_folder_name: {"path": Path, "cases": [...], "total_cases": int}}
    """
    structure: Dict = {}

    for source_folder in sorted(os.listdir(root_path)):
        source_path = Path(root_path) / source_folder

        if not source_path.is_dir():
            continue
        if source_folder.startswith(".") or source_folder == "processed_data":
            continue

        logger.info("\n📁 Scanning source: %s", source_folder)

        html_folder    = source_path / "html"
        json_folder    = source_path / "json"
        metadata_file  = source_path / "metadata" / "all_metadata.json"

        if not html_folder.exists():
            logger.warning("  ⚠️  html/ not found in %s", source_folder)
            continue
        if not json_folder.exists():
            logger.warning("  ⚠️  json/ not found in %s", source_folder)
            continue

        html_files = list(html_folder.glob("*.html"))
        json_files = list(json_folder.glob("*.json"))
        logger.info("  Found %d HTML, %d JSON files", len(html_files), len(json_files))

        html_map = {f.stem: f for f in html_files}
        json_map = {f.stem: f for f in json_files}
        common   = set(html_map) & set(json_map)

        master_metadata = _load_master_metadata(metadata_file)

        cases = [
            _build_case_entry(name, source_folder, html_map, json_map, master_metadata)
            for name in common
        ]

        structure[source_folder] = {
            "path":        source_path,
            "cases":       cases,
            "total_cases": len(cases),
        }
        logger.info("  ✅ Matched %d cases", len(cases))

        _warn_unmatched(html_map, json_map, common)

    return structure


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_master_metadata(metadata_file: Path) -> Dict:
    """Load all_metadata.json and return a dict keyed by id and name."""
    if not metadata_file.exists():
        return {}
    try:
        with open(metadata_file, "r", encoding="utf-8") as fh:
            metadata_list = json.load(fh)
        result: Dict = {}
        for meta in metadata_list:
            case_id   = str(meta.get("id", ""))
            case_name = meta.get("name", "")
            if case_id:
                result[case_id] = meta
            if case_name:
                result[case_name] = meta
        logger.info("  Loaded %d metadata entries", len(result))
        return result
    except Exception as exc:
        logger.warning("  Could not load metadata: %s", exc)
        return {}


def _build_case_entry(
    name: str,
    source_folder: str,
    html_map: Dict,
    json_map: Dict,
    master_metadata: Dict,
) -> Dict:
    """Build a single case dict from matched file stems."""
    case_id_match = re.search(r"(\d{4})", name)
    case_id       = case_id_match.group(1) if case_id_match else name

    case_metadata: Optional[Dict] = None
    for key in [case_id, name, (case_id_match.group(0) if case_id_match else None)]:
        if key and key in master_metadata:
            case_metadata = master_metadata[key]
            break

    return {
        "source":    source_folder,
        "case_name": name,
        "case_id":   case_id,
        "html_file": html_map[name],
        "json_file": json_map[name],
        "metadata":  case_metadata,
    }


def _warn_unmatched(html_map: Dict, json_map: Dict, common: set) -> None:
    unmatched_html = set(html_map) - common
    unmatched_json = set(json_map) - common
    if unmatched_html:
        logger.warning("  ⚠️  Unmatched HTML: %s …", list(unmatched_html)[:3])
    if unmatched_json:
        logger.warning("  ⚠️  Unmatched JSON: %s …", list(unmatched_json)[:3])
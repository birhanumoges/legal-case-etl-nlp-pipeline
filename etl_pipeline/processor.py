"""
pipeline/processor.py
----------------------
Core ETL orchestrator.

Flow
----
  discover files → (skip checkpointed) → extract each case
  → validate → save → report

Uses the proven extractor logic:
  • verdict_extractor  → 90 % known-verdict rate
  • classifier         → 26 % Criminal Law detection
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from extractors.file_discovery     import discover_and_match_files
from extractors.text_extractor     import extract_text_from_html
from extractors.metadata_extractor import (extract_metadata_from_json,
                                            extract_year, extract_court,
                                            extract_case_name)
from extractors.verdict_extractor  import extract_verdict
from extractors.classifier         import extract_case_type, extract_sub_type
from extractors.citation_extractor import extract_citations
from pipeline.validator            import validate
from pipeline.checkpoint           import Checkpoint
from pipeline.reporter             import save_outputs, generate_summary_report
from utils.logger                  import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Single-case extraction (module-level for ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def process_single_case(case_info: Dict) -> Optional[Dict]:
    """
    Run every extractor on one case dict and return a flat record dict.
    Returns None on empty text or unrecoverable error.
    """
    try:
        case_text, text_length = extract_text_from_html(case_info["html_file"])
        if not case_text:
            return None

        json_meta = extract_metadata_from_json(case_info["json_file"])
        verdict   = extract_verdict(case_text)
        case_type = extract_case_type(case_text)
        sub_type  = extract_sub_type(case_text, case_type)
        citations = extract_citations(case_text, json_meta)
        year      = extract_year(case_text, json_meta)
        court     = extract_court(json_meta, case_text)
        case_name = extract_case_name(json_meta, case_text, case_info["case_name"])

        return {
            "Case_ID":               case_info["case_id"],
            "Case_Name":             case_name,
            "Source_Folder":         case_info["source"],
            "Year":                  year,
            "Court":                 court,
            "Case_Text":             case_text[: config.CASE_TEXT_MAX_CHARS],
            "Case_Text_Full_Length": text_length,
            "Verdict":               verdict,
            "Case_Type":             case_type,
            "Sub_Type":              sub_type,
            "Num_Citations":         len(citations),
            "Legal_Citations":       "; ".join(citations[: config.MAX_CITATIONS_STORED]),
            "Decision_Date":         json_meta.get("decision_date", ""),
            "Docket_Number":         json_meta.get("docket_number", ""),
            "First_Page":            json_meta.get("first_page", ""),
            "Last_Page":             json_meta.get("last_page", ""),
        }
    except Exception as exc:
        logger.error("Error processing %s: %s", case_info.get("case_name", "?"), exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline run
# ─────────────────────────────────────────────────────────────────────────────

def run_etl_pipeline(resume: bool = True) -> pd.DataFrame:
    """
    Full ETL pipeline.

    Parameters
    ----------
    resume : if True, skip cases already recorded in the checkpoint file.

    Returns
    -------
    pd.DataFrame with all extracted cases (empty on failure).
    """
    logger.info("=" * 65)
    logger.info("ETL PIPELINE  START  %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 65)

    # Step 1 – Discover files
    logger.info("Step 1: Discovering files …")
    structure = discover_and_match_files(config.ROOT_PATH)
    if not structure:
        logger.error("No valid sources found under: %s", config.ROOT_PATH)
        return pd.DataFrame()

    all_cases: List[Dict] = []
    for src, info in structure.items():
        logger.info("  %-35s → %d cases", src, info["total_cases"])
        all_cases.extend(info["cases"])
    logger.info("Total cases found: %d", len(all_cases))

    # Step 2 – Checkpoint filter
    ckpt_path  = os.path.join(config.CHECKPOINT_DIR, "etl_checkpoint.json")
    checkpoint = Checkpoint(ckpt_path)

    if resume and len(checkpoint) > 0:
        before = len(all_cases)
        all_cases = [c for c in all_cases if not checkpoint.is_done(c["case_id"])]
        logger.info(
            "Checkpoint: skipping %d already-done cases, %d remaining.",
            before - len(all_cases), len(all_cases),
        )

    if not all_cases:
        logger.info("All cases already processed. Loading existing output.")
        existing = os.path.join(config.ETL_DIR, "legal_cases_complete.csv")
        return pd.read_csv(existing) if os.path.exists(existing) else pd.DataFrame()

    # Step 3 – Process in parallel
    logger.info("Step 3: Processing %d cases (workers=%d) …",
                len(all_cases), config.MAX_WORKERS)
    results: List[Dict] = []

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        future_map = {pool.submit(process_single_case, c): c for c in all_cases}
        total      = len(future_map)

        for idx, future in enumerate(as_completed(future_map), 1):
            case_info = future_map[future]
            try:
                record = future.result()
                if record:
                    results.append(record)
                    checkpoint.mark_done(case_info["case_id"])
                    logger.info("[%d/%d] ✅  %s", idx, total, case_info["case_name"])
                else:
                    logger.warning("[%d/%d] ⚠️   skipped %s", idx, total,
                                   case_info["case_name"])
            except Exception as exc:
                logger.error("[%d/%d] ❌  %s — %s", idx, total,
                             case_info["case_name"], exc)

    # Step 4 – Build DataFrame
    logger.info("Step 4: Building DataFrame …")
    df = pd.DataFrame(results)
    if df.empty:
        logger.error("No records produced — aborting.")
        return df

    # Step 5 – Validate
    logger.info("Step 5: Validating output …")
    vr = validate(df)
    vr.report()

    # Step 6 – Save
    logger.info("Step 6: Saving outputs …")
    save_outputs(df, config.ETL_DIR)

    # Step 7 – Report
    generate_summary_report(df, config.ETL_DIR)

    logger.info("=" * 65)
    logger.info("ETL COMPLETE  %d cases  %s",
                len(df), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("=" * 65)
    return df
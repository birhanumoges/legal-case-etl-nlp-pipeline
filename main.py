"""
main.py
--------
Entry point for the Legal NLP ETL Pipeline.

Usage
-----
    python main.py

Edit config.py to change ROOT_PATH and OUTPUT_DIR before running.
"""

import logging
import sys
from pathlib import Path

# ── Ensure project root is always on sys.path ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from etl_pipeline.processor import run_etl_pipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    df = run_etl_pipeline()

    if df.empty:
        logger.error("Pipeline produced no output. Check ROOT_PATH in config.py.")
        sys.exit(1)

    # ── Quick console preview ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (first 5 rows):")
    print("=" * 70)
    preview_cols = ["Case_ID", "Year", "Court", "Case_Type", "Sub_Type", "Verdict", "Num_Citations"]
    available = [c for c in preview_cols if c in df.columns]
    print(df[available].head(5).to_string(index=False))

    # ── Spot-check: Murphy case ───────────────────────────────────────────────
    murphy = df[df["Case_Name"].str.contains("Murphy", na=False)]
    if not murphy.empty:
        print("\n" + "=" * 70)
        print("SPOT-CHECK: Murphy case(s)")
        print("=" * 70)
        row = murphy.iloc[0]
        print(f"  Case_Type : {row['Case_Type']}")
        print(f"  Sub_Type  : {row['Sub_Type']}")   # should be "Larceny", not "General"
        print(f"  Verdict   : {row['Verdict']}")
        print(f"  Citations : {row['Num_Citations']}")


if __name__ == "__main__":
    main()
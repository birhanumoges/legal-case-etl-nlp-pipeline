"""
config.py
---------
Central configuration for the Legal NLP ETL Pipeline.
Edit ROOT_PATH and OUTPUT_DIR to match your environment.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_PATH  = "C:/Users/DELL/Downloads/Legal Case"          # folder that holds allcase/
OUTPUT_DIR = "C:/Users/DELL/Downloads/Legal Case/processed_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Required output columns ──────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Case_ID", "Year", "Court", "Case_Text",
    "Verdict", "Legal_Citations", "Case_Type", "Sub_Type",
]

# ── Text limits ──────────────────────────────────────────────────────────────
CASE_TEXT_MAX_CHARS   = 10_000   # characters stored in Case_Text column
MAX_CITATIONS_STORED  = 20       # max citations stored per case

# ── Parallel processing ──────────────────────────────────────────────────────
MAX_WORKERS = 4                  # ProcessPoolExecutor workers

# ── Logging format ───────────────────────────────────────────────────────────
LOG_FORMAT  = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL   = "INFO"             # DEBUG / INFO / WARNING / ERROR
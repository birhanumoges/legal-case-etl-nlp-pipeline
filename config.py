"""
config.py
---------
Single source of truth for all paths, constants, and hyper-parameters
used across ETL, preprocessing, modeling, and evaluation stages.

Edit the two PATH lines at the top before running.
"""

import os

# ── 1. PATHS  (edit these) ────────────────────────────────────────────────────
ROOT_PATH  = "C:/Users/DELL/Downloads/Legal Case"          # raw HTML + JSON root
OUTPUT_DIR = "C:/Users/DELL/Downloads/Legal Case/output"   # all generated files

# ── 2. Derived paths (auto-built, do not edit) ────────────────────────────────
ETL_DIR        = os.path.join(OUTPUT_DIR, "etl")          # CSV / JSON / Parquet
SPLITS_DIR     = os.path.join(OUTPUT_DIR, "splits")       # train / val / test CSVs
MODELS_DIR     = os.path.join(OUTPUT_DIR, "models")       # saved .pkl model files
EVAL_DIR       = os.path.join(OUTPUT_DIR, "evaluation")   # metrics + plots
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")  # ETL resume state
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")         # log files

for _d in (ETL_DIR, SPLITS_DIR, MODELS_DIR, EVAL_DIR, CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

# ── 3. ETL settings ───────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Case_ID", "Year", "Court", "Case_Text",
    "Verdict", "Legal_Citations", "Case_Type", "Sub_Type",
]
CASE_TEXT_MAX_CHARS  = 10_000   # characters stored per case
MAX_CITATIONS_STORED = 20       # max citations stored per case
MAX_WORKERS          = 4        # parallel ETL workers

# ── 4. Preprocessing settings ─────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE  = (1, 2)
TEST_SIZE          = 0.15       # fraction for test set
VAL_SIZE           = 0.15       # fraction for validation set
RANDOM_STATE       = 42

# ── 5. Modeling targets ───────────────────────────────────────────────────────
TARGET_CASE_TYPE  = "Case_Type"   # top-level classifier
TARGET_SUB_TYPE   = "Sub_Type"    # hierarchical sub-type classifier
TARGET_VERDICT    = "Verdict"     # verdict predictor

# ── 6. Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
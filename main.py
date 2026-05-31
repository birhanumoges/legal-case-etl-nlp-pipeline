"""
main.py
--------
Command-line entry point for the Legal NLP Pipeline.

Stages
------
  1. etl         – discover raw files → extract → save CSV/JSON/Parquet
  2. preprocess  – clean → features → TF-IDF encode → train/val/test split
  3. model       – train CaseClassifier, SubtypeClassifier, VerdictPredictor
  4. evaluate    – metrics, confusion matrices, per-class reports, plots

Usage
-----
    python main.py                    # run all four stages end-to-end
    python main.py --stage etl
    python main.py --stage preprocess
    python main.py --stage model
    python main.py --stage evaluate
    python main.py --stage model --model-type svm
    python main.py --no-resume        # ignore checkpoint, reprocess all

Edit config.py to change ROOT_PATH, OUTPUT_DIR, or any hyperparameter.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Ensure project root is always importable ─────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – ETL
# ─────────────────────────────────────────────────────────────────────────────

def stage_etl(resume: bool = True):
    """Discover raw HTML/JSON files, extract all fields, save structured output."""
    from etl_pipeline.processor import run_etl_pipeline
    from evaluation.class_balance import print_balance_summary
    from visualization.distribution_plots import plot_all

    logger.info("=" * 65)
    logger.info("STAGE 1 — ETL")
    logger.info("=" * 65)

    df = run_etl_pipeline(resume=resume)

    if df.empty:
        logger.error("ETL produced no output. Check config.ROOT_PATH.")
        sys.exit(1)

    # Quick console preview
    print("\n" + "=" * 65)
    print("SAMPLE (first 5 rows)")
    print("=" * 65)
    cols = ["Case_ID", "Year", "Court", "Case_Type", "Sub_Type",
            "Verdict", "Num_Citations"]
    print(df[[c for c in cols if c in df.columns]].head(5).to_string(index=False))

    # Class balance report
    print_balance_summary(df, cols=["Case_Type", "Sub_Type", "Verdict"])

    # Distribution plots
    viz_dir = os.path.join(config.EVAL_DIR, "etl_plots")
    plot_all(df, viz_dir)
    logger.info("ETL distribution plots → %s", viz_dir)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 – Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def stage_preprocess(df=None):
    """Clean text, build features, TF-IDF encode, split into train/val/test."""
    import pandas as pd
    from preprocessing.cleaner  import clean_dataframe
    from preprocessing.features import build_feature_matrix, LEGAL_FLAG_COLS, build_legal_flags
    from preprocessing.encoder  import encode_labels, TfidfEncoder
    from preprocessing.splitter import split_data, save_splits

    logger.info("=" * 65)
    logger.info("STAGE 2 — PREPROCESSING")
    logger.info("=" * 65)

    # Load from disk if not passed in
    if df is None:
        csv = os.path.join(config.ETL_DIR, "legal_cases_complete.csv")
        if not os.path.exists(csv):
            logger.error("ETL CSV not found: %s  — run --stage etl first.", csv)
            sys.exit(1)
        df = pd.read_csv(csv, encoding="utf-8")
        logger.info("Loaded %d cases ← %s", len(df), csv)

    # 2a. Clean text
    logger.info("2a. Cleaning text …")
    df = clean_dataframe(df, col="Case_Text")

    # 2b. Add legal flag features
    logger.info("2b. Building legal flag features …")
    df = build_legal_flags(df, text_col="Case_Text")

    # 2c. Split
    logger.info("2c. Splitting into train/val/test …")
    splits = split_data(
        df,
        strategy     = "random",
        label_col    = config.TARGET_CASE_TYPE,
        test_size    = config.TEST_SIZE,
        val_size     = config.VAL_SIZE,
        random_state = config.RANDOM_STATE,
    )
    save_splits(splits, config.SPLITS_DIR)

    # 2d. TF-IDF encode (fit on train only)
    logger.info("2d. TF-IDF encoding …")
    extra_cols = [c for c in LEGAL_FLAG_COLS if c in splits["train"].columns]
    encoder = TfidfEncoder(
        max_features       = config.TFIDF_MAX_FEATURES,
        ngram_range        = config.TFIDF_NGRAM_RANGE,
        extra_feature_cols = extra_cols,
    )
    X_train = encoder.fit_transform(splits["train"], text_col="Case_Text")
    X_val   = encoder.transform(splits["val"],       text_col="Case_Text")
    X_test  = encoder.transform(splits["test"],      text_col="Case_Text")

    enc_path = os.path.join(config.SPLITS_DIR, "tfidf_encoder.pkl")
    encoder.save(enc_path)

    # 2e. Encode labels
    logger.info("2e. Encoding labels …")
    y_train, classes = encode_labels(splits["train"], col=config.TARGET_CASE_TYPE)
    y_val,   _       = encode_labels(splits["val"],   col=config.TARGET_CASE_TYPE)
    y_test,  _       = encode_labels(splits["test"],  col=config.TARGET_CASE_TYPE)

    logger.info(
        "Preprocessing complete.  train=%s  val=%s  test=%s  classes=%d",
        X_train.shape, X_val.shape, X_test.shape, len(classes),
    )

    return {
        "splits":  splits,
        "encoder": encoder,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "classes": classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – Modeling
# ─────────────────────────────────────────────────────────────────────────────

def stage_model(prep=None, model_type: str = "svm"):
    """Train CaseClassifier, SubtypeClassifier, and VerdictPredictor."""
    import os
    import pandas as pd
    from preprocessing.encoder  import TfidfEncoder
    from preprocessing.splitter import load_splits
    from modeling.case_classifier    import CaseClassifier
    from modeling.subtype_classifier import SubtypeClassifier
    from modeling.verdict_predictor  import VerdictPredictor
    from sklearn.metrics import f1_score

    logger.info("=" * 65)
    logger.info("STAGE 3 — MODELING  (model_type=%s)", model_type)
    logger.info("=" * 65)

    # Load preprocessed artefacts if not passed in
    if prep is None:
        enc_path = os.path.join(config.SPLITS_DIR, "tfidf_encoder.pkl")
        if not os.path.exists(enc_path):
            logger.error("Encoder not found: %s  — run --stage preprocess first.", enc_path)
            sys.exit(1)
        splits  = load_splits(config.SPLITS_DIR)
        encoder = TfidfEncoder.load(enc_path)
        prep    = {"splits": splits, "encoder": encoder}

    splits = prep["splits"]

    # ── 3a. CaseClassifier ────────────────────────────────────────────────────
    logger.info("\n3a. Training CaseClassifier …")
    case_model = CaseClassifier(
        model_type   = model_type,
        max_features = config.TFIDF_MAX_FEATURES,
        ngram_range  = config.TFIDF_NGRAM_RANGE,
    ).train(splits["train"])

    val_m = case_model.evaluate(splits["val"])
    logger.info("  Val accuracy=%.4f  F1-macro=%.4f",
                val_m["accuracy"], val_m["f1_macro"])

    case_path = os.path.join(config.MODELS_DIR, "case_classifier.pkl")
    case_model.save(case_path)

    # ── 3b. SubtypeClassifier ─────────────────────────────────────────────────
    logger.info("\n3b. Training SubtypeClassifier (hierarchical) …")
    sub_model = SubtypeClassifier(
        model_type   = model_type,
        max_features = min(config.TFIDF_MAX_FEATURES, 20_000),
        ngram_range  = config.TFIDF_NGRAM_RANGE,
    ).train(splits["train"])

    sub_val_m = sub_model.evaluate(splits["val"])
    logger.info("  Val accuracy=%.4f  F1-macro=%.4f  sub-models=%d",
                sub_val_m["accuracy"], sub_val_m["f1_macro"],
                sub_val_m["num_sub_models"])

    sub_path = os.path.join(config.MODELS_DIR, "subtype_classifier.pkl")
    sub_model.save(sub_path)

    # ── 3c. VerdictPredictor ──────────────────────────────────────────────────
    logger.info("\n3c. Training VerdictPredictor …")
    verdict_model = VerdictPredictor(
        model_type   = model_type,
        max_features = config.TFIDF_MAX_FEATURES,
        ngram_range  = config.TFIDF_NGRAM_RANGE,
    ).train(splits["train"])

    verdict_val_m = verdict_model.evaluate(splits["val"])
    logger.info("  Val accuracy=%.4f  F1-macro=%.4f",
                verdict_val_m["accuracy"], verdict_val_m["f1_macro"])

    verdict_path = os.path.join(config.MODELS_DIR, "verdict_predictor.pkl")
    verdict_model.save(verdict_path)

    logger.info("\nAll models saved → %s", config.MODELS_DIR)

    return {
        "case_model":    case_model,
        "sub_model":     sub_model,
        "verdict_model": verdict_model,
        "splits":        splits,
        "val_metrics": {
            "case_classifier":    val_m,
            "subtype_classifier": sub_val_m,
            "verdict_predictor":  verdict_val_m,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def stage_evaluate(modeling_result=None):
    """Evaluate all three models on the held-out test set."""
    import os
    from preprocessing.splitter  import load_splits
    from modeling.case_classifier    import CaseClassifier
    from modeling.subtype_classifier import SubtypeClassifier
    from modeling.verdict_predictor  import VerdictPredictor
    from evaluation.reporter         import EvalReporter
    from evaluation.class_balance    import print_balance_summary
    from visualization.distribution_plots import plot_all
    from visualization.confusion_matrix   import plot_all_confusion_matrices
    from preprocessing.encoder import encode_labels
    from modeling.verdict_predictor import map_verdict_to_group

    logger.info("=" * 65)
    logger.info("STAGE 4 — EVALUATION")
    logger.info("=" * 65)

    reporter = EvalReporter(output_dir=config.EVAL_DIR)

    # Load from disk if not passed in
    if modeling_result is None:
        splits        = load_splits(config.SPLITS_DIR)
        case_model    = CaseClassifier.load(
            os.path.join(config.MODELS_DIR, "case_classifier.pkl"))
        sub_model     = SubtypeClassifier.load(
            os.path.join(config.MODELS_DIR, "subtype_classifier.pkl"))
        verdict_model = VerdictPredictor.load(
            os.path.join(config.MODELS_DIR, "verdict_predictor.pkl"))
    else:
        splits        = modeling_result["splits"]
        case_model    = modeling_result["case_model"]
        sub_model     = modeling_result["sub_model"]
        verdict_model = modeling_result["verdict_model"]

    df_test = splits["test"]

    # ── 4a. CaseClassifier ────────────────────────────────────────────────────
    logger.info("\n4a. Evaluating CaseClassifier …")
    case_metrics = reporter.report_case_classifier(case_model, df_test)

    # ── 4b. SubtypeClassifier ─────────────────────────────────────────────────
    logger.info("\n4b. Evaluating SubtypeClassifier …")
    sub_metrics = reporter.report_subtype_classifier(sub_model, df_test)

    # ── 4c. VerdictPredictor ──────────────────────────────────────────────────
    logger.info("\n4c. Evaluating VerdictPredictor …")
    verdict_metrics = reporter.report_verdict_predictor(verdict_model, df_test)

    # ── 4d. Summary table ─────────────────────────────────────────────────────
    all_metrics = {
        "CaseClassifier":    case_metrics,
        "SubtypeClassifier": sub_metrics,
        "VerdictPredictor":  verdict_metrics,
    }
    reporter.print_summary(all_metrics)

    # ── 4e. Class balance on test set ─────────────────────────────────────────
    print_balance_summary(df_test, cols=["Case_Type", "Sub_Type"])

    # ── 4f. Additional distribution plots ────────────────────────────────────
    plot_all(df_test, os.path.join(config.EVAL_DIR, "test_plots"))

    logger.info("\nEvaluation complete. Reports → %s", config.EVAL_DIR)
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Legal NLP Pipeline — ETL → Preprocess → Model → Evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python main.py                          # full pipeline
  python main.py --stage etl              # ETL only
  python main.py --stage preprocess       # preprocess only (needs ETL output)
  python main.py --stage model            # train models (needs preprocess output)
  python main.py --stage model --model-type logistic
  python main.py --stage evaluate         # evaluate (needs model output)
  python main.py --no-resume              # reprocess all files (ignore checkpoint)
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["etl", "preprocess", "model", "evaluate", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--model-type",
        default="svm",
        choices=["logistic", "svm", "random_forest", "naive_bayes", "xgboost"],
        help="sklearn model used for all three classifiers (default: svm)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore ETL checkpoint and reprocess all files from scratch",
    )
    return parser.parse_args()


def main():
    args   = _parse_args()
    resume = not args.no_resume

    df   = None
    prep = None
    mod  = None

    if args.stage in ("etl", "all"):
        df = stage_etl(resume=resume)

    if args.stage in ("preprocess", "all"):
        prep = stage_preprocess(df=df)

    if args.stage in ("model", "all"):
        mod = stage_model(prep=prep, model_type=args.model_type)

    if args.stage in ("evaluate", "all"):
        stage_evaluate(modeling_result=mod)

    logger.info("\n🎉 Pipeline finished successfully.")


if __name__ == "__main__":
    main()
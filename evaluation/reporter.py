"""
evaluation/reporter.py
-----------------------
Generates full evaluation reports for each of the three models:
  CaseClassifier, SubtypeClassifier, VerdictPredictor.

Responsibilities
----------------
  • Run inference on the test split.
  • Compute metrics via evaluation/metrics.py.
  • Save JSON metric files.
  • Save per-class CSV reports.
  • Produce confusion-matrix and class-distribution plots.
  • Print a human-readable summary to stdout.

Public API
----------
    EvalReporter(output_dir)
    .report_case_classifier(model, df_test)   -> dict
    .report_subtype_classifier(model, df_test) -> dict
    .report_verdict_predictor(model, df_test)  -> dict
    .save_metrics(metrics, filename)
    .print_summary(all_metrics)
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from evaluation.metrics      import (compute_metrics, confusion_matrix_df,
                                      per_class_report)
from preprocessing.encoder   import encode_labels
from modeling.verdict_predictor import map_verdict_to_group

logger = logging.getLogger(__name__)


class EvalReporter:
    """
    Writes all evaluation artefacts to *output_dir*.

    Parameters
    ----------
    output_dir : directory where reports, CSVs, and plots are saved.
                 Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Case_Type classifier ──────────────────────────────────────────────────

    def report_case_classifier(
        self,
        model,
        df_test: pd.DataFrame,
    ) -> dict:
        """
        Full evaluation report for CaseClassifier.
        Saves: test_case_type_metrics.json,
               test_case_type_per_class.csv,
               test_case_type_confusion.png
        Returns the metrics dict.
        """
        logger.info("Evaluating CaseClassifier on %d test rows …", len(df_test))
        from preprocessing.features import build_feature_matrix

        X, _      = build_feature_matrix(df_test, vectorizer=model._vectorizer,
                                          max_features=model.max_features,
                                          ngram_range=model.ngram_range)
        y_true, _ = encode_labels(df_test, col="Case_Type")
        y_pred    = model._trainer.predict(X)
        classes   = model._classes

        metrics = compute_metrics(y_true, y_pred, classes, prefix="test_")
        self.save_metrics(metrics, "test_case_type_metrics.json")
        self._save_per_class(y_true, y_pred, classes, "test_case_type_per_class.csv")
        self._plot_confusion(y_true, y_pred, classes,
                             title="Case Type – Confusion Matrix",
                             filename="test_case_type_confusion.png")
        return metrics

    # ── Sub_Type classifier ───────────────────────────────────────────────────

    def report_subtype_classifier(
        self,
        model,
        df_test: pd.DataFrame,
    ) -> dict:
        """
        Full evaluation report for SubtypeClassifier.
        Saves: test_subtype_metrics.json,
               test_subtype_per_class.csv,
               test_subtype_confusion.png
        Returns the metrics dict.
        """
        logger.info("Evaluating SubtypeClassifier on %d test rows …", len(df_test))

        # Use true Case_Type for routing (test-time gold label)
        preds   = model.predict(df_test, case_type_col="Case_Type")
        y_true  = df_test["Sub_Type"].fillna("Unclassified").astype(str)
        y_pred  = preds.astype(str)

        # Encode to int for sklearn metrics
        all_labels = sorted(set(y_true) | set(y_pred))
        lbl2idx    = {l: i for i, l in enumerate(all_labels)}
        yt = np.array([lbl2idx[l] for l in y_true])
        yp = np.array([lbl2idx[l] for l in y_pred])
        classes = np.array(all_labels)

        metrics = compute_metrics(yt, yp, classes, prefix="test_")
        self.save_metrics(metrics, "test_subtype_metrics.json")
        self._save_per_class(yt, yp, classes, "test_subtype_per_class.csv")
        self._plot_confusion(yt, yp, classes,
                             title="Sub-Type – Confusion Matrix",
                             filename="test_subtype_confusion.png")
        return metrics

    # ── Verdict predictor ─────────────────────────────────────────────────────

    def report_verdict_predictor(
        self,
        model,
        df_test: pd.DataFrame,
    ) -> dict:
        """
        Full evaluation report for VerdictPredictor.
        Saves: test_verdict_metrics.json,
               test_verdict_per_class.csv,
               test_verdict_confusion.png
        Returns the metrics dict.
        """
        logger.info("Evaluating VerdictPredictor on %d test rows …", len(df_test))

        df = df_test.copy()
        df["Verdict_Group"] = df["Verdict"].apply(map_verdict_to_group)
        known   = df[df["Verdict_Group"] != "unknown"].copy()

        if known.empty:
            logger.warning("No known-verdict rows in test set — skipping.")
            return {}

        preds   = model.predict(known)
        y_true  = known["Verdict_Group"].astype(str)
        y_pred  = preds.astype(str)

        all_labels = sorted(set(y_true) | set(y_pred))
        lbl2idx    = {l: i for i, l in enumerate(all_labels)}
        yt = np.array([lbl2idx[l] for l in y_true])
        yp = np.array([lbl2idx[l] for l in y_pred])
        classes = np.array(all_labels)

        metrics = compute_metrics(yt, yp, classes, prefix="test_")
        self.save_metrics(metrics, "test_verdict_metrics.json")
        self._save_per_class(yt, yp, classes, "test_verdict_per_class.csv")
        self._plot_confusion(yt, yp, classes,
                             title="Verdict Group – Confusion Matrix",
                             filename="test_verdict_confusion.png")
        return metrics

    # ── Shared helpers ────────────────────────────────────────────────────────

    def save_metrics(self, metrics: dict, filename: str) -> None:
        """Persist a metrics dict as a formatted JSON file."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved → %s", path)

    def print_summary(self, all_metrics: Dict[str, dict]) -> None:
        """Print a side-by-side summary of all three model evaluations."""
        lines = [
            "",
            "=" * 65,
            "EVALUATION SUMMARY",
            "=" * 65,
            f"  {'Model':<30} {'Accuracy':>10} {'F1-macro':>10} {'Kappa':>10}",
            "-" * 65,
        ]
        for model_name, metrics in all_metrics.items():
            acc   = metrics.get("test_accuracy",   metrics.get("accuracy",   0.0))
            f1    = metrics.get("test_f1_macro",   metrics.get("f1_macro",   0.0))
            kappa = metrics.get("test_cohen_kappa",metrics.get("cohen_kappa",0.0))
            lines.append(f"  {model_name:<30} {acc:>10.4f} {f1:>10.4f} {kappa:>10.4f}")
        lines.append("=" * 65)
        print("\n".join(lines))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _save_per_class(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: np.ndarray,
        filename: str,
    ) -> None:
        df = per_class_report(y_true, y_pred, classes)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, encoding="utf-8")
        logger.info("Per-class report → %s", path)

    def _plot_confusion(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: np.ndarray,
        title: str,
        filename: str,
    ) -> None:
        """Save a confusion-matrix heatmap. Silently skips if matplotlib missing."""
        try:
            import matplotlib
            matplotlib.use("Agg")          # non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping %s.", filename)
            return

        cm_df   = confusion_matrix_df(y_true, y_pred, classes)
        n       = len(cm_df)
        figsize = (max(8, n // 2), max(6, n // 2))
        annotate= n <= 25

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm_df, annot=annotate, fmt="d",
            cmap="Blues", linewidths=0.3, ax=ax,
        )
        ax.set_title(title, fontsize=13, pad=10)
        ax.set_ylabel("True label", fontsize=10)
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix → %s", path)
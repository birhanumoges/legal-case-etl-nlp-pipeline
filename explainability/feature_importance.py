"""explainability/feature_importance.py — TF-IDF feature importance from model coefficients."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PLOT_DIR, REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureImportanceAnalyzer:

    def analyze(self, pipeline, feature_names: list[str], target_name: str, top_n: int = 30):
        clf = self._get_clf(pipeline)
        if clf is None:
            return
        coef = self._get_coef(clf)
        if coef is None:
            return

        if coef.ndim == 2:
            # Multi-class: mean absolute value across classes
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.abs(coef)

        if len(importance) != len(feature_names):
            logger.warning("Feature count mismatch; skipping importance plot")
            return

        top_idx   = np.argsort(importance)[::-1][:top_n]
        top_names = [feature_names[i] for i in top_idx]
        top_vals  = importance[top_idx]

        # Save CSV
        df = pd.DataFrame({"feature": top_names, "importance": top_vals})
        df.to_csv(REPORT_DIR / f"{target_name}_feature_importance.csv", index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.3)))
        ax.barh(top_names[::-1], top_vals[::-1], color="teal")
        ax.set_title(f"Feature Importance — {target_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean |Coefficient|")
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f"feature_importance_{target_name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Feature importance plot saved for %s", target_name)

    @staticmethod
    def _get_clf(pipeline):
        if hasattr(pipeline, "named_steps"):
            return pipeline.named_steps.get("clf")
        return pipeline

    @staticmethod
    def _get_coef(clf):
        # Unwrap calibrated classifiers
        if hasattr(clf, "calibrated_classifiers_"):
            try:
                base = clf.calibrated_classifiers_[0].estimator
                return base.coef_ if hasattr(base, "coef_") else None
            except Exception:
                return None
        if hasattr(clf, "coef_"):
            return clf.coef_
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
        return None

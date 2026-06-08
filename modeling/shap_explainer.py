"""modeling/shap_explainer.py — SHAP integration for model explainability."""

from __future__ import annotations
import numpy as np
import pandas as pd
import json
from pathlib import Path
from config import REPORT_DIR, PLOT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    """
    Compute SHAP values for a fitted pipeline.
    Supports LinearExplainer (LR / SVM) and TreeExplainer (XGBoost).
    Falls back gracefully if SHAP not installed.
    """

    def __init__(self, pipeline, feature_names: list[str] | None = None,
                 target_name: str = "target"):
        self.pipeline      = pipeline
        self.feature_names = feature_names
        self.target_name   = target_name
        self._shap_values  = None

    def explain(self, X, max_samples: int = 500) -> dict | None:
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed; skipping explainability.")
            return None

        # Extract final estimator
        clf = self._get_clf()
        if clf is None:
            return None

        # Sample for speed
        if hasattr(X, "toarray"):
            X_sample = X[:max_samples]
        else:
            X_sample = X[:max_samples]

        try:
            explainer = shap.LinearExplainer(clf, X_sample, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_sample)
        except Exception:
            try:
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_sample)
            except Exception as e:
                logger.warning("SHAP explainer failed: %s", e)
                return None

        self._shap_values = shap_vals
        summary = self._top_features(shap_vals, n=30)
        out = {"target": self.target_name, "top_features": summary}

        path = REPORT_DIR / f"{self.target_name}_shap_summary.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        logger.info("SHAP summary saved → %s", path)
        return out

    def _top_features(self, shap_vals, n: int = 30) -> list[dict]:
        if isinstance(shap_vals, list):
            vals = np.abs(np.array(shap_vals)).mean(axis=0)
            if vals.ndim > 1:
                vals = vals.mean(axis=0)
        else:
            vals = np.abs(shap_vals).mean(axis=0)
            if vals.ndim > 1:
                vals = vals.mean(axis=0)

        names = self.feature_names or [f"f{i}" for i in range(len(vals))]
        top_idx = np.argsort(vals)[::-1][:n]
        return [{"feature": names[i], "mean_abs_shap": float(vals[i])} for i in top_idx]

    def _get_clf(self):
        if hasattr(self.pipeline, "named_steps"):
            return self.pipeline.named_steps.get("clf")
        if hasattr(self.pipeline, "estimator"):
            return self.pipeline.estimator
        return self.pipeline

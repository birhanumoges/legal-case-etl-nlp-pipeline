"""explainability/shap_analysis.py — Run SHAP for all three targets."""

from __future__ import annotations
from modeling.shap_explainer import SHAPExplainer
from utils.logger import get_logger

logger = get_logger(__name__)


class ShapAnalysis:
    def run(self, models: dict, X_test, feature_names: list[str]):
        """
        models: { target_name: (best_name, best_pipeline) }
        """
        results = {}
        for target, (name, pipeline) in models.items():
            logger.info("Running SHAP for %s — %s", target, name)
            explainer = SHAPExplainer(pipeline, feature_names=feature_names, target_name=target)
            results[target] = explainer.explain(X_test)
        return results

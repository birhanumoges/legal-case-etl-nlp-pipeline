"""explainability/model_interpretation.py — High-level interpretation runner."""

from __future__ import annotations
from explainability.feature_importance import FeatureImportanceAnalyzer
from explainability.shap_analysis import ShapAnalysis
from visualization.shap_plots import plot_shap_summary
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelInterpreter:
    def __init__(self):
        self.fi   = FeatureImportanceAnalyzer()
        self.shap = ShapAnalysis()

    def run_all(self, best_models: dict, X_test, feature_names: list[str]):
        """
        best_models: { target_name: (best_model_name, fitted_pipeline) }
        """
        logger.info("Running model interpretation …")
        for target, (name, pipeline) in best_models.items():
            self.fi.analyze(pipeline, feature_names, target_name=target)

        self.shap.run(best_models, X_test, feature_names)

        for target in best_models:
            plot_shap_summary(target)

        logger.info("Model interpretation complete.")

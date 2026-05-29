"""
modeling/trainer.py
--------------------
Shared training logic used by case_classifier, subtype_classifier,
and verdict_predictor.

Supported model types
---------------------
  "logistic"     – LogisticRegression  (strong text baseline)
  "svm"          – LinearSVC + calibration (best for TF-IDF sparse)
  "random_forest"– RandomForestClassifier
  "naive_bayes"  – MultinomialNB (very fast)
  "xgboost"      – XGBClassifier (optional)

Public API
----------
    Trainer(model_type, **kwargs)
    .fit(X, y)
    .predict(X)            -> np.ndarray
    .predict_proba(X)      -> np.ndarray  (n_samples, n_classes)
    .cross_validate(X, y)  -> dict
    .save(path)
    Trainer.load(path)     -> Trainer
"""

import logging
import os
import pickle
import time
from typing import Dict, Optional

import numpy as np
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes     import MultinomialNB
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import MaxAbsScaler
from sklearn.svm             import LinearSVC

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "logistic":      {"max_iter": 1000, "C": 1.0,
                      "class_weight": "balanced", "solver": "saga"},
    "svm":           {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
    "random_forest": {"n_estimators": 200, "class_weight": "balanced", "n_jobs": -1},
    "naive_bayes":   {"alpha": 0.1},
}
_CLASSES = {
    "logistic":      LogisticRegression,
    "svm":           LinearSVC,
    "random_forest": RandomForestClassifier,
    "naive_bayes":   MultinomialNB,
}


class Trainer:
    def __init__(self, model_type: str = "logistic",
                 calibrate: bool = True, scale: bool = False, **kwargs):
        self.model_type = model_type.lower()
        self._pipeline  = self._build(calibrate, scale, kwargs)
        self._classes_  = None

    def _build(self, calibrate, scale, extra) -> Pipeline:
        if self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                est = XGBClassifier(n_estimators=300, max_depth=6,
                                    learning_rate=0.1, eval_metric="mlogloss",
                                    **extra)
            except ImportError:
                logger.warning("xgboost not found – falling back to logistic.")
                self.model_type = "logistic"
                return self._build(calibrate, scale, extra)
        elif self.model_type not in _CLASSES:
            raise ValueError(f"Unknown model_type '{self.model_type}'. "
                             f"Choose from: {list(_CLASSES)}")
        else:
            params = {**_DEFAULTS.get(self.model_type, {}), **extra}
            est    = _CLASSES[self.model_type](**params)

        if self.model_type == "svm" and calibrate:
            est = CalibratedClassifierCV(est, cv=3)

        steps = []
        if scale:
            steps.append(("scaler", MaxAbsScaler()))
        steps.append(("clf", est))
        return Pipeline(steps)

    def fit(self, X, y: np.ndarray) -> "Trainer":
        logger.info("Training %s on %d samples …", self.model_type, len(y))
        t0 = time.time()
        self._pipeline.fit(X, y)
        self._classes_ = np.unique(y)
        logger.info("Done in %.1fs.", time.time() - t0)
        return self

    def predict(self, X) -> np.ndarray:
        return self._pipeline.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self._pipeline.predict_proba(X)
        scores = clf.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def cross_validate(self, X, y, cv: int = 5,
                       scoring: str = "f1_macro") -> Dict:
        scores = cross_val_score(self._pipeline, X, y,
                                 cv=cv, scoring=scoring, n_jobs=-1)
        result = {"mean": float(scores.mean()), "std": float(scores.std()),
                  "scores": scores.tolist()}
        logger.info("CV %s: %.4f ± %.4f", scoring, result["mean"], result["std"])
        return result

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Trainer saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "Trainer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Trainer loaded ← %s", path)
        return obj

    @property
    def classes_(self) -> Optional[np.ndarray]:
        return self._classes_
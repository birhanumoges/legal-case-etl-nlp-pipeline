"""
modeling/case_classifier.py
-----------------------------
Top-level Case_Type classifier.

Predicts one of:
  Criminal Law | Civil Procedure | Contract Law - Debt |
  Property Law - Ejectment | Property Law - Execution Sale |
  Torts | Torts - Defamation | Unclassified

Public API
----------
    CaseClassifier(model_type)
    .train(df_train)
    .evaluate(df_test)    -> dict of metrics
    .predict(df)          -> pd.Series
    .predict_proba(df)    -> pd.DataFrame (label + confidence)
    .save(path)
    CaseClassifier.load(path)
"""

import os
import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from modeling.trainer        import Trainer
from preprocessing.features  import build_feature_matrix
from preprocessing.encoder   import encode_labels

logger = logging.getLogger(__name__)

TARGET = "Case_Type"


class CaseClassifier:
    """
    Top-level legal case-type classifier.

    Parameters
    ----------
    model_type : passed to Trainer ("logistic", "svm", "random_forest", …)
    max_features : TF-IDF vocabulary size
    ngram_range  : TF-IDF n-gram range
    """

    def __init__(self, model_type: str = "svm",
                 max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        self.model_type   = model_type
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self._trainer:  Optional[Trainer]  = None
        self._vectorizer                   = None
        self._classes:  Optional[np.ndarray] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df_train: pd.DataFrame) -> "CaseClassifier":
        """Fit TF-IDF + classifier on df_train."""
        logger.info("Training CaseClassifier on %d samples …", len(df_train))
        X, self._vectorizer = build_feature_matrix(
            df_train, max_features=self.max_features, ngram_range=self.ngram_range)
        y, self._classes = encode_labels(df_train, col=TARGET)

        self._trainer = Trainer(model_type=self.model_type, calibrate=True)
        self._trainer.fit(X, y)
        return self

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        X, _ = build_feature_matrix(df, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        y, _ = encode_labels(df, col=TARGET)
        return self._trainer.cross_validate(X, y, cv=cv)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return predicted Case_Type label for each row."""
        X, _ = build_feature_matrix(df, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        idx    = self._trainer.predict(X)
        labels = self._classes[idx]
        return pd.Series(labels, index=df.index, name="Predicted_Case_Type")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predicted label + confidence score for each row."""
        X, _ = build_feature_matrix(df, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        proba  = self._trainer.predict_proba(X)
        idx    = proba.argmax(axis=1)
        conf   = proba.max(axis=1)
        labels = self._classes[idx]
        return pd.DataFrame(
            {"Predicted_Case_Type": labels, "Case_Type_Confidence": conf},
            index=df.index,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Compute accuracy + F1 on df_test."""
        X, _ = build_feature_matrix(df_test, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        y_true, _ = encode_labels(df_test, col=TARGET)
        y_pred    = self._trainer.predict(X)
        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        logger.info("CaseClassifier eval — Accuracy: %.4f  F1-macro: %.4f",
                    metrics["accuracy"], metrics["f1_macro"])
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("CaseClassifier saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "CaseClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("CaseClassifier loaded ← %s", path)
        return obj
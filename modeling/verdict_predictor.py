"""
modeling/verdict_predictor.py
------------------------------
Verdict predictor — predicts a simplified verdict group from case text.

Verdict groups (simplified for ML)
------------------------------------
  affirmed | reversed | remanded | dismissed |
  plaintiff_wins | defendant_wins |
  award_granted | award_denied | unknown

Public API
----------
    VerdictPredictor(model_type)
    .train(df_train)
    .predict(df)         -> pd.Series of verdict group labels
    .predict_proba(df)   -> pd.DataFrame (label + confidence)
    .evaluate(df_test)   -> dict
    .save(path)
    VerdictPredictor.load(path)
"""

import logging
import os
import pickle
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from modeling.trainer       import Trainer
from preprocessing.features import build_feature_matrix
from preprocessing.encoder  import encode_labels

logger = logging.getLogger(__name__)

VERDICT_COL       = "Verdict"
VERDICT_GROUP_COL = "Verdict_Group"

# ── Verdict → simplified group mapping ───────────────────────────────────────
_VERDICT_GROUPS = [
    ("affirmed",       [r"affirm"]),
    ("reversed",       [r"revers"]),
    ("remanded",       [r"remand"]),
    ("dismissed",      [r"dismiss"]),
    ("plaintiff_wins", [r"judgment for plaintiff", r"verdict for plaintiff",
                        r"decree for plaintiff"]),
    ("defendant_wins", [r"judgment for defendant", r"verdict for defendant"]),
    ("award_granted",  [r"award granted", r"claim allowed", r"award recommended"]),
    ("award_denied",   [r"award denied", r"claim disallowed", r"claim denied",
                        r"award denied"]),
]


def map_verdict_to_group(verdict: str) -> str:
    """Map a raw verdict string to a simplified group label."""
    if not isinstance(verdict, str) or not verdict.strip():
        return "unknown"
    v = verdict.lower()
    for group, patterns in _VERDICT_GROUPS:
        if any(re.search(p, v) for p in patterns):
            return group
    return "unknown"


class VerdictPredictor:
    """
    Predicts simplified verdict groups from case text + features.

    Parameters
    ----------
    model_type   : "logistic" | "svm" | "random_forest" | "naive_bayes"
    max_features : TF-IDF vocabulary size
    """

    def __init__(self, model_type: str = "logistic",
                 max_features: int = 50_000, ngram_range: tuple = (1, 2)):
        self.model_type   = model_type
        self.max_features = max_features
        self.ngram_range  = ngram_range
        self._trainer:    Optional[Trainer]    = None
        self._vectorizer                       = None
        self._classes:    Optional[np.ndarray] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df_train: pd.DataFrame) -> "VerdictPredictor":
        """
        Add Verdict_Group column (simplified labels) and train the model.
        Rows with verdict group 'unknown' are excluded from training.
        """
        df = df_train.copy()
        df[VERDICT_GROUP_COL] = df[VERDICT_COL].apply(map_verdict_to_group)

        # Drop 'unknown' rows — they add noise
        df = df[df[VERDICT_GROUP_COL] != "unknown"]
        logger.info(
            "VerdictPredictor: training on %d samples (%d verdict groups) …",
            len(df), df[VERDICT_GROUP_COL].nunique(),
        )
        logger.info("Verdict group distribution:\n%s",
                    df[VERDICT_GROUP_COL].value_counts().to_string())

        X, self._vectorizer = build_feature_matrix(
            df, max_features=self.max_features, ngram_range=self.ngram_range)
        y, self._classes    = encode_labels(df, col=VERDICT_GROUP_COL)

        self._trainer = Trainer(model_type=self.model_type, calibrate=True)
        self._trainer.fit(X, y)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict verdict group for each row."""
        X, _ = build_feature_matrix(df, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        idx    = self._trainer.predict(X)
        labels = self._classes[idx]
        return pd.Series(labels, index=df.index, name="Predicted_Verdict_Group")

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict verdict group + confidence for each row."""
        X, _ = build_feature_matrix(df, vectorizer=self._vectorizer,
                                     max_features=self.max_features,
                                     ngram_range=self.ngram_range)
        proba  = self._trainer.predict_proba(X)
        idx    = proba.argmax(axis=1)
        conf   = proba.max(axis=1)
        labels = self._classes[idx]
        return pd.DataFrame(
            {"Predicted_Verdict_Group": labels, "Verdict_Confidence": conf},
            index=df.index,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Evaluate on df_test using Verdict_Group as ground truth."""
        df       = df_test.copy()
        df[VERDICT_GROUP_COL] = df[VERDICT_COL].apply(map_verdict_to_group)
        known    = df[df[VERDICT_GROUP_COL] != "unknown"]

        if known.empty:
            logger.warning("No known-verdict rows in test set.")
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        preds  = self.predict(known)
        y_true = known[VERDICT_GROUP_COL].astype(str)
        y_pred = preds.astype(str)

        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        logger.info("VerdictPredictor eval — Accuracy: %.4f  F1-macro: %.4f",
                    metrics["accuracy"], metrics["f1_macro"])
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("VerdictPredictor saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "VerdictPredictor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("VerdictPredictor loaded ← %s", path)
        return obj
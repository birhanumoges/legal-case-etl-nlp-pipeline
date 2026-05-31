"""
modeling/subtype_classifier.py
--------------------------------
Hierarchical Sub_Type classifier.

Strategy – one model per Case_Type bucket
------------------------------------------
Instead of one giant flat multi-class model, we train a separate
lightweight classifier for each Case_Type group.
This mirrors the real-world hierarchy:

    Criminal Law     → Larceny | Homicide | Assault | …
    Civil Procedure  → Demurrer | Appeal | Pleading | …
    Torts            → Negligence | Trespass | Conversion | …
    …

At inference time the predicted Case_Type (from CaseClassifier)
selects which sub-model to call.

Public API
----------
    SubtypeClassifier(model_type)
    .train(df_train)              # fits one model per Case_Type group
    .predict(df)   -> pd.Series  # uses df["Predicted_Case_Type"] or df["Case_Type"]
    .predict_full(df) -> pd.DataFrame
    .evaluate(df_test) -> dict
    .save(path)
    SubtypeClassifier.load(path)
"""

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from modeling.trainer       import Trainer
from preprocessing.features import build_feature_matrix
from preprocessing.encoder  import encode_labels

logger = logging.getLogger(__name__)

CASE_TYPE_COL = "Case_Type"
SUB_TYPE_COL  = "Sub_Type"
MIN_SAMPLES   = 10   # minimum samples needed to train a sub-model


class SubtypeClassifier:
    """
    Hierarchical sub-type classifier: one Trainer per Case_Type group.

    Parameters
    ----------
    model_type   : sklearn model name passed to Trainer.
    max_features : TF-IDF vocabulary per sub-model.
    """

    def __init__(self, model_type: str = "logistic",
                 max_features: int = 20_000, ngram_range: tuple = (1, 2)):
        self.model_type   = model_type
        self.max_features = max_features
        self.ngram_range  = ngram_range
        # dict: case_type_str -> {"trainer": Trainer, "vectorizer": ..., "classes": ...}
        self._models: Dict[str, dict] = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df_train: pd.DataFrame) -> "SubtypeClassifier":
        """
        For each Case_Type group in df_train that has enough samples,
        fit a separate sub-type classifier.
        """
        groups = df_train.groupby(CASE_TYPE_COL)
        logger.info("Training SubtypeClassifier on %d Case_Type groups …",
                    len(groups))

        for case_type, group_df in groups:
            sub_types = group_df[SUB_TYPE_COL].dropna().unique()
            if len(group_df) < MIN_SAMPLES or len(sub_types) < 2:
                logger.debug("Skipping '%s': %d rows, %d classes.",
                             case_type, len(group_df), len(sub_types))
                continue

            logger.info("  Fitting sub-model for '%s' (%d rows, %d classes) …",
                        case_type, len(group_df), len(sub_types))
            try:
                X, vec     = build_feature_matrix(group_df,
                                                   max_features=self.max_features,
                                                   ngram_range=self.ngram_range)
                y, classes = encode_labels(group_df, col=SUB_TYPE_COL)
                trainer    = Trainer(model_type=self.model_type, calibrate=True)
                trainer.fit(X, y)
                self._models[case_type] = {
                    "trainer":    trainer,
                    "vectorizer": vec,
                    "classes":    classes,
                }
            except Exception as exc:
                logger.warning("Sub-model for '%s' failed: %s", case_type, exc)

        logger.info("SubtypeClassifier: %d sub-models trained.", len(self._models))
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame,
                case_type_col: str = "Predicted_Case_Type") -> pd.Series:
        """
        Predict Sub_Type for each row.

        Uses df[case_type_col] to select the right sub-model.
        Falls back to "Unclassified" for unseen Case_Types.
        """
        # Use Predicted_Case_Type if available, else Case_Type
        if case_type_col not in df.columns:
            case_type_col = CASE_TYPE_COL

        predictions = pd.Series("Unclassified", index=df.index,
                                 name="Predicted_Sub_Type")

        for case_type, model_info in self._models.items():
            mask = df[case_type_col] == case_type
            if not mask.any():
                continue
            subset = df[mask]
            try:
                X, _ = build_feature_matrix(
                    subset,
                    vectorizer   = model_info["vectorizer"],
                    max_features = self.max_features,
                    ngram_range  = self.ngram_range,
                )
                idx    = model_info["trainer"].predict(X)
                labels = model_info["classes"][idx]
                predictions[mask] = labels
            except Exception as exc:
                logger.warning("Sub-model predict failed for '%s': %s", case_type, exc)

        return predictions

    def predict_full(self, df: pd.DataFrame,
                     case_type_col: str = "Predicted_Case_Type") -> pd.DataFrame:
        """Return predictions + confidence scores."""
        if case_type_col not in df.columns:
            case_type_col = CASE_TYPE_COL

        labels = pd.Series("Unclassified", index=df.index)
        confs  = pd.Series(0.0,            index=df.index)

        for case_type, model_info in self._models.items():
            mask = df[case_type_col] == case_type
            if not mask.any():
                continue
            subset = df[mask]
            try:
                X, _ = build_feature_matrix(subset,
                                             vectorizer   = model_info["vectorizer"],
                                             max_features = self.max_features,
                                             ngram_range  = self.ngram_range)
                proba  = model_info["trainer"].predict_proba(X)
                idx    = proba.argmax(axis=1)
                conf   = proba.max(axis=1)
                labels[mask] = model_info["classes"][idx]
                confs[mask]  = conf
            except Exception as exc:
                logger.warning("Sub-model predict_full failed for '%s': %s", case_type, exc)

        return pd.DataFrame(
            {"Predicted_Sub_Type": labels, "Sub_Type_Confidence": confs},
            index=df.index,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, df_test: pd.DataFrame) -> dict:
        """Evaluate sub-type predictions against df_test[Sub_Type]."""
        preds   = self.predict(df_test, case_type_col=CASE_TYPE_COL)
        y_true  = df_test[SUB_TYPE_COL].fillna("Unclassified").astype(str)
        y_pred  = preds.astype(str)
        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "num_sub_models": len(self._models),
        }
        logger.info("SubtypeClassifier eval — Accuracy: %.4f  F1-macro: %.4f",
                    metrics["accuracy"], metrics["f1_macro"])
        return metrics

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("SubtypeClassifier saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "SubtypeClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("SubtypeClassifier loaded ← %s", path)
        return obj
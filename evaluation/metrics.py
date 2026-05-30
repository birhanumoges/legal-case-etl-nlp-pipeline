"""
evaluation/metrics.py
----------------------
Computes every classification metric used across Case_Type,
Sub_Type, and Verdict evaluation.

Public API
----------
    compute_metrics(y_true, y_pred, classes, prefix) -> dict
    confusion_matrix_df(y_true, y_pred, classes)     -> pd.DataFrame
    per_class_report(y_true, y_pred, classes)         -> pd.DataFrame
    top_k_accuracy(y_true, y_proba, classes, k)       -> float
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    top_k_accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[np.ndarray] = None,
    prefix: str = "",
) -> dict:
    """
    Compute a full set of multi-class classification metrics.

    Parameters
    ----------
    y_true   : integer ground-truth labels.
    y_pred   : integer predicted labels.
    classes  : array of class-name strings (for log readability only).
    prefix   : string prepended to every key, e.g. "test_" or "val_".

    Returns
    -------
    dict with keys:
      {prefix}accuracy, f1_macro, f1_weighted, f1_micro,
      precision_macro, recall_macro, cohen_kappa,
      support_total, num_classes
    """
    p = prefix
    metrics = {
        f"{p}accuracy":          float(accuracy_score(y_true, y_pred)),
        f"{p}f1_macro":          float(f1_score(y_true, y_pred,
                                       average="macro",    zero_division=0)),
        f"{p}f1_weighted":       float(f1_score(y_true, y_pred,
                                       average="weighted", zero_division=0)),
        f"{p}f1_micro":          float(f1_score(y_true, y_pred,
                                       average="micro",    zero_division=0)),
        f"{p}precision_macro":   float(precision_score(y_true, y_pred,
                                       average="macro",    zero_division=0)),
        f"{p}recall_macro":      float(recall_score(y_true, y_pred,
                                       average="macro",    zero_division=0)),
        f"{p}cohen_kappa":       float(cohen_kappa_score(y_true, y_pred)),
        f"{p}support_total":     int(len(y_true)),
        f"{p}num_classes":       int(len(np.unique(y_true))),
    }
    logger.info(
        "%sAccuracy=%.4f  F1-macro=%.4f  F1-weighted=%.4f  Kappa=%.4f",
        p,
        metrics[f"{p}accuracy"],
        metrics[f"{p}f1_macro"],
        metrics[f"{p}f1_weighted"],
        metrics[f"{p}cohen_kappa"],
    )
    return metrics


def confusion_matrix_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """
    Return the confusion matrix as a labelled DataFrame.
    Rows = true labels, Columns = predicted labels.
    Values are raw counts.
    """
    cm = confusion_matrix(y_true, y_pred)
    # confusion_matrix only includes classes present in y_true/y_pred
    # Align to the full classes array
    present = sorted(set(y_true) | set(y_pred))
    labels  = classes[present] if len(classes) > max(present) else np.array(present).astype(str)
    return pd.DataFrame(cm, index=labels, columns=labels)


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> pd.DataFrame:
    """
    Return sklearn classification_report as a tidy DataFrame.
    Columns: precision, recall, f1-score, support.
    Row index includes every class plus macro/weighted/accuracy averages.
    """
    # Only pass target_names for classes that actually appear
    present     = sorted(set(y_true) | set(y_pred))
    target_names = (
        [str(classes[i]) for i in present]
        if len(classes) > max(present)
        else [str(i) for i in present]
    )
    report = classification_report(
        y_true,
        y_pred,
        labels       = present,
        target_names = target_names,
        output_dict  = True,
        zero_division= 0,
    )
    df = pd.DataFrame(report).T
    df["support"] = df["support"].astype(int)
    return df.round(4)


def top_k_accuracy(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    k: int = 3,
) -> float:
    """
    Fraction of samples where the true label appears in the top-k predictions.
    Requires probability matrix (n_samples, n_classes).
    """
    try:
        score = float(top_k_accuracy_score(y_true, y_proba, k=k))
    except Exception:
        # Fallback: manual calculation
        top_k  = np.argsort(y_proba, axis=1)[:, -k:]
        hits   = [y_true[i] in top_k[i] for i in range(len(y_true))]
        score  = float(np.mean(hits))
    logger.info("Top-%d accuracy: %.4f", k, score)
    return score
"""
visualization/confusion_matrix.py
-----------------------------------
Confusion matrix visualisation utilities used by both the evaluation
reporter and Jupyter notebooks.

Provides two views:
  1. Raw-count heatmap   (default)
  2. Row-normalised heatmap  (shows per-class recall directly)

Public API
----------
    plot_confusion_matrix(y_true, y_pred, classes,
                          title, output_dir, filename,
                          normalise, figsize)  -> Optional[Figure]

    plot_all_confusion_matrices(results_dict, output_dir)
        results_dict: {"model_name": {"y_true": ..., "y_pred": ..., "classes": ...}}
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        return plt, sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — confusion matrix skipped.")
        return None, None


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
    title: str = "Confusion Matrix",
    output_dir: Optional[str] = None,
    filename: str = "confusion_matrix.png",
    normalise: bool = False,
    figsize: Optional[tuple] = None,
) -> Optional[object]:
    """
    Plot (and optionally save) a labelled confusion matrix heatmap.

    Parameters
    ----------
    y_true     : integer ground-truth labels.
    y_pred     : integer predicted labels.
    classes    : string class names (index maps to integer label).
    title      : figure title string.
    output_dir : if provided, the figure is saved here.
    filename   : filename used when output_dir is provided.
    normalise  : if True, normalise each row to show recall per class.
    figsize    : (width, height) tuple; auto-sized if None.

    Returns
    -------
    matplotlib Figure, or None if matplotlib is unavailable.
    """
    plt, sns = _get_mpl()
    if plt is None:
        return None

    from sklearn.metrics import confusion_matrix

    # Restrict to classes actually present in y_true/y_pred
    present   = sorted(set(y_true) | set(y_pred))
    cm        = confusion_matrix(y_true, y_pred, labels=present)
    labels    = (
        [str(classes[i]) for i in present]
        if len(classes) > max(present)
        else [str(i) for i in present]
    )
    n = len(labels)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.divide(cm.astype(float), row_sums,
                             where=row_sums != 0)
        fmt      = ".2f"
        cbar_label = "Recall (row-normalised)"
    else:
        cm_plot    = cm
        fmt        = "d"
        cbar_label = "Count"

    if figsize is None:
        side    = max(6, n * 0.45)
        figsize = (side + 1, side)

    annotate = n <= 25
    fig, ax  = plt.subplots(figsize=figsize)
    sns.heatmap(
        pd.DataFrame(cm_plot, index=labels, columns=labels),
        annot     = annotate,
        fmt       = fmt,
        cmap      = "Blues",
        linewidths= 0.25,
        cbar_kws  = {"label": cbar_label},
        ax        = ax,
    )
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_ylabel("True label",      fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix → %s", path)

    return fig


def plot_all_confusion_matrices(
    results: Dict[str, dict],
    output_dir: str,
) -> None:
    """
    Generate both raw and normalised confusion matrices for every model
    in *results*.

    Parameters
    ----------
    results : dict of the form
        {
            "case_type":  {"y_true": ..., "y_pred": ..., "classes": ...},
            "sub_type":   {"y_true": ..., "y_pred": ..., "classes": ...},
            "verdict":    {"y_true": ..., "y_pred": ..., "classes": ...},
        }
    output_dir : directory where all PNG files are saved.
    """
    for model_name, data in results.items():
        y_true  = data["y_true"]
        y_pred  = data["y_pred"]
        classes = data["classes"]

        # Raw counts
        plot_confusion_matrix(
            y_true, y_pred, classes,
            title      = f"{model_name.replace('_', ' ').title()} – Confusion Matrix",
            output_dir = output_dir,
            filename   = f"{model_name}_confusion_raw.png",
            normalise  = False,
        )
        # Row-normalised (shows recall per class)
        plot_confusion_matrix(
            y_true, y_pred, classes,
            title      = f"{model_name.replace('_', ' ').title()} – Normalised",
            output_dir = output_dir,
            filename   = f"{model_name}_confusion_norm.png",
            normalise  = True,
        )
    logger.info("All confusion matrices saved → %s", output_dir)
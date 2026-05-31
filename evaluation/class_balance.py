"""
evaluation/class_balance.py
-----------------------------
Analyses class imbalance across the three prediction targets
(Case_Type, Sub_Type, Verdict_Group) and provides recommendations
for handling it.

Public API
----------
    analyse_balance(df, col, label)        -> BalanceReport
    recommend_strategy(report)             -> str
    print_balance_summary(df, cols)        -> None
    compute_class_weights(df, col)         -> dict {class_label: weight}

BalanceReport fields
--------------------
    col            – column analysed
    counts         – pd.Series  (class → count)
    frequencies    – pd.Series  (class → fraction)
    imbalance_ratio– float  (max_count / min_count)
    majority_class – str
    minority_class – str
    is_imbalanced  – bool  (ratio > 5)
    is_severely_imbalanced – bool (ratio > 20)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_IMBALANCE_THRESHOLD        = 5.0    # ratio above which data is considered imbalanced
_SEVERE_IMBALANCE_THRESHOLD = 20.0   # ratio above which it is severely imbalanced


@dataclass
class BalanceReport:
    col:                    str
    counts:                 pd.Series
    frequencies:            pd.Series
    imbalance_ratio:        float
    majority_class:         str
    minority_class:         str
    is_imbalanced:          bool
    is_severely_imbalanced: bool
    recommendations:        List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"CLASS BALANCE REPORT  ←  {self.col}",
            f"{'='*55}",
            f"  Classes            : {len(self.counts)}",
            f"  Total samples      : {self.counts.sum()}",
            f"  Majority class     : {self.majority_class} "
            f"({self.counts.max()} samples)",
            f"  Minority class     : {self.minority_class} "
            f"({self.counts.min()} samples)",
            f"  Imbalance ratio    : {self.imbalance_ratio:.1f}x",
            f"  Imbalanced         : {'⚠️  YES' if self.is_imbalanced else '✅ No'}",
            f"  Severely imbalanced: "
            f"{'🔴 YES' if self.is_severely_imbalanced else '✅ No'}",
            "",
            "  Top 10 classes:",
        ]
        for cls, cnt in self.counts.head(10).items():
            pct = self.frequencies[cls] * 100
            bar = "█" * int(pct / 2)
            lines.append(f"    {str(cls):<35} {cnt:>6}  {pct:5.1f}%  {bar}")
        if self.recommendations:
            lines.append("\n  💡 Recommendations:")
            for r in self.recommendations:
                lines.append(f"    • {r}")
        lines.append("=" * 55)
        return "\n".join(lines)


def analyse_balance(
    df: pd.DataFrame,
    col: str,
    label: Optional[str] = None,
) -> BalanceReport:
    """
    Analyse class balance of column *col* in *df*.

    Parameters
    ----------
    df    : DataFrame containing the column.
    col   : column name to analyse.
    label : optional display label (defaults to col).

    Returns
    -------
    BalanceReport dataclass.
    """
    label = label or col
    counts = df[col].fillna("Unknown").astype(str).value_counts()
    freqs  = counts / counts.sum()

    max_cnt  = int(counts.max())
    min_cnt  = int(counts.min())
    ratio    = max_cnt / max(min_cnt, 1)
    majority = str(counts.idxmax())
    minority = str(counts.idxmin())

    report = BalanceReport(
        col                    = label,
        counts                 = counts,
        frequencies            = freqs,
        imbalance_ratio        = ratio,
        majority_class         = majority,
        minority_class         = minority,
        is_imbalanced          = ratio > _IMBALANCE_THRESHOLD,
        is_severely_imbalanced = ratio > _SEVERE_IMBALANCE_THRESHOLD,
    )
    report.recommendations = recommend_strategy(report)
    return report


def recommend_strategy(report: BalanceReport) -> List[str]:
    """
    Return a list of actionable recommendations based on the balance report.
    """
    recs = []
    if not report.is_imbalanced:
        recs.append("Data is reasonably balanced — no special strategy required.")
        return recs

    recs.append(
        f"Use class_weight='balanced' in all sklearn models "
        f"(ratio={report.imbalance_ratio:.1f}x)."
    )
    if report.is_severely_imbalanced:
        recs.append(
            "Data is severely imbalanced. Consider oversampling minority "
            "classes with SMOTE (pip install imbalanced-learn) or "
            "undersampling the majority class."
        )
        recs.append(
            f"Minority class '{report.minority_class}' has only "
            f"{report.counts.min()} samples — consider merging rare classes "
            "into an 'Other' bucket."
        )
    recs.append(
        "Use F1-macro (not accuracy) as the primary evaluation metric "
        "to avoid majority-class bias."
    )
    recs.append(
        "Use stratified train/val/test splits to preserve class distributions."
    )
    return recs


def print_balance_summary(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> None:
    """
    Print balance reports for each column in *cols*.
    Defaults to Case_Type, Sub_Type, and Verdict if cols is None.
    """
    if cols is None:
        cols = [c for c in ("Case_Type", "Sub_Type", "Verdict") if c in df.columns]
    for col in cols:
        report = analyse_balance(df, col)
        print(report.summary())


def compute_class_weights(
    df: pd.DataFrame,
    col: str,
) -> Dict[str, float]:
    """
    Compute inverse-frequency class weights suitable for use with
    sklearn's sample_weight parameter or class_weight dict.

    Returns
    -------
    dict mapping class label (str) → weight (float).
    Weights are normalised so their mean equals 1.0.
    """
    counts  = df[col].fillna("Unknown").astype(str).value_counts()
    total   = counts.sum()
    n_class = len(counts)
    weights = {cls: (total / (n_class * cnt)) for cls, cnt in counts.items()}
    # Normalise so mean weight = 1.0
    mean_w  = float(np.mean(list(weights.values())))
    weights = {cls: w / mean_w for cls, w in weights.items()}
    logger.info(
        "Computed class weights for '%s': min=%.3f  max=%.3f  n_classes=%d",
        col,
        min(weights.values()),
        max(weights.values()),
        n_class,
    )
    return weights
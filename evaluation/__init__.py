"""
evaluation/__init__.py
-----------------------
Public API for the evaluation package.
"""

from .metrics       import (compute_metrics, confusion_matrix_df,
                              per_class_report, top_k_accuracy)
from .class_balance import (analyse_balance, recommend_strategy,
                              print_balance_summary, compute_class_weights,
                              BalanceReport)
from .reporter      import EvalReporter

__all__ = [
    "compute_metrics",
    "confusion_matrix_df",
    "per_class_report",
    "top_k_accuracy",
    "analyse_balance",
    "recommend_strategy",
    "print_balance_summary",
    "compute_class_weights",
    "BalanceReport",
    "EvalReporter",
]
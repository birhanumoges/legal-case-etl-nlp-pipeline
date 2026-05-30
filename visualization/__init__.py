"""
visualization/__init__.py
--------------------------
Public API for the visualization package.
"""

from .distribution_plots import (
    plot_case_type_distribution,
    plot_subtype_distribution,
    plot_verdict_distribution,
    plot_year_distribution,
    plot_court_distribution,
    plot_citations_histogram,
    plot_text_length_distribution,
    plot_all,
)
from .confusion_matrix import (
    plot_confusion_matrix,
    plot_all_confusion_matrices,
)

__all__ = [
    "plot_case_type_distribution",
    "plot_subtype_distribution",
    "plot_verdict_distribution",
    "plot_year_distribution",
    "plot_court_distribution",
    "plot_citations_histogram",
    "plot_text_length_distribution",
    "plot_all",
    "plot_confusion_matrix",
    "plot_all_confusion_matrices",
]
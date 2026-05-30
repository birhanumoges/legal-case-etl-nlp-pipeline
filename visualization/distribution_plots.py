"""
visualization/distribution_plots.py
-------------------------------------
Publication-quality distribution plots for exploratory data analysis
and post-ETL inspection.

All functions save the figure to disk AND return the figure object so
callers can embed in notebooks.  Every function degrades gracefully
(logs a warning, returns None) when matplotlib is not installed.

Public API
----------
    plot_case_type_distribution(df, output_dir, top_n)
    plot_subtype_distribution(df, output_dir, top_n)
    plot_verdict_distribution(df, output_dir, top_n)
    plot_year_distribution(df, output_dir)
    plot_court_distribution(df, output_dir, top_n)
    plot_citations_histogram(df, output_dir)
    plot_text_length_distribution(df, output_dir)
    plot_all(df, output_dir)      ← convenience wrapper
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _get_plt():
    """Import matplotlib with Agg backend. Returns (plt, sns) or (None, None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
        return plt, sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — plot skipped.")
        return None, None


def _save(fig, output_dir: str, filename: str) -> Optional[str]:
    """Save figure and return its path. Returns None on any error."""
    try:
        plt, _ = _get_plt()
        path   = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Plot saved → %s", path)
        return path
    except Exception as exc:
        logger.error("Failed to save plot %s: %s", filename, exc)
        return None


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_case_type_distribution(
    df: pd.DataFrame,
    output_dir: str,
    top_n: int = 10,
) -> Optional[object]:
    """Horizontal bar chart of top-N Case_Type values."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    counts = df["Case_Type"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[0])
    ax.set_title(f"Case Type Distribution (top {top_n})", fontsize=13)
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("")
    for bar in ax.patches:
        ax.text(bar.get_width() + counts.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width()):,}", va="center", fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "case_type_distribution.png")
    return fig


def plot_subtype_distribution(
    df: pd.DataFrame,
    output_dir: str,
    top_n: int = 15,
) -> Optional[object]:
    """Horizontal bar chart of top-N Sub_Type values."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    counts = df["Sub_Type"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[2])
    ax.set_title(f"Sub-Type Distribution (top {top_n})", fontsize=13)
    ax.set_xlabel("Number of cases")
    plt.tight_layout()
    _save(fig, output_dir, "subtype_distribution.png")
    return fig


def plot_verdict_distribution(
    df: pd.DataFrame,
    output_dir: str,
    top_n: int = 12,
) -> Optional[object]:
    """Horizontal bar chart of top-N Verdict values."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    counts = df["Verdict"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[3])
    ax.set_title(f"Verdict Distribution (top {top_n})", fontsize=13)
    ax.set_xlabel("Number of cases")
    plt.tight_layout()
    _save(fig, output_dir, "verdict_distribution.png")
    return fig


def plot_year_distribution(
    df: pd.DataFrame,
    output_dir: str,
) -> Optional[object]:
    """Line chart showing number of cases per year."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    years  = pd.to_numeric(df["Year"], errors="coerce").dropna()
    years  = years[(years >= 1700) & (years <= 2100)]
    counts = years.astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(counts.index, counts.values,
                    alpha=0.4, color=sns.color_palette("muted")[0])
    ax.plot(counts.index, counts.values,
            color=sns.color_palette("muted")[0], linewidth=1.5)
    ax.set_title("Cases Per Year", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of cases")
    plt.tight_layout()
    _save(fig, output_dir, "year_distribution.png")
    return fig


def plot_court_distribution(
    df: pd.DataFrame,
    output_dir: str,
    top_n: int = 10,
) -> Optional[object]:
    """Horizontal bar chart of top-N courts."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    counts = df["Court"].value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("muted")[4])
    ax.set_title(f"Top {top_n} Courts", fontsize=13)
    ax.set_xlabel("Number of cases")
    plt.tight_layout()
    _save(fig, output_dir, "court_distribution.png")
    return fig


def plot_citations_histogram(
    df: pd.DataFrame,
    output_dir: str,
) -> Optional[object]:
    """Histogram of citations per case."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    cites = pd.to_numeric(df.get("Num_Citations", 0), errors="coerce").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(cites, bins=40, color=sns.color_palette("muted")[1], edgecolor="white")
    ax.set_title("Citations Per Case", fontsize=13)
    ax.set_xlabel("Number of citations")
    ax.set_ylabel("Cases")
    ax.axvline(cites.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {cites.mean():.1f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "citations_histogram.png")
    return fig


def plot_text_length_distribution(
    df: pd.DataFrame,
    output_dir: str,
) -> Optional[object]:
    """Histogram of case text length (characters)."""
    plt, sns = _get_plt()
    if plt is None:
        return None

    lengths = df["Case_Text"].fillna("").str.len()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths, bins=50, color=sns.color_palette("muted")[5], edgecolor="white")
    ax.set_title("Case Text Length Distribution", fontsize=13)
    ax.set_xlabel("Characters")
    ax.set_ylabel("Cases")
    ax.axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {lengths.mean():.0f}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "text_length_distribution.png")
    return fig


def plot_all(df: pd.DataFrame, output_dir: str) -> None:
    """
    Convenience wrapper — generate every distribution plot in one call.
    All files are saved to *output_dir*.
    """
    logger.info("Generating all distribution plots → %s", output_dir)
    plot_case_type_distribution(df, output_dir)
    plot_subtype_distribution(df, output_dir)
    plot_verdict_distribution(df, output_dir)
    plot_year_distribution(df, output_dir)
    plot_court_distribution(df, output_dir)
    plot_citations_histogram(df, output_dir)
    plot_text_length_distribution(df, output_dir)
    logger.info("All distribution plots saved.")
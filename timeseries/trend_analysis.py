"""time_series/trend_analysis.py — Case trends over time."""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import PLOT_DIR, REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class TrendAnalyzer:
    """
    Build time-indexed aggregate tables and trend charts.
    Features used: Year_Numeric, Case_Type_Mapped, Sub_Type_Mapped,
                   Verdict_Mapped, Court (ordinal encoded), Num_Citations.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df[df["Year_Numeric"].notna()].copy() if "Year_Numeric" in df.columns else df.copy()
        if "Year_Numeric" in self.df.columns:
            self.df["Year_Numeric"] = self.df["Year_Numeric"].astype(int)

    def _save(self, fig, name: str):
        p = PLOT_DIR / name
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Plot → %s", p)
        return p

    # ── Case type trends ──────────────────────────────────────────────
    def case_type_trends(self) -> pd.DataFrame:
        if "Case_Type_Mapped" not in self.df.columns:
            return pd.DataFrame()
        pivot = (
            self.df.groupby(["Year_Numeric", "Case_Type_Mapped"])
            .size().unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot.plot(ax=ax, linewidth=1.5)
        ax.set_title("Case Type Trends Over Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Number of Cases")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        self._save(fig, "ts_case_type_trends.png")
        pivot.to_csv(REPORT_DIR / "ts_case_type_trends.csv")
        return pivot

    # ── Subtype trends ────────────────────────────────────────────────
    def subtype_trends(self, top_n: int = 10) -> pd.DataFrame:
        if "Sub_Type_Mapped" not in self.df.columns:
            return pd.DataFrame()
        top = self.df["Sub_Type_Mapped"].value_counts().head(top_n).index
        sub = self.df[self.df["Sub_Type_Mapped"].isin(top)]
        pivot = sub.groupby(["Year_Numeric", "Sub_Type_Mapped"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 6))
        pivot.plot(ax=ax, linewidth=1.2, alpha=0.85)
        ax.set_title(f"Top {top_n} Sub-Type Trends", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Cases")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        self._save(fig, "ts_subtype_trends.png")
        pivot.to_csv(REPORT_DIR / "ts_subtype_trends.csv")
        return pivot

    # ── Verdict trends ────────────────────────────────────────────────
    def verdict_trends(self) -> pd.DataFrame:
        if "Verdict_Mapped" not in self.df.columns:
            return pd.DataFrame()
        pivot = self.df.groupby(["Year_Numeric", "Verdict_Mapped"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(14, 5))
        pivot.plot(ax=ax, linewidth=1.5)
        ax.set_title("Verdict Trends Over Time", fontsize=13, fontweight="bold")
        ax.set_xlabel("Year"); ax.set_ylabel("Cases")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        self._save(fig, "ts_verdict_trends.png")
        pivot.to_csv(REPORT_DIR / "ts_verdict_trends.csv")
        return pivot

    # ── Citation growth ───────────────────────────────────────────────
    def citation_growth(self) -> pd.DataFrame:
        if "Num_Citations" not in self.df.columns:
            return pd.DataFrame()
        annual = (
            self.df.groupby("Year_Numeric")["Num_Citations"]
            .agg(["mean", "sum", "count"])
            .rename(columns={"mean": "avg_citations", "sum": "total_citations", "count": "n_cases"})
        )
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        annual["avg_citations"].plot(ax=axes[0], color="steelblue", linewidth=1.5)
        axes[0].set_title("Average Citations per Case Over Time", fontsize=12)
        axes[0].set_ylabel("Avg Citations")

        annual["total_citations"].plot(ax=axes[1], color="coral", linewidth=1.5)
        axes[1].set_title("Total Citations per Year", fontsize=12)
        axes[1].set_ylabel("Total Citations")
        axes[1].set_xlabel("Year")
        plt.tight_layout()
        self._save(fig, "ts_citation_growth.png")
        annual.to_csv(REPORT_DIR / "ts_citation_growth.csv")
        return annual

    def run_all(self):
        logger.info("Running all trend analyses …")
        self.case_type_trends()
        self.subtype_trends()
        self.verdict_trends()
        self.citation_growth()
        logger.info("Trend analysis complete.")

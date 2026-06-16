"""time_series/yearly_statistics.py — Per-year summary statistics."""

from __future__ import annotations
import pandas as pd
from config import REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class YearlyStatistics:
    def __init__(self, df: pd.DataFrame):
        self.df = df[df["Year_Numeric"].notna()].copy() if "Year_Numeric" in df.columns else df.copy()

    def compute(self) -> pd.DataFrame:
        if "Year_Numeric" not in self.df.columns:
            return pd.DataFrame()
        grp = self.df.groupby("Year_Numeric")
        stats = grp.agg(
            n_cases=("Case_ID", "count"),
            avg_citations=("Num_Citations", "mean"),
            total_citations=("Num_Citations", "sum"),
        ).round(2)

        # Most common case type and verdict per year
        if "Case_Type_Mapped" in self.df.columns:
            stats["top_case_type"] = grp["Case_Type_Mapped"].agg(lambda x: x.mode()[0] if len(x) > 0 else "")
        if "Verdict_Mapped" in self.df.columns:
            stats["top_verdict"] = grp["Verdict_Mapped"].agg(lambda x: x.mode()[0] if len(x) > 0 else "")

        path = REPORT_DIR / "ts_yearly_statistics.csv"
        stats.to_csv(path)
        logger.info("Yearly statistics saved → %s", path)
        return stats

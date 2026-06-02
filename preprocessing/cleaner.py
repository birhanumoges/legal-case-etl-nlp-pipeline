"""preprocessing/cleaner.py — DataFrame-level cleaning and class mapping."""

from __future__ import annotations
import pandas as pd
import numpy as np
from config import CASE_TYPE_MAP, SUBTYPE_MAP, VERDICT_MAP, TARGETS
from utils.logger import get_logger

logger = get_logger(__name__)

UNKNOWN_VERDICT  = "Verdict Unknown"
UNKNOWN_CASETYPE = "Unclassified"


class DataCleaner:
    """
    Apply hierarchical class merging, split clean vs unknown rows,
    fix year column, and return both DataFrames.
    """

    # ── public API ───────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (clean_df, unknown_df).
        clean_df  : rows where Case_Type, Sub_Type, AND Verdict are all known.
        unknown_df: rows where at least one target is unknown/unclassified.
        """
        df = df.copy()
        df = self._fix_year(df)
        df = self._apply_mappings(df)
        df = self._drop_empty_text(df)

        unknown_mask = (
            (df["Case_Type_Mapped"] == UNKNOWN_CASETYPE)
            | (df["Verdict_Mapped"]  == "OTHER")     # OTHER = unmapped verdicts
            | df["Case_Text"].str.strip().eq("")
        )

        unknown_df = df[unknown_mask].copy()
        clean_df   = df[~unknown_mask].copy()

        logger.info("Clean rows: %d | Unknown/unclassified rows: %d", len(clean_df), len(unknown_df))
        return clean_df, unknown_df

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _fix_year(df: pd.DataFrame) -> pd.DataFrame:
        """Replace missing / non-numeric year strings with 'Unknown'."""
        if "Year" not in df.columns:
            df["Year"] = "Unknown"
            return df

        def _norm(v):
            s = str(v).strip()
            if not s or s.lower() in {"nan", "none", "unknown", "#########"}:
                return "Unknown"
            import re
            m = re.search(r"(1[6-9]\d{2}|20[0-2]\d)", s)
            return m.group(1) if m else "Unknown"

        df["Year"] = df["Year"].apply(_norm)
        return df

    @staticmethod
    def _apply_mappings(df: pd.DataFrame) -> pd.DataFrame:
        # Case_Type → high-level mapped
        df["Case_Type_Mapped"] = df["Case_Type"].map(CASE_TYPE_MAP).fillna(UNKNOWN_CASETYPE)
        # Sub_Type → hierarchical mapped
        df["Sub_Type_Mapped"]  = df["Sub_Type"].map(SUBTYPE_MAP).fillna("UNCLASSIFIED__General")
        # Verdict → grouped
        df["Verdict_Mapped"]   = df["Verdict"].map(VERDICT_MAP).fillna("OTHER")
        return df

    @staticmethod
    def _drop_empty_text(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[df["Case_Text"].notna() & (df["Case_Text"].str.strip() != "")]
        logger.info("Dropped %d rows with empty text", before - len(df))
        return df

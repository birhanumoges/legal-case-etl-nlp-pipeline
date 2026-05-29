"""
pipeline/validator.py
---------------------
Validates the ETL output DataFrame before it is saved or passed
to the preprocessing stage.

Checks
------
1. Required columns are present.
2. No fully-empty Case_Text rows.
3. Duplicate Case_IDs.
4. Verdict / Case_Type / Sub_Type coverage rates.
5. Year range sanity (1600–2100).

Public API
----------
    validate(df) -> ValidationResult
    ValidationResult.ok       – bool
    ValidationResult.report() – prints + returns summary string
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List
from utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLS = [
    "Case_ID", "Year", "Court", "Case_Text",
    "Verdict", "Legal_Citations", "Case_Type", "Sub_Type",
]


@dataclass
class ValidationResult:
    ok:       bool         = True
    errors:   List[str]    = field(default_factory=list)
    warnings: List[str]    = field(default_factory=list)
    stats:    dict         = field(default_factory=dict)

    def report(self) -> str:
        lines = ["", "=" * 60, "VALIDATION REPORT", "=" * 60]
        lines.append(f"Status : {'✅ PASSED' if self.ok else '❌ FAILED'}")
        lines.append(f"Rows   : {self.stats.get('total_rows', '?')}")
        if self.errors:
            lines.append("\n❌ ERRORS:")
            for e in self.errors:
                lines.append(f"  • {e}")
        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for w in self.warnings:
                lines.append(f"  • {w}")
        lines.append("\n📊 COVERAGE:")
        for k, v in self.stats.items():
            if k != "total_rows":
                lines.append(f"  {k:<35} {v}")
        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text


def validate(df: pd.DataFrame) -> ValidationResult:
    """Run all validation checks on the ETL output DataFrame."""
    result = ValidationResult()
    result.stats["total_rows"] = len(df)

    # 1. Required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        result.ok = False
        result.errors.append(f"Missing required columns: {missing}")

    if df.empty:
        result.ok = False
        result.errors.append("DataFrame is empty — ETL produced no records.")
        return result

    # 2. Empty Case_Text
    empty_text = df["Case_Text"].isna() | (df["Case_Text"].str.strip() == "")
    n_empty = empty_text.sum()
    if n_empty > 0:
        result.warnings.append(f"{n_empty} rows have empty Case_Text.")

    # 3. Duplicate Case_IDs
    if "Case_ID" in df.columns:
        dupes = df["Case_ID"].duplicated().sum()
        if dupes:
            result.warnings.append(f"{dupes} duplicate Case_IDs found.")

    # 4. Coverage rates
    n = len(df)
    for col, bad_val in [
        ("Verdict",   "Verdict Unknown"),
        ("Case_Type", "Unclassified"),
        ("Sub_Type",  "Unclassified"),
    ]:
        if col in df.columns:
            known  = (df[col] != bad_val).sum()
            pct    = known / n * 100
            label  = f"{col} coverage"
            result.stats[label] = f"{known}/{n}  ({pct:.1f}%)"
            if pct < 70:
                result.warnings.append(f"{col} coverage below 70% ({pct:.1f}%).")

    # 5. Year range
    if "Year" in df.columns:
        years = pd.to_numeric(df["Year"], errors="coerce")
        bad_years = years[(years < 1600) | (years > 2100)].dropna()
        if len(bad_years):
            result.warnings.append(
                f"{len(bad_years)} rows have suspicious year values."
            )
        result.stats["year_range"] = (
            f"{years.min():.0f} – {years.max():.0f}"
            if not years.isna().all() else "unknown"
        )

    logger.info(
        "Validation %s — %d errors, %d warnings",
        "PASSED" if result.ok else "FAILED",
        len(result.errors), len(result.warnings),
    )
    return result
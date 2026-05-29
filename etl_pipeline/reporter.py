"""
pipeline/reporter.py
---------------------
Save ETL output and generate the summary report.
"""

import os
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


def save_outputs(df: pd.DataFrame, output_dir: str) -> None:
    """Save DataFrame as CSV, JSON, and Parquet."""
    os.makedirs(output_dir, exist_ok=True)

    csv = os.path.join(output_dir, "legal_cases_complete.csv")
    df.to_csv(csv, index=False, encoding="utf-8")
    logger.info("  ✅ CSV     → %s", csv)

    jsn = os.path.join(output_dir, "legal_cases_complete.json")
    df.to_json(jsn, orient="records", indent=2, force_ascii=False)
    logger.info("  ✅ JSON    → %s", jsn)

    try:
        pq = os.path.join(output_dir, "legal_cases_complete.parquet")
        df.to_parquet(pq, index=False)
        logger.info("  ✅ Parquet → %s", pq)
    except Exception as exc:
        logger.warning("  ⚠️  Parquet skipped: %s", exc)


def generate_summary_report(df: pd.DataFrame, output_dir: str) -> None:
    """Print and save a text summary report."""
    lines = _build(df)
    text  = "\n".join(lines)
    print("\n" + text)
    path = os.path.join(output_dir, "etl_summary_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("  📄 Report  → %s", path)


def _build(df: pd.DataFrame) -> list:
    sep = "=" * 70
    n   = len(df)
    lines = [sep, "LEGAL NLP ETL PIPELINE — SUMMARY REPORT", sep,
             f"\nGenerated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"Total cases: {n}"]

    if "Source_Folder" in df.columns:
        lines.append("\n📁 SOURCE DISTRIBUTION:")
        for src, cnt in df["Source_Folder"].value_counts().items():
            lines.append(f"  - {src}: {cnt} ({cnt/n*100:.1f}%)")

    lines.append("\n📚 CASE TYPE:")
    for ct, cnt in df["Case_Type"].value_counts().head(10).items():
        lines.append(f"  {ct}: {cnt} ({cnt/n*100:.1f}%)")

    lines.append("\n🔍 SUB-TYPE:")
    for st, cnt in df["Sub_Type"].value_counts().head(15).items():
        lines.append(f"  {st}: {cnt} ({cnt/n*100:.1f}%)")

    lines.append("\n⚖️  VERDICT:")
    for v, cnt in df["Verdict"].value_counts().head(10).items():
        lines.append(f"  {v}: {cnt} ({cnt/n*100:.1f}%)")

    if "Year" in df.columns:
        yr = pd.to_numeric(df["Year"], errors="coerce")
        lines.append(f"\n📅 YEAR: {yr.min():.0f} – {yr.max():.0f}")

    if "Num_Citations" in df.columns:
        lines.append(f"📊 CITATIONS: total={df['Num_Citations'].sum()}  "
                     f"avg={df['Num_Citations'].mean():.2f}/case")

    kv = (df["Verdict"]   != "Verdict Unknown").sum()
    kc = (df["Case_Type"] != "Unclassified").sum()
    ks = (df["Sub_Type"]  != "Unclassified").sum()
    lines += ["\n✅ QUALITY:", f"  Known verdicts : {kv}/{n} ({kv/n*100:.1f}%)",
              f"  Classified     : {kc}/{n} ({kc/n*100:.1f}%)",
              f"  Specific sub   : {ks}/{n} ({ks/n*100:.1f}%)", sep]
    return lines
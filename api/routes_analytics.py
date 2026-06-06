"""api/routes_analytics.py — Analytics, statistics, forecasts, and model reports."""

from __future__ import annotations
import json
import math
import pandas as pd
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import (
    StatsResponse, DistributionItem, YearlyStatsResponse, YearlyStatItem,
    ForecastResponse, ForecastItem, ModelReportResponse, ModelComparisonItem,
)
from api.dependencies import get_optional_user
from config import OUTPUT_DIR, REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    p = Path(OUTPUT_DIR) / "clean_data.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def _dist(series: pd.Series, top_n: int = 10) -> list[DistributionItem]:
    vc    = series.value_counts().head(top_n)
    total = series.notna().sum()
    return [
        DistributionItem(
            label   = str(lbl),
            count   = int(cnt),
            percent = round(cnt / total * 100, 2) if total else 0,
        )
        for lbl, cnt in vc.items()
    ]


# ── Overview stats ────────────────────────────────────────────────
@router.get("/stats", response_model=StatsResponse, summary="High-level corpus statistics")
def get_stats(_user = Depends(get_optional_user)):
    df = _df()

    # also read unknown file for total
    unk_path = Path(OUTPUT_DIR) / "unknown_case_data.csv"
    unk_count = 0
    if unk_path.exists():
        try:
            unk_count = len(pd.read_csv(unk_path, usecols=["Case_ID"], low_memory=False))
        except Exception:
            pass

    clean = len(df)
    total = clean + unk_count

    avg_cite = float(df["Num_Citations"].mean()) if "Num_Citations" in df.columns and not df.empty else 0.0

    year_range: dict = {}
    if "Year_Numeric" in df.columns:
        yn = pd.to_numeric(df["Year_Numeric"], errors="coerce").dropna()
        if not yn.empty:
            year_range = {"min": int(yn.min()), "max": int(yn.max())}

    return StatsResponse(
        total_cases    = total,
        clean_cases    = clean,
        unknown_cases  = unk_count,
        case_type_dist = _dist(df["Case_Type_Mapped"]) if "Case_Type_Mapped" in df.columns else [],
        verdict_dist   = _dist(df["Verdict_Mapped"])   if "Verdict_Mapped"   in df.columns else [],
        top_courts     = _dist(df["Court"], top_n=8)   if "Court"            in df.columns else [],
        avg_citations  = round(avg_cite, 2),
        year_range     = year_range,
    )


# ── Yearly statistics ─────────────────────────────────────────────
@router.get("/yearly", response_model=YearlyStatsResponse,
            summary="Per-year case volume and citation statistics")
def yearly_stats(_user = Depends(get_optional_user)):
    p = Path(REPORT_DIR) / "ts_yearly_statistics.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Yearly statistics not yet generated. Run main.py first.")

    df = pd.read_csv(p)
    items = []
    for _, row in df.iterrows():
        try:
            items.append(YearlyStatItem(
                year            = int(row.get("Year_Numeric", 0)),
                n_cases         = int(row.get("n_cases", 0)),
                avg_citations   = float(row.get("avg_citations", 0)),
                total_citations = int(row.get("total_citations", 0)),
                top_case_type   = str(row["top_case_type"]) if "top_case_type" in row and pd.notna(row.get("top_case_type")) else None,
                top_verdict     = str(row["top_verdict"])   if "top_verdict"   in row and pd.notna(row.get("top_verdict"))   else None,
            ))
        except Exception:
            continue

    return YearlyStatsResponse(items=sorted(items, key=lambda x: x.year))


# ── Forecast ──────────────────────────────────────────────────────
@router.get("/forecast", response_model=ForecastResponse,
            summary="ARIMA case-volume forecast")
def forecast(_user = Depends(get_optional_user)):
    p = Path(REPORT_DIR) / "ts_forecast_summary.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Forecast not yet generated. Run main.py first.")

    with open(p) as f:
        data = json.load(f)

    arima = data.get("arima") or {}
    if not arima:
        raise HTTPException(status_code=404, detail="ARIMA forecast unavailable.")

    fc_items = [
        ForecastItem(year=int(yr), forecast=round(float(val), 1))
        for yr, val in (arima.get("forecast") or {}).items()
    ]
    return ForecastResponse(
        method   = arima.get("method", "ARIMA"),
        order    = arima.get("order"),
        aic      = arima.get("aic"),
        forecast = sorted(fc_items, key=lambda x: x.year),
    )


# ── Case type distribution endpoint ──────────────────────────────
@router.get("/distribution/case-type", response_model=list[DistributionItem],
            summary="Case type distribution")
def case_type_dist(_user = Depends(get_optional_user)):
    df = _df()
    if "Case_Type_Mapped" not in df.columns:
        return []
    return _dist(df["Case_Type_Mapped"], top_n=10)


@router.get("/distribution/verdict", response_model=list[DistributionItem],
            summary="Verdict distribution")
def verdict_dist(_user = Depends(get_optional_user)):
    df = _df()
    if "Verdict_Mapped" not in df.columns:
        return []
    return _dist(df["Verdict_Mapped"], top_n=10)


@router.get("/distribution/sub-type", response_model=list[DistributionItem],
            summary="Sub-type distribution (top 21)")
def subtype_dist(_user = Depends(get_optional_user)):
    df = _df()
    if "Sub_Type_Mapped" not in df.columns:
        return []
    return _dist(df["Sub_Type_Mapped"], top_n=21)


# ── Model reports ─────────────────────────────────────────────────
@router.get("/models", response_model=list[ModelReportResponse],
            summary="All three model performance reports")
def model_reports(_user = Depends(get_optional_user)):
    reports = []
    for target in ["Case_Type_Mapped", "Sub_Type_Mapped", "Verdict_Mapped"]:
        p = Path(REPORT_DIR) / f"{target}_best_model_report.json"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            comparison = [
                ModelComparisonItem(**item)
                for item in d.get("all_model_comparison", [])
            ]
            reports.append(ModelReportResponse(
                target                 = d.get("target", target),
                best_model             = d.get("best_model", ""),
                test_accuracy          = d.get("test_accuracy", 0),
                test_macro_f1          = d.get("test_macro_f1", 0),
                test_weighted_f1       = d.get("test_weighted_f1", 0),
                test_precision         = d.get("test_precision", 0),
                test_recall            = d.get("test_recall", 0),
                classification_report  = d.get("classification_report", ""),
                all_model_comparison   = comparison,
            ))
        except Exception as exc:
            logger.warning("Could not load report %s: %s", p, exc)
    return reports


@router.get("/models/{target}", response_model=ModelReportResponse,
            summary="Single target model report")
def model_report(target: str, _user = Depends(get_optional_user)):
    # normalise target name
    target_map = {
        "case_type": "Case_Type_Mapped",
        "sub_type":  "Sub_Type_Mapped",
        "verdict":   "Verdict_Mapped",
    }
    key = target_map.get(target.lower(), target)
    p   = Path(REPORT_DIR) / f"{key}_best_model_report.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Report for '{target}' not found. Run main.py first.")
    with open(p) as f:
        d = json.load(f)
    comparison = [ModelComparisonItem(**item) for item in d.get("all_model_comparison", [])]
    return ModelReportResponse(
        target                = d.get("target", key),
        best_model            = d.get("best_model", ""),
        test_accuracy         = d.get("test_accuracy", 0),
        test_macro_f1         = d.get("test_macro_f1", 0),
        test_weighted_f1      = d.get("test_weighted_f1", 0),
        test_precision        = d.get("test_precision", 0),
        test_recall           = d.get("test_recall", 0),
        classification_report = d.get("classification_report", ""),
        all_model_comparison  = comparison,
    )

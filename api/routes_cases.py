"""api/routes_cases.py — Case browsing, search, and detail endpoints."""

from __future__ import annotations
import math
import pandas as pd
from pathlib import Path
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Query, Depends
from api.schemas import (
    CaseBase, CaseDetail, CaseListResponse, CaseSearchRequest,
    SimilarCasesResponse, SimilarCaseItem,
)
from api.dependencies import get_optional_user
from config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cases", tags=["Cases"])


# ── lazy-load CSV once ────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    path = Path(OUTPUT_DIR) / "clean_data.csv"
    if not path.exists():
        logger.warning("clean_data.csv not found at %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    logger.info("Cases loaded: %d rows", len(df))
    return df


def _row_to_base(row: pd.Series) -> CaseBase:
    def _s(v):
        return str(v) if pd.notna(v) else None
    def _i(v):
        try: return int(v)
        except Exception: return None
    return CaseBase(
        case_id          = _s(row.get("Case_ID", "")),
        case_name        = _s(row.get("Case_Name")),
        year             = _s(row.get("Year")),
        court            = _s(row.get("Court")),
        case_type        = _s(row.get("Case_Type")),
        sub_type         = _s(row.get("Sub_Type")),
        verdict          = _s(row.get("Verdict")),
        case_type_mapped = _s(row.get("Case_Type_Mapped")),
        sub_type_mapped  = _s(row.get("Sub_Type_Mapped")),
        verdict_mapped   = _s(row.get("Verdict_Mapped")),
        num_citations    = _i(row.get("Num_Citations")),
        text_length      = _i(row.get("text_length")),
        word_count       = _i(row.get("word_count")),
    )


def _row_to_detail(row: pd.Series) -> CaseDetail:
    base = _row_to_base(row)
    def _s(v):
        return str(v) if pd.notna(v) else None
    def _f(v):
        try: return float(v)
        except Exception: return None
    return CaseDetail(
        **base.dict(),
        case_text       = _s(row.get("Case_Text")),
        legal_citations = _s(row.get("Legal_Citations")),
        source_folder   = _s(row.get("Source_Folder")),
        year_numeric    = _f(row.get("Year_Numeric")),
    )


# ── endpoints ─────────────────────────────────────────────────────
@router.get("", response_model=CaseListResponse, summary="List cases with pagination")
def list_cases(
    page:      int          = Query(1,  ge=1),
    size:      int          = Query(20, ge=1, le=100),
    case_type: str | None   = Query(None),
    verdict:   str | None   = Query(None),
    court:     str | None   = Query(None),
    year_from: int | None   = Query(None),
    year_to:   int | None   = Query(None),
    _user = Depends(get_optional_user),
):
    df = _load_df()
    if df.empty:
        return CaseListResponse(total=0, page=page, size=size, pages=0, items=[])

    mask = pd.Series([True] * len(df), index=df.index)
    if case_type and "Case_Type_Mapped" in df.columns:
        mask &= df["Case_Type_Mapped"].str.upper() == case_type.upper()
    if verdict and "Verdict_Mapped" in df.columns:
        mask &= df["Verdict_Mapped"].str.upper() == verdict.upper()
    if court and "Court" in df.columns:
        mask &= df["Court"].str.contains(court, case=False, na=False)
    if year_from and "Year_Numeric" in df.columns:
        mask &= pd.to_numeric(df["Year_Numeric"], errors="coerce").fillna(0) >= year_from
    if year_to and "Year_Numeric" in df.columns:
        mask &= pd.to_numeric(df["Year_Numeric"], errors="coerce").fillna(9999) <= year_to

    filtered = df[mask]
    total    = len(filtered)
    pages    = math.ceil(total / size) if size else 1
    offset   = (page - 1) * size
    chunk    = filtered.iloc[offset: offset + size]

    items = [_row_to_base(row) for _, row in chunk.iterrows()]
    return CaseListResponse(total=total, page=page, size=size, pages=pages, items=items)


@router.post("/search", response_model=CaseListResponse, summary="Full-text search cases")
def search_cases(req: CaseSearchRequest, _user = Depends(get_optional_user)):
    df = _load_df()
    if df.empty:
        return CaseListResponse(total=0, page=req.page, size=req.size, pages=0, items=[])

    mask = pd.Series([True] * len(df), index=df.index)

    if req.query and "Case_Text" in df.columns:
        mask &= df["Case_Text"].str.contains(req.query, case=False, na=False)
    if req.case_type and "Case_Type_Mapped" in df.columns:
        mask &= df["Case_Type_Mapped"].str.upper() == req.case_type.upper()
    if req.verdict and "Verdict_Mapped" in df.columns:
        mask &= df["Verdict_Mapped"].str.upper() == req.verdict.upper()
    if req.court and "Court" in df.columns:
        mask &= df["Court"].str.contains(req.court, case=False, na=False)
    if req.year_from and "Year_Numeric" in df.columns:
        mask &= pd.to_numeric(df["Year_Numeric"], errors="coerce").fillna(0) >= req.year_from
    if req.year_to and "Year_Numeric" in df.columns:
        mask &= pd.to_numeric(df["Year_Numeric"], errors="coerce").fillna(9999) <= req.year_to

    filtered = df[mask]
    total    = len(filtered)
    pages    = math.ceil(total / req.size) if req.size else 1
    offset   = (req.page - 1) * req.size
    chunk    = filtered.iloc[offset: offset + req.size]

    items = [_row_to_base(row) for _, row in chunk.iterrows()]
    return CaseListResponse(total=total, page=req.page, size=req.size,
                            pages=pages, items=items)


@router.get("/{case_id}", response_model=CaseDetail, summary="Get full case detail")
def get_case(case_id: str, _user = Depends(get_optional_user)):
    df = _load_df()
    if df.empty or "Case_ID" not in df.columns:
        raise HTTPException(status_code=404, detail="Case not found")

    rows = df[df["Case_ID"].astype(str) == case_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    return _row_to_detail(rows.iloc[0])


@router.get("/{case_id}/similar", response_model=SimilarCasesResponse,
            summary="Find semantically similar cases via FAISS")
def similar_cases(case_id: str, top_k: int = Query(5, ge=1, le=20),
                  _user = Depends(get_optional_user)):
    """Return the top-k most similar cases to the given case_id using
    the FAISS vector index built during pipeline execution."""
    df = _load_df()
    if df.empty or "Case_ID" not in df.columns:
        raise HTTPException(status_code=404, detail="Dataset not available")

    rows = df[df["Case_ID"].astype(str) == case_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    query_text = str(rows.iloc[0].get("Case_Text", ""))[:500]

    try:
        from vectorstore.similarity_search import SimilaritySearch
        ss = SimilaritySearch()
        ss.store.load()
        ss.index.load()
        raw = ss.search(query_text, top_k=top_k + 1)
        # exclude self
        results = [
            SimilarCaseItem(id=r["id"], score=r["score"],
                            text=r["text"], metadata=r["metadata"])
            for r in raw if r["id"].split("_chunk_")[0] != case_id
        ][:top_k]
    except Exception as exc:
        logger.warning("FAISS search failed: %s", exc)
        results = []

    return SimilarCasesResponse(query=query_text[:100], results=results)

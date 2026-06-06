"""api/schemas.py — All Pydantic request / response schemas.
Existing: PredictRequest, PredictResponse, RAGQueryRequest,
          RAGQueryResponse, HealthResponse  — kept exactly as-is.
New additions below.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── Already-existing schemas (unchanged) ─────────────────────────
class PredictRequest(BaseModel):
    case_text:     str           = Field(..., min_length=10)
    court:         Optional[str] = "Unknown Court"
    num_citations: Optional[int] = 0


class PredictResponse(BaseModel):
    case_type:  str
    sub_type:   str
    verdict:    str
    confidence: Optional[float] = None


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k:    int = Field(default=5, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    question: str
    answer:   str
    sources:  List[str]


class HealthResponse(BaseModel):
    status:  str
    version: str = "1.0.0"


# ── NEW: Auth ─────────────────────────────────────────────────────
class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = 3600

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


# ── NEW: Cases ────────────────────────────────────────────────────
class CaseBase(BaseModel):
    case_id:           str
    case_name:         Optional[str]  = None
    year:              Optional[str]  = None
    court:             Optional[str]  = None
    case_type:         Optional[str]  = None
    sub_type:          Optional[str]  = None
    verdict:           Optional[str]  = None
    case_type_mapped:  Optional[str]  = None
    sub_type_mapped:   Optional[str]  = None
    verdict_mapped:    Optional[str]  = None
    num_citations:     Optional[int]  = 0
    text_length:       Optional[int]  = None
    word_count:        Optional[int]  = None

class CaseDetail(CaseBase):
    case_text:       Optional[str]   = None
    legal_citations: Optional[str]   = None
    source_folder:   Optional[str]   = None
    year_numeric:    Optional[float] = None

class CaseListResponse(BaseModel):
    total: int
    page:  int
    size:  int
    pages: int
    items: List[CaseBase]

class CaseSearchRequest(BaseModel):
    query:     Optional[str] = None
    case_type: Optional[str] = None
    verdict:   Optional[str] = None
    court:     Optional[str] = None
    year_from: Optional[int] = None
    year_to:   Optional[int] = None
    page:      int           = Field(default=1, ge=1)
    size:      int           = Field(default=20, ge=1, le=100)


# ── NEW: Analytics ────────────────────────────────────────────────
class DistributionItem(BaseModel):
    label:   str
    count:   int
    percent: float

class StatsResponse(BaseModel):
    total_cases:    int
    clean_cases:    int
    unknown_cases:  int
    case_type_dist: List[DistributionItem]
    verdict_dist:   List[DistributionItem]
    top_courts:     List[DistributionItem]
    avg_citations:  float
    year_range:     Dict[str, Any]

class YearlyStatItem(BaseModel):
    year:            int
    n_cases:         int
    avg_citations:   float
    total_citations: int
    top_case_type:   Optional[str] = None
    top_verdict:     Optional[str] = None

class YearlyStatsResponse(BaseModel):
    items: List[YearlyStatItem]

class ForecastItem(BaseModel):
    year:     int
    forecast: float

class ForecastResponse(BaseModel):
    method:   str
    order:    Optional[List[int]] = None
    aic:      Optional[float]     = None
    forecast: List[ForecastItem]


# ── NEW: Model reports ────────────────────────────────────────────
class ModelComparisonItem(BaseModel):
    model:        str
    target:       str
    val_macro_f1: float
    val_accuracy: float
    train_time_s: float

class ModelReportResponse(BaseModel):
    target:               str
    best_model:           str
    test_accuracy:        float
    test_macro_f1:        float
    test_weighted_f1:     float
    test_precision:       float
    test_recall:          float
    classification_report: str
    all_model_comparison: List[ModelComparisonItem]


# ── NEW: Batch prediction ─────────────────────────────────────────
class BatchPredictItem(BaseModel):
    id:            str
    case_text:     str  = Field(..., min_length=10)
    court:         Optional[str] = "Unknown Court"
    num_citations: Optional[int] = 0

class BatchPredictRequest(BaseModel):
    items: List[BatchPredictItem]

class BatchPredictResultItem(BaseModel):
    id:        str
    case_type: str
    sub_type:  str
    verdict:   str
    error:     Optional[str] = None

class BatchPredictResponse(BaseModel):
    total:   int
    results: List[BatchPredictResultItem]


# ── NEW: Similar cases ────────────────────────────────────────────
class SimilarCaseItem(BaseModel):
    id:       str
    score:    float
    text:     str
    metadata: Dict[str, Any]

class SimilarCasesResponse(BaseModel):
    query:   str
    results: List[SimilarCaseItem]


# ── NEW: Error ────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    detail:    str
    code:      Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

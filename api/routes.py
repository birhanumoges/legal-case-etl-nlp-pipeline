"""api/routes.py — FastAPI route definitions."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from api.schemas import (
    PredictRequest, PredictResponse,
    RAGQueryRequest, RAGQueryResponse,
    HealthResponse,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

# These are populated by app.py on startup
_classifiers: dict = {}
_rag_engine  = None


def set_classifiers(classifiers: dict):
    global _classifiers
    _classifiers = classifiers


def set_rag_engine(engine):
    global _rag_engine
    _rag_engine = engine


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run case-type, sub-type, and verdict prediction."""
    try:
        import pandas as pd
        row = pd.DataFrame([{
            "Case_Text":     req.case_text,
            "Court":         req.court,
            "Num_Citations": req.num_citations,
        }])
        results = {}
        for target, (encoder, model) in _classifiers.items():
            X   = encoder.transform(row)
            pred = model.predict(X)[0]
            results[target] = encoder.decode_labels(target, [pred])[0]

        return PredictResponse(
            case_type=results.get("Case_Type_Mapped", "UNKNOWN"),
            sub_type =results.get("Sub_Type_Mapped",  "UNKNOWN"),
            verdict  =results.get("Verdict_Mapped",   "UNKNOWN"),
        )
    except Exception as e:
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(req: RAGQueryRequest):
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialised")
    result = _rag_engine.ask_full(req.question, top_k=req.top_k)
    return RAGQueryResponse(
        question=result["question"],
        answer  =result["answer"],
        sources =result["sources"],
    )

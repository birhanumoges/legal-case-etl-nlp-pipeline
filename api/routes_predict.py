"""api/routes_predict.py — Extended prediction: batch + case enrichment.
NOTE: Single /predict and /rag/query endpoints already exist in routes.py.
      This module adds batch prediction and enrichment without duplication.
"""

from __future__ import annotations
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import (
    BatchPredictRequest, BatchPredictResponse, BatchPredictResultItem,
)
from api.dependencies import get_current_user
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# Classifiers injected at startup by app.py
_classifiers: dict = {}


def set_batch_classifiers(classifiers: dict):
    global _classifiers
    _classifiers = classifiers


def _predict_one(case_text: str, court: str, num_citations: int) -> dict:
    row = pd.DataFrame([{
        "Case_Text":     case_text,
        "Court":         court,
        "Num_Citations": num_citations,
    }])
    results = {}
    for target, (encoder, model) in _classifiers.items():
        X    = encoder.transform(row)
        pred = model.predict(X)[0]
        results[target] = encoder.decode_labels(target, [pred])[0]
    return results


@router.post("/batch", response_model=BatchPredictResponse,
             summary="Predict case type, sub-type, and verdict for multiple cases at once")
def batch_predict(req: BatchPredictRequest, user: str = Depends(get_current_user)):
    """
    Accepts up to 50 case texts and returns predictions for all three targets.
    Requires authentication.
    """
    if not _classifiers:
        raise HTTPException(status_code=503, detail="Models not loaded. Run main.py first.")

    results = []
    for item in req.items:
        try:
            preds = _predict_one(
                item.case_text,
                item.court or "Unknown Court",
                item.num_citations or 0,
            )
            results.append(BatchPredictResultItem(
                id        = item.id,
                case_type = preds.get("Case_Type_Mapped", "UNKNOWN"),
                sub_type  = preds.get("Sub_Type_Mapped",  "UNKNOWN"),
                verdict   = preds.get("Verdict_Mapped",   "UNKNOWN"),
            ))
        except Exception as exc:
            results.append(BatchPredictResultItem(
                id        = item.id,
                case_type = "ERROR",
                sub_type  = "ERROR",
                verdict   = "ERROR",
                error     = str(exc),
            ))

    logger.info("Batch prediction: %d items by user '%s'", len(results), user)
    return BatchPredictResponse(total=len(results), results=results)

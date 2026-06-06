"""api/app.py — FastAPI application entry point.
Registers all routers. Existing startup logic (model loading, RAG engine)
is kept exactly as-is; new routers added below it.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# existing router
from api.routes import router, set_classifiers, set_rag_engine

# new routers
from api.routes_auth      import router as auth_router
from api.routes_cases     import router as cases_router
from api.routes_analytics import router as analytics_router
from api.routes_predict   import router as predict_router, set_batch_classifiers
from api.middleware.logging_middleware import LoggingMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title       = "Legal NLP API",
    description = (
        "End-to-end legal case classification, verdict prediction, "
        "RAG querying, analytics, and semantic search."
    ),
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── Middleware ────────────────────────────────────────────────────
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Routers ───────────────────────────────────────────────────────
app.include_router(router,           prefix="/api/v1")           # existing: health, predict, rag/query
app.include_router(auth_router,      prefix="/api/v1")           # new: auth/login
app.include_router(cases_router,     prefix="/api/v1")           # new: cases CRUD
app.include_router(analytics_router, prefix="/api/v1")           # new: analytics
app.include_router(predict_router,   prefix="/api/v1")           # new: batch predict

# ── Serve built frontend if present ──────────────────────────────
_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load saved models and RAG engine on server startup."""
    logger.info("API startup — loading models …")
    try:
        import joblib
        from config import MODEL_DIR
        from preprocessing.encoder import FeatureEncoder

        encoder    = FeatureEncoder.load(MODEL_DIR / "feature_encoder.joblib")
        classifiers = {}
        for target_file in MODEL_DIR.glob("*.joblib"):
            if "feature_encoder" in target_file.name:
                continue
            name = target_file.stem
            if   name.startswith("case_type"): key = "Case_Type_Mapped"
            elif name.startswith("sub_type"):  key = "Sub_Type_Mapped"
            elif name.startswith("verdict"):   key = "Verdict_Mapped"
            else: continue
            model = joblib.load(target_file)
            classifiers[key] = (encoder, model)
            logger.info("  Loaded model: %s → %s", target_file.name, key)

        set_classifiers(classifiers)
        set_batch_classifiers(classifiers)
    except Exception as e:
        logger.warning("Could not load classifiers: %s", e)

    try:
        from rag.query_engine import QueryEngine
        engine = QueryEngine(llm=None)
        set_rag_engine(engine)
        logger.info("RAG engine initialised.")
    except Exception as e:
        logger.warning("Could not initialise RAG engine: %s", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

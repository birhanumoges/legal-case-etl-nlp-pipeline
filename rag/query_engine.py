"""rag/query_engine.py — User-facing query interface."""

from __future__ import annotations
from rag.rag_pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryEngine:
    def __init__(self, llm=None):
        self.pipeline = RAGPipeline(llm=llm)

    def ask(self, question: str, top_k: int = 5) -> str:
        result = self.pipeline.query(question, top_k=top_k)
        logger.info("Query: %s | Sources: %s", question[:60], result["sources"])
        return result["answer"]

    def ask_full(self, question: str, top_k: int = 5) -> dict:
        return self.pipeline.query(question, top_k=top_k)

"""rag/rag_pipeline.py — End-to-end RAG flow."""

from __future__ import annotations
from rag.retriever import LegalRetriever
from rag.generator import LegalGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    def __init__(self, llm=None):
        self.retriever = LegalRetriever()
        self.generator = LegalGenerator()
    def query(self, question: str, top_k: int = 5) -> dict:
        docs     = self.retriever.retrieve(question, top_k=top_k)
        context  = self.retriever.format_context(docs)
        answer   = self.generator.generate(question, context)
        return {
            "question":  question,
            "answer":    answer,
            "sources":   [d["id"] for d in docs],
            "retrieved": docs,
        }

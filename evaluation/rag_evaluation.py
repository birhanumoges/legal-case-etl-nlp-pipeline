"""evaluation/rag_evaluation.py — RAG quality evaluation (faithfulness, relevance)."""

from __future__ import annotations
import json
from pathlib import Path
from config import REPORT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """
    Lightweight RAG evaluator using string-overlap heuristics.
    For production use, replace with RAGAS or TruLens.
    """

    def evaluate(self, queries: list[str], retrieved_docs: list[list[str]],
                 generated_answers: list[str]) -> dict:
        scores = []
        for q, docs, ans in zip(queries, retrieved_docs, generated_answers):
            scores.append(self._score_one(q, docs, ans))

        avg = {
            "avg_context_precision": round(sum(s["context_precision"] for s in scores) / len(scores), 3),
            "avg_answer_relevance":  round(sum(s["answer_relevance"]  for s in scores) / len(scores), 3),
            "n_queries": len(scores),
        }
        path = REPORT_DIR / "rag_evaluation.json"
        with open(path, "w") as f:
            json.dump({"summary": avg, "per_query": scores}, f, indent=2)
        logger.info("RAG evaluation saved → %s", path)
        return avg

    @staticmethod
    def _score_one(query: str, docs: list[str], answer: str) -> dict:
        q_words   = set(query.lower().split())
        doc_text  = " ".join(docs).lower()
        doc_words = set(doc_text.split())
        ctx_prec  = len(q_words & doc_words) / (len(q_words) + 1e-9)

        ans_words = set(answer.lower().split())
        ans_rel   = len(ans_words & doc_words) / (len(ans_words) + 1e-9)
        return {
            "query":             query,
            "context_precision": min(round(float(ctx_prec), 3), 1.0),
            "answer_relevance":  min(round(float(ans_rel),  3), 1.0),
        }

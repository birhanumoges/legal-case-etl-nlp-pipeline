"""tests/test_rag.py — Tests for RAG retriever, generator, and pipeline."""

import pytest
from unittest.mock import MagicMock, patch

from rag.prompt_templates import LEGAL_QA_TEMPLATE
from rag.generator        import LegalGenerator
from rag.retriever        import LegalRetriever
from rag.rag_pipeline     import RAGPipeline
from rag.query_engine     import QueryEngine
from evaluation.rag_evaluation import RAGEvaluator


# ── Prompt templates ──────────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_legal_qa_has_placeholders(self):
        assert "{context}"  in LEGAL_QA_TEMPLATE
        assert "{question}" in LEGAL_QA_TEMPLATE

    def test_legal_qa_fills_correctly(self):
        filled = LEGAL_QA_TEMPLATE.format(
            context="Case A: plaintiff won.", question="Who won?"
        )
        assert "Case A" in filled
        assert "Who won?" in filled


# ── LegalGenerator ────────────────────────────────────────────────────────────

class TestLegalGenerator:
    def test_no_llm_returns_stub(self):
        gen    = LegalGenerator(llm=None)
        answer = gen.generate("Who won?", "The plaintiff was awarded damages.")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_custom_llm_called(self):
        mock_llm = MagicMock(return_value="The plaintiff won.")
        gen      = LegalGenerator(llm=mock_llm)
        answer   = gen.generate("Who won?", "Context here.")
        mock_llm.assert_called_once()
        assert answer == "The plaintiff won."


# ── LegalRetriever ────────────────────────────────────────────────────────────

class TestLegalRetriever:
    def test_format_context_with_docs(self):
        retriever = LegalRetriever()
        docs = [
            {"id": "1001", "score": 0.95,
             "text": "Plaintiff brought suit for breach of contract.",
             "metadata": {"Case_Type": "CONTRACT", "Verdict": "AFFIRMED"}},
            {"id": "1002", "score": 0.88,
             "text": "Criminal larceny case dismissed on appeal.",
             "metadata": {"Case_Type": "CRIMINAL", "Verdict": "DENIED"}},
        ]
        ctx = retriever.format_context(docs)
        assert "1001" in ctx
        assert "CONTRACT" in ctx
        assert "AFFIRMED" in ctx
        assert "1002" in ctx

    def test_format_context_empty(self):
        retriever = LegalRetriever()
        assert retriever.format_context([]) == ""


# ── RAGPipeline ───────────────────────────────────────────────────────────────

class TestRAGPipeline:
    def test_query_returns_dict_keys(self):
        # Patch the retriever so no real vector store is needed
        pipeline = RAGPipeline(llm=None)
        pipeline.retriever.search = MagicMock(return_value=[
            {"id": "X1", "score": 0.9,
             "text": "Judgment affirmed for plaintiff.",
             "metadata": {"Case_Type": "CONTRACT"}}
        ])
        result = pipeline.query("Was the plaintiff successful?", top_k=1)
        assert "question"  in result
        assert "answer"    in result
        assert "sources"   in result
        assert "retrieved" in result
        assert result["question"] == "Was the plaintiff successful?"

    def test_sources_list(self):
        pipeline = RAGPipeline(llm=None)
        pipeline.retriever.search = MagicMock(return_value=[
            {"id": "A1", "score": 0.85, "text": "Text A.", "metadata": {}},
            {"id": "A2", "score": 0.80, "text": "Text B.", "metadata": {}},
        ])
        result = pipeline.query("test", top_k=2)
        assert "A1" in result["sources"]
        assert "A2" in result["sources"]


# ── QueryEngine ───────────────────────────────────────────────────────────────

class TestQueryEngine:
    def test_ask_returns_string(self):
        engine = QueryEngine(llm=None)
        engine.pipeline.retriever.search = MagicMock(return_value=[])
        answer = engine.ask("What is contract law?")
        assert isinstance(answer, str)

    def test_ask_full_returns_dict(self):
        engine = QueryEngine(llm=None)
        engine.pipeline.retriever.search = MagicMock(return_value=[])
        result = engine.ask_full("Define larceny.")
        assert isinstance(result, dict)
        assert "answer" in result


# ── RAGEvaluator ─────────────────────────────────────────────────────────────

class TestRAGEvaluator:
    def test_evaluate_returns_summary(self, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        evaluator = RAGEvaluator()
        result    = evaluator.evaluate(
            queries   =["Who won the contract case?", "What was the verdict?"],
            retrieved_docs=[
                ["plaintiff won contract dispute damages awarded"],
                ["judgment affirmed appeal denied"],
            ],
            generated_answers=[
                "The plaintiff won the contract case.",
                "The verdict was affirmed on appeal.",
            ],
        )
        assert "avg_context_precision" in result
        assert "avg_answer_relevance"  in result
        assert result["n_queries"] == 2
        assert (tmp_path / "rag_evaluation.json").exists()

    def test_scores_in_range(self, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "REPORT_DIR", tmp_path)

        evaluator = RAGEvaluator()
        result    = evaluator.evaluate(
            queries=["contract breach judgment"],
            retrieved_docs=[["contract breach plaintiff judgment awarded"]],
            generated_answers=["The plaintiff was awarded damages for contract breach."],
        )
        assert 0.0 <= result["avg_context_precision"] <= 1.0
        assert 0.0 <= result["avg_answer_relevance"]  <= 1.0

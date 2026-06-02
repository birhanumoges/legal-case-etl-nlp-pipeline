"""tests/test_vectorstore.py — Tests for vectorstore components."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from vectorstore.faiss_index       import FaissIndex
from vectorstore.embeddings        import EmbeddingStore
from vectorstore.similarity_search import SimilaritySearch


# ── FaissIndex ────────────────────────────────────────────────────────────────

class TestFaissIndex:
    def test_build_and_search_with_faiss(self, tmp_path, monkeypatch):
        """Skip gracefully if faiss is not installed."""
        pytest.importorskip("faiss")
        import config as cfg
        monkeypatch.setattr(cfg, "VECTORSTORE_DIR", tmp_path)

        # Rebuild module-level constant
        import vectorstore.faiss_index as fi_mod
        monkeypatch.setattr(fi_mod, "INDEX_PATH", tmp_path / "faiss.index")

        vecs  = np.random.rand(20, 32).astype(np.float32)
        # Normalise so inner product ≈ cosine
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        idx = FaissIndex()
        idx.build(vecs)
        assert idx._index is not None
        assert idx._index.ntotal == 20

        query  = vecs[0]
        scores, idxs = idx.search(query, top_k=3)
        assert len(scores) == 3
        assert len(idxs)   == 3
        assert idxs[0] == 0          # top match should be itself

    def test_search_without_build_returns_empty(self):
        idx = FaissIndex()
        scores, idxs = idx.search(np.zeros(32), top_k=3)
        assert scores == []
        assert idxs   == []


# ── EmbeddingStore ────────────────────────────────────────────────────────────

class TestEmbeddingStore:
    def test_build_mocked(self, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "VECTORSTORE_DIR", tmp_path)

        mock_vecs = np.random.rand(5, 64).astype(np.float32)
        store = EmbeddingStore()
        store.preparer.encode = MagicMock(return_value=mock_vecs)

        texts = ["text one", "text two", "text three", "text four", "text five"]
        ids   = ["id1", "id2", "id3", "id4", "id5"]
        vecs  = store.build(texts, ids)

        assert vecs is not None
        assert vecs.shape == (5, 64)
        assert (tmp_path / "vectors.npy").exists()
        assert (tmp_path / "ids.npy").exists()

    def test_load_after_build(self, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "VECTORSTORE_DIR", tmp_path)

        mock_vecs = np.random.rand(4, 32).astype(np.float32)
        store = EmbeddingStore()
        store.preparer.encode = MagicMock(return_value=mock_vecs)
        store.build(["a", "b", "c", "d"], ["1", "2", "3", "4"])

        store2 = EmbeddingStore()
        loaded = store2.load()
        assert loaded is not None
        assert loaded.shape == (4, 32)


# ── SimilaritySearch ──────────────────────────────────────────────────────────

class TestSimilaritySearch:
    def test_search_returns_results(self, tmp_path, monkeypatch):
        import config as cfg
        monkeypatch.setattr(cfg, "VECTORSTORE_DIR", tmp_path)

        mock_vecs = np.random.rand(10, 32).astype(np.float32)
        mock_vecs /= np.linalg.norm(mock_vecs, axis=1, keepdims=True)

        ss = SimilaritySearch()
        ss.store.preparer.encode = MagicMock(return_value=mock_vecs)

        texts    = [f"legal text about case {i}" for i in range(10)]
        ids      = [f"case_{i:03d}" for i in range(10)]
        metadata = [{"Case_Type": "CIVIL"} for _ in range(10)]

        # Mock faiss so test runs without faiss installed
        ss.index.build  = MagicMock()
        ss.index.search = MagicMock(return_value=([0.9, 0.85, 0.80], [0, 2, 5]))

        ss.build(texts, ids, metadata)

        query_enc = np.random.rand(1, 32).astype(np.float32)
        ss.store.preparer.encode = MagicMock(return_value=query_enc)

        results = ss.search("contract breach judgment", top_k=3)
        # With mocked index.search returning idxs [0,2,5]
        assert len(results) == 3
        assert results[0]["id"] == "case_000"
        assert results[1]["id"] == "case_002"
        assert results[2]["id"] == "case_005"
        assert all("score" in r for r in results)

    def test_search_no_results_on_empty(self):
        ss = SimilaritySearch()
        ss.store.preparer.encode = MagicMock(return_value=None)
        results = ss.search("anything")
        assert results == []

"""vectorstore/similarity_search.py — High-level similarity search interface."""

from __future__ import annotations
from typing import List, Dict
import numpy as np
from vectorstore.faiss_index import FaissIndex
from vectorstore.embeddings  import EmbeddingStore
from utils.logger import get_logger

logger = get_logger(__name__)


class SimilaritySearch:
    def __init__(self):
        self.store = EmbeddingStore()
        self.index = FaissIndex()

    def build(self, texts: List[str], ids: List[str], metadata: List[Dict] | None = None):
        vecs = self.store.build(texts, ids)
        if vecs is not None:
            self.index.build(vecs)
        self._metadata = metadata or [{}] * len(texts)
        self._ids       = ids
        self._texts     = texts

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        q_vec = self.store.preparer.encode([query])
        if q_vec is None:
            return []
        scores, idxs = self.index.search(q_vec[0], top_k)
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0:
                continue
            results.append({
                "id":       self._ids[idx],
                "score":    round(float(score), 4),
                "text":     self._texts[idx][:300],
                "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
            })
        return results

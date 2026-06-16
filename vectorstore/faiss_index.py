"""vectorstore/faiss_index.py — FAISS index build / load / search."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from config import VECTORSTORE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"


class FaissIndex:
    def __init__(self):
        self._index = None
        self.dim: int | None = None

    def build(self, vectors: np.ndarray):
        try:
            import faiss
        except ImportError:
            logger.warning("faiss-cpu not installed; vector index unavailable.")
            return
        self.dim = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self.dim)   # Inner product (cosine if normalised)
        self._index.add(vectors.astype(np.float32))
        faiss.write_index(self._index, str(INDEX_PATH))
        logger.info("FAISS index built (%d vectors, dim=%d) → %s",
                    vectors.shape[0], self.dim, INDEX_PATH)

    def load(self):
        try:
            import faiss
            if INDEX_PATH.exists():
                self._index = faiss.read_index(str(INDEX_PATH))
                self.dim    = self._index.d
                logger.info("FAISS index loaded (%d vectors)", self._index.ntotal)
        except ImportError:
            logger.warning("faiss-cpu not installed.")

    def search(self, query_vec: np.ndarray, top_k: int = 5):
        if self._index is None:
            return [], []
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, idxs = self._index.search(q, top_k)
        return scores[0].tolist(), idxs[0].tolist()

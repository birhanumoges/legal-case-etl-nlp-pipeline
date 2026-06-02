"""preprocessing/embedding_preparer.py — Prepare text chunks for vector DB ingestion."""

from __future__ import annotations
from typing import List, Dict
import numpy as np
from config import EMBEDDING_MODEL, VECTORSTORE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingPreparer:
    """
    Wraps sentence-transformers to encode text chunks.
    Falls back gracefully if torch/transformers not installed.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info("Loaded embedding model: %s", self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed; embeddings unavailable.")
        return self._model

    def encode(self, texts: List[str], batch_size: int = 64,
               show_progress: bool = True) -> np.ndarray | None:
        model = self._load()
        if model is None:
            return None
        vectors = model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors

    def encode_chunks(self, chunks: List[Dict],
                      batch_size: int = 64) -> List[Dict]:
        """Adds 'embedding' key to each chunk dict."""
        texts = [c["text"] for c in chunks]
        vecs  = self.encode(texts, batch_size=batch_size)
        if vecs is None:
            for c in chunks:
                c["embedding"] = None
        else:
            for c, v in zip(chunks, vecs):
                c["embedding"] = v.tolist()
        return chunks

    def save_embeddings(self, vectors: np.ndarray, filename: str = "embeddings.npy"):
        path = VECTORSTORE_DIR / filename
        np.save(path, vectors)
        logger.info("Embeddings saved → %s  shape=%s", path, vectors.shape)
        return path

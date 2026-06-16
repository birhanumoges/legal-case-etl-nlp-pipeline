"""vectorstore/embeddings.py — Generate and persist embeddings."""

from __future__ import annotations
import numpy as np
from pathlib import Path
from config import VECTORSTORE_DIR
from preprocessing.embedding_preparer import EmbeddingPreparer
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingStore:
    def __init__(self, model_name: str | None = None):
        self.preparer = EmbeddingPreparer(model_name) if model_name else EmbeddingPreparer()
        self.vectors: np.ndarray | None = None
        self.ids: list[str] = []

    def build(self, texts: list[str], ids: list[str]) -> np.ndarray | None:
        logger.info("Building embeddings for %d texts …", len(texts))
        self.vectors = self.preparer.encode(texts)
        self.ids     = ids
        if self.vectors is not None:
            np.save(VECTORSTORE_DIR / "vectors.npy",  self.vectors)
            np.save(VECTORSTORE_DIR / "ids.npy",       np.array(ids))
            logger.info("Embeddings saved. Shape: %s", self.vectors.shape)
        return self.vectors

    def load(self):
        vp = VECTORSTORE_DIR / "vectors.npy"
        ip = VECTORSTORE_DIR / "ids.npy"
        if vp.exists() and ip.exists():
            self.vectors = np.load(vp)
            self.ids     = np.load(ip).tolist()
            logger.info("Loaded embeddings %s", self.vectors.shape)
        return self.vectors

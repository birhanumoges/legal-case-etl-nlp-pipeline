"""preprocessing/encoder.py — TF-IDF + label encoding for ML pipeline."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from scipy.sparse import hstack, csr_matrix
import joblib
from pathlib import Path
from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEncoder:
    """
    Encodes:
      - Case_Text  → TF-IDF sparse matrix
      - Court      → OrdinalEncoder (dense column)
      - Num_Citations → passthrough scalar
    Combines into a single feature matrix.
    Target columns are LabelEncoded separately.
    """

    def __init__(self, max_features: int = TFIDF_MAX_FEATURES,
                 ngram_range: tuple = TFIDF_NGRAM_RANGE):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2,
        )
        self.court_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    # ── fit / transform ──────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        logger.info("Fitting TF-IDF on %d rows …", len(df))
        tfidf_mat = self.tfidf.fit_transform(df["Case_Text"].fillna(""))
        court_mat = self._fit_court(df)
        cite_mat  = self._citations_matrix(df)
        self._fitted = True
        return hstack([tfidf_mat, court_mat, cite_mat], format="csr")

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        assert self._fitted, "Call fit_transform first"
        tfidf_mat = self.tfidf.transform(df["Case_Text"].fillna(""))
        court_mat = self._transform_court(df)
        cite_mat  = self._citations_matrix(df)
        return hstack([tfidf_mat, court_mat, cite_mat], format="csr")

    def fit_labels(self, df: pd.DataFrame, target_cols: list[str]) -> dict[str, np.ndarray]:
        encoded = {}
        for col in target_cols:
            le = LabelEncoder()
            encoded[col] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
            self.label_encoders[col] = le
            logger.info("Label encoder '%s': %d classes", col, len(le.classes_))
        return encoded

    def encode_labels(self, df: pd.DataFrame, col: str) -> np.ndarray:
        le = self.label_encoders[col]
        return le.transform(df[col].astype(str).fillna("UNKNOWN"))

    def decode_labels(self, col: str, indices: np.ndarray) -> np.ndarray:
        return self.label_encoders[col].inverse_transform(indices)

    # ── persistence ──────────────────────────────────────────────────
    def save(self, path: Path | str | None = None):
        path = Path(path or MODEL_DIR / "feature_encoder.joblib")
        joblib.dump(self, path)
        logger.info("FeatureEncoder saved → %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "FeatureEncoder":
        return joblib.load(path)

    # ── private ──────────────────────────────────────────────────────
    def _fit_court(self, df: pd.DataFrame) -> csr_matrix:
        vals = df[["Court"]].fillna("unknown")
        mat = self.court_enc.fit_transform(vals)
        return csr_matrix(mat)

    def _transform_court(self, df: pd.DataFrame) -> csr_matrix:
        vals = df[["Court"]].fillna("unknown")
        mat = self.court_enc.transform(vals)
        return csr_matrix(mat)

    @staticmethod
    def _citations_matrix(df: pd.DataFrame) -> csr_matrix:
        col = "Num_Citations" if "Num_Citations" in df.columns else None
        if col:
            arr = df[col].fillna(0).values.reshape(-1, 1).astype(float)
        else:
            arr = np.zeros((len(df), 1))
        return csr_matrix(arr)

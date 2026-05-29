"""
preprocessing/features.py
--------------------------
Builds numeric feature matrices for the three ML targets:
  - Case_Type   (top-level classifier)
  - Sub_Type    (hierarchical sub-type classifier)
  - Verdict     (verdict predictor)

Public API
----------
    build_tfidf(df, text_col, max_features, ngram_range) -> sparse matrix + vectorizer
    build_legal_flags(df) -> pd.DataFrame of binary flag columns
    build_feature_matrix(df, vectorizer=None) -> (X_sparse, vectorizer)
    LEGAL_FLAG_COLS   – list of hand-crafted binary column names
"""

import re
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── Hand-crafted binary flag patterns ────────────────────────────────────────
_FLAG_PATTERNS = {
    "flag_per_curiam":      r"per curiam",
    "flag_dissent":         r"dissent(?:ing)?",
    "flag_has_amount":      r"\$[\d,]+(?:\.\d{2})?",
    "flag_criminal":        r"indictment|guilty|felony|larceny|homicide",
    "flag_contract":        r"promissory note|breach of contract|consideration",
    "flag_property":        r"ejectment|mortgage|foreclosure|adverse possession",
    "flag_appeal":          r"writ of error|on appeal|appealed from",
    "flag_negligence":      r"negligence|contributory|proximate cause",
    "flag_injunction":      r"injunction|enjoined",
    "flag_award":           r"award (?:granted|denied|made|refused)",
    "flag_affirmed":        r"\baff?irm(?:ed|ance)?\b",
    "flag_reversed":        r"\brevers(?:ed|al)?\b",
}
_FLAG_RE    = {n: re.compile(p, re.IGNORECASE) for n, p in _FLAG_PATTERNS.items()}
LEGAL_FLAG_COLS = list(_FLAG_PATTERNS.keys())


def build_tfidf(
    df: pd.DataFrame,
    text_col: str = "Case_Text",
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Fit (or reuse) a TF-IDF vectoriser and return the sparse matrix.

    Parameters
    ----------
    vectorizer : if supplied, transform only (no fit). Pass the
                 training-set vectorizer when encoding val/test sets.

    Returns
    -------
    (X_sparse, fitted_vectorizer)
    """
    texts = df[text_col].fillna("").astype(str)
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = ngram_range,
            sublinear_tf = True,
            strip_accents = "unicode",
            min_df       = 2,
            token_pattern= r"[\w']+",
        )
        X = vectorizer.fit_transform(texts)
        logger.info("TF-IDF fit: vocab=%d, shape=%s", len(vectorizer.vocabulary_), X.shape)
    else:
        X = vectorizer.transform(texts)
        logger.info("TF-IDF transform: shape=%s", X.shape)
    return X, vectorizer


def build_legal_flags(df: pd.DataFrame, text_col: str = "Case_Text") -> pd.DataFrame:
    """
    Add binary (0/1) legal-flag columns to a copy of df.
    These columns are named flag_* and listed in LEGAL_FLAG_COLS.
    """
    out   = df.copy()
    texts = df[text_col].fillna("").astype(str)
    for col, pattern in _FLAG_RE.items():
        out[col] = texts.apply(lambda t: int(bool(pattern.search(t))))
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    text_col: str = "Case_Text",
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Build a combined feature matrix:
      TF-IDF sparse matrix  +  hand-crafted binary flags (stacked)
      
    Returns
    -------
    (X_combined_sparse, fitted_vectorizer)
    """
    X_tfidf, vec = build_tfidf(df, text_col, max_features, ngram_range, vectorizer)

    flag_df  = build_legal_flags(df, text_col)
    flag_mat = csr_matrix(flag_df[LEGAL_FLAG_COLS].values.astype(np.float32))

    X = hstack([X_tfidf, flag_mat], format="csr")
    logger.info("Combined feature matrix: shape=%s (tfidf + %d flags)",
                X.shape, len(LEGAL_FLAG_COLS))
    return X, vec
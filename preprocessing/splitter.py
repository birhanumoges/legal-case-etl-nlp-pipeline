"""preprocessing/splitter.py — Stratified train/val/test splitting."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE
from utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """Stratified split → train / val / test."""

    def __init__(self, test_size: float = TEST_SIZE, val_size: float = VAL_SIZE,
                 random_state: int = RANDOM_STATE):
        self.test_size    = test_size
        self.val_size     = val_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame, stratify_col: str
              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train_df, val_df, test_df)."""
        y = df[stratify_col].astype(str)

        # Remove classes with < 2 samples (can't stratify)
        counts = y.value_counts()
        valid  = counts[counts >= 2].index
        df_ok  = df[y.isin(valid)]
        y_ok   = df_ok[stratify_col].astype(str)

        if len(df_ok) < len(df):
            logger.warning("Dropped %d rows with singleton classes for stratification",
                           len(df) - len(df_ok))

        train_val, test = train_test_split(
            df_ok, test_size=self.test_size, random_state=self.random_state,
            stratify=y_ok
        )
        val_ratio = self.val_size / (1 - self.test_size)
        train, val = train_test_split(
            train_val, test_size=val_ratio, random_state=self.random_state,
            stratify=train_val[stratify_col].astype(str)
        )
        logger.info("Split (%s): train=%d  val=%d  test=%d",
                    stratify_col, len(train), len(val), len(test))
        return train, val, test

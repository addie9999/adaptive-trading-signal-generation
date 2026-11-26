"""Orchestrates the cleaning pipeline on raw OHLCV frames.

Order of operations matters and is deliberate:

1. de-duplicate the index,
2. ensure a sorted, regular business-day-ish index,
3. forward-fill gaps (prices persist; volume gaps are zero-filled),
4. detect and drop outliers on *returns* (not raw prices),
5. optionally fit/apply a scaler for downstream feature columns.

The fitted scaler is retained on the instance so the exact same transform can
be re-applied to live data at serving time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.cleaning.outliers import IQROutlierDetector, IsolationForestOutlierDetector
from src.data.base import OHLCV_COLUMNS
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Configurable cleaning pipeline for a single instrument's OHLCV frame."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.cleaning
        self._scaler = None  # fitted lazily in `scale`

    # ------------------------------------------------------------------ #
    def clean(self, frame: pd.DataFrame, *, keep_outliers: bool = False) -> pd.DataFrame:
        """Run the structural cleaning steps (no scaling)."""
        df = frame.copy()
        df = self._dedupe(df)
        df = self._sort_index(df)
        df = self._fill_missing(df)
        if not keep_outliers:
            df = self._drop_outliers(df)
        logger.info("Cleaning complete: %d rows retained", len(df))
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        if before != len(df):
            logger.info("Removed %d duplicate index rows", before - len(df))
        return df

    @staticmethod
    def _sort_index(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index()

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cols = [c for c in OHLCV_COLUMNS if c != "volume" and c in df.columns]
        if self.cfg.fill_method == "ffill":
            df[price_cols] = df[price_cols].ffill()
        # Leading NaNs that ffill cannot reach are dropped; volume gaps -> 0.
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0.0)
        df = df.dropna(subset=price_cols)
        return df

    def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.cfg.outlier_method
        if method == "none":
            return df

        # Detect on returns to avoid flagging legitimate price-level trends.
        returns = df["close"].pct_change().to_frame("ret").dropna()
        if returns.empty:
            return df

        if method == "iqr":
            detector = IQROutlierDetector(k=self.cfg.iqr_k, columns=["ret"])
        elif method == "isolation_forest":
            detector = IsolationForestOutlierDetector(
                contamination=self.cfg.isolation_contamination, columns=["ret"]
            )
        else:
            raise ValueError(f"Unknown outlier_method '{method}'")

        mask = detector.inlier_mask(returns)
        outliers = (~mask).sum()
        if outliers:
            logger.info("Flagged %d return outliers via %s", int(outliers), method)
        keep_idx = returns.index[mask]
        # Always keep the first bar (no return defined for it).
        keep = df.index.isin(keep_idx) | (df.index == df.index[0])
        return df[keep]

    # ------------------------------------------------------------------ #
    def scale(self, frame: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric feature columns with the configured scaler.

        Parameters
        ----------
        fit:
            When ``True`` (training), fit a fresh scaler and store it. When
            ``False`` (serving), reuse the previously fitted scaler.
        """
        if not self.cfg.scale_features or self.cfg.scaler == "none":
            return frame

        from sklearn.preprocessing import RobustScaler, StandardScaler

        df = frame.copy()
        cols = df.select_dtypes("number").columns.tolist()
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

        if fit or self._scaler is None:
            self._scaler = (
                RobustScaler() if self.cfg.scaler == "robust" else StandardScaler()
            )
            df[cols] = self._scaler.fit_transform(df[cols].fillna(0.0))
        else:
            df[cols] = self._scaler.transform(df[cols].fillna(0.0))
        return df

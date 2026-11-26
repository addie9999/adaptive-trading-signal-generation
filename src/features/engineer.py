"""Composes feature groups and builds the supervised learning matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import FeatureGenerator
from src.features.generators import (
    LagFeatures,
    MomentumFeatures,
    StatisticalFeatures,
    TimeFeatures,
    TrendFeatures,
    VolatilityFeatures,
    VolumeFeatures,
)
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET_COL = "target_return"
DIRECTION_COL = "target_direction"


class FeatureEngineer:
    """Build the full feature matrix and forward-return label.

    The label is the ``target_horizon``-bar forward simple return of the close.
    Crucially the label is shifted so that row *t* only contains information
    available at *t*, with the target referring to *t+horizon* — preventing
    look-ahead leakage.
    """

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.features
        self.generators: list[FeatureGenerator] = [
            TrendFeatures(self.cfg.trend_windows),
            MomentumFeatures(self.cfg.momentum_windows),
            VolatilityFeatures(self.cfg.volatility_windows),
            VolumeFeatures(self.cfg.volume_windows),
            StatisticalFeatures(self.cfg.stat_windows),
            LagFeatures(self.cfg.lags),
            TimeFeatures(),
        ]

    # ------------------------------------------------------------------ #
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the OHLCV frame augmented with all engineered features."""
        blocks = [df]
        for gen in self.generators:
            block = gen.transform(df)
            logger.debug("Generated %d %s features", block.shape[1], gen.group)
            blocks.append(block)
        features = pd.concat(blocks, axis=1)
        features = features.replace([np.inf, -np.inf], np.nan)
        logger.info("Engineered %d feature columns", features.shape[1] - df.shape[1])
        return features

    def add_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append forward-return regression target and a direction label."""
        horizon = self.cfg.target_horizon
        out = df.copy()
        fwd = out["close"].shift(-horizon) / out["close"] - 1.0
        out[TARGET_COL] = fwd
        out[DIRECTION_COL] = np.sign(fwd).astype("float")
        return out

    def build_dataset(
        self, df: pd.DataFrame, dropna: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Convenience: return an ``(X, y)`` pair ready for modelling.

        Non-feature OHLCV columns and the direction label are excluded from
        ``X``; ``y`` is the forward return. Rows with NaNs (warm-up windows and
        the final unlabelled bars) are dropped when ``dropna`` is set.
        """
        full = self.add_label(self.build_features(df))
        feature_cols = self.feature_columns(full)
        data = full[feature_cols + [TARGET_COL]]
        if dropna:
            data = data.dropna()
        return data[feature_cols], data[TARGET_COL]

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        """Feature columns = everything except raw OHLCV and label columns."""
        from src.data.base import OHLCV_COLUMNS

        exclude = set(OHLCV_COLUMNS) | {TARGET_COL, DIRECTION_COL}
        return [c for c in df.columns if c not in exclude]

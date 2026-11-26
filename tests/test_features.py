"""Tests for cleaning and feature engineering (incl. no-leakage check)."""

from __future__ import annotations

import numpy as np

from src.cleaning import DataCleaner
from src.features import FeatureEngineer
from src.features.engineer import TARGET_COL


def test_cleaning_removes_nan_and_sorts(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    assert clean.index.is_monotonic_increasing
    assert not clean[["open", "high", "low", "close"]].isna().any().any()


def test_feature_count_is_substantial(config, synthetic_ohlcv):
    X, y = FeatureEngineer(config).build_dataset(synthetic_ohlcv)
    assert X.shape[1] >= 40           # ~50 engineered features
    assert len(X) == len(y)
    assert not X.isna().any().any()


def test_label_has_no_lookahead(config, synthetic_ohlcv):
    eng = FeatureEngineer(config)
    full = eng.add_label(eng.build_features(synthetic_ohlcv))
    # Forward return at t must equal close[t+h]/close[t]-1.
    h = config.features.target_horizon
    expected = full["close"].shift(-h) / full["close"] - 1.0
    np.testing.assert_allclose(
        full[TARGET_COL].dropna().values, expected.dropna().values, rtol=1e-9
    )

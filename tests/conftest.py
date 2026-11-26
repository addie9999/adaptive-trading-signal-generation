"""Shared pytest fixtures: synthetic market data and config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.config import AppConfig, load_config


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """A deterministic 1,200-bar OHLCV frame with regime-like structure."""
    rng = np.random.default_rng(7)
    n = 1200
    idx = pd.bdate_range("2016-01-01", periods=n)

    # Concatenate drift regimes so the HMM has structure to find.
    drifts = np.concatenate(
        [
            np.full(400, 0.0006),   # bull
            np.full(300, -0.0006),  # bear
            np.full(300, 0.0000),   # sideways
            np.full(200, 0.0002),   # recovery
        ]
    )
    vols = np.concatenate(
        [np.full(400, 0.010), np.full(300, 0.022), np.full(300, 0.008), np.full(200, 0.014)]
    )
    rets = rng.normal(drifts, vols)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Default config with a small validation window so folds form on test data."""
    return load_config(
        overrides={
            "validation": {"train_period": 400, "test_period": 150, "step": 150, "purge": 2},
            "models": {"enabled": ["arima", "xgboost"]},
        }
    )

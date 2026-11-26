"""Tests for walk-forward validation, backtesting and drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtesting import BacktestEngine
from src.cleaning import DataCleaner
from src.drift import DriftMonitor, population_stability_index
from src.features import FeatureEngineer
from src.models import build_model
from src.validation import WalkForwardValidator


def test_walk_forward_produces_ordered_folds(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    X, y = FeatureEngineer(config).build_dataset(clean)
    validator = WalkForwardValidator(config)

    folds = list(validator.split(len(X)))
    assert len(folds) >= 1
    for sp in folds:
        # Train strictly precedes test (purge gap enforced).
        assert sp.train_idx[-1] < sp.test_idx[0]


def test_walk_forward_evaluate(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    X, y = FeatureEngineer(config).build_dataset(clean)
    results = WalkForwardValidator(config).evaluate(
        lambda: build_model("xgboost", config), X, y
    )
    assert not results.empty
    assert "directional_accuracy" in results.columns


def test_backtest_runs_and_reports(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    prices = clean["close"]
    positions = pd.Series(np.where(prices.pct_change() > 0, 1, -1), index=prices.index)
    result = BacktestEngine(config).run(prices, positions)
    assert len(result.equity_curve) == len(prices)
    assert "sharpe" in result.metrics
    assert "n_trades" in result.metrics


def test_psi_detects_shift():
    a = np.random.normal(0, 1, 1000)
    b = np.random.normal(3, 1, 1000)  # shifted distribution
    assert population_stability_index(a, a) < 0.1
    assert population_stability_index(a, b) > 0.2


def test_drift_monitor_flags_retrain(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    X, _ = FeatureEngineer(config).build_dataset(clean)
    ref, cur = X.iloc[:300], X.iloc[300:] + 5.0  # force drift
    report = DriftMonitor(config).check(ref, cur)
    assert report.retrain_recommended

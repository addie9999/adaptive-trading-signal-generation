"""Tests for forecast-error and trading metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics import directional_accuracy, forecast_metrics, rmse, trading_metrics
from src.analytics.metrics import max_drawdown, sharpe_ratio


def test_rmse_zero_for_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0


def test_directional_accuracy():
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([0.5, -0.2, -0.1, -2.0])
    assert directional_accuracy(y_true, y_pred) == 0.75


def test_forecast_metrics_keys():
    y = np.random.randn(50)
    m = forecast_metrics(y, y + 0.01)
    assert set(m) == {"rmse", "mae", "directional_accuracy"}


def test_sharpe_and_drawdown_signs():
    up = pd.Series([0.01] * 100)
    assert sharpe_ratio(up) > 0
    mixed = pd.Series([0.05, -0.10, 0.02, -0.03])
    assert max_drawdown(mixed) <= 0


def test_trading_metrics_bundle():
    r = pd.Series(np.random.normal(0.001, 0.01, 252))
    m = trading_metrics(r)
    for key in ("sharpe", "sortino", "calmar", "max_drawdown", "cagr", "win_rate"):
        assert key in m

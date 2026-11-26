"""Forecast-error and trading-performance metrics.

Two families:

* **Forecast metrics** evaluate a model's numeric predictions (RMSE, MAE,
  directional accuracy).
* **Trading metrics** evaluate a realised return stream (Sharpe, Sortino,
  Calmar, max drawdown, CAGR, profit factor, win rate, ...).

All functions are pure and operate on NumPy arrays or pandas Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Forecast-error metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions whose sign matches the realised sign."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Bundle the standard forecast-error metrics."""
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Trading / return-stream metrics
# --------------------------------------------------------------------------- #
def _to_series(returns) -> pd.Series:
    if isinstance(returns, pd.Series):
        return returns.dropna().astype(float)
    return pd.Series(np.asarray(returns, float)).dropna()


def sharpe_ratio(returns, periods: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _to_series(returns) - rf / periods
    sd = r.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / sd)


def sortino_ratio(returns, periods: int = TRADING_DAYS, rf: float = 0.0) -> float:
    r = _to_series(returns) - rf / periods
    downside = r[r < 0].std()
    if downside == 0 or np.isnan(downside):
        return 0.0
    return float(np.sqrt(periods) * r.mean() / downside)


def max_drawdown(returns) -> float:
    """Maximum peak-to-trough drawdown of the cumulative equity curve."""
    r = _to_series(returns)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else 0.0


def cagr(returns, periods: int = TRADING_DAYS) -> float:
    r = _to_series(returns)
    if len(r) == 0:
        return 0.0
    total = (1 + r).prod()
    years = len(r) / periods
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def calmar_ratio(returns, periods: int = TRADING_DAYS) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(cagr(returns, periods) / mdd)


def profit_factor(returns) -> float:
    r = _to_series(returns)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(returns) -> float:
    r = _to_series(returns)
    nonzero = r[r != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def annual_return(returns, periods: int = TRADING_DAYS) -> float:
    return float(_to_series(returns).mean() * periods)


def annual_volatility(returns, periods: int = TRADING_DAYS) -> float:
    return float(_to_series(returns).std() * np.sqrt(periods))


def trading_metrics(returns, periods: int = TRADING_DAYS) -> dict[str, float]:
    """Full performance report for a realised return stream."""
    return {
        "sharpe": sharpe_ratio(returns, periods),
        "sortino": sortino_ratio(returns, periods),
        "calmar": calmar_ratio(returns, periods),
        "max_drawdown": max_drawdown(returns),
        "cagr": cagr(returns, periods),
        "annual_return": annual_return(returns, periods),
        "annual_volatility": annual_volatility(returns, periods),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
    }

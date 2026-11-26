"""Classical statistical tests used during alpha research.

These wrap statsmodels/SciPy primitives into small, typed result objects so
notebooks and reports can present them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationarityResult:
    """Outcome of a unit-root / stationarity test."""

    test: str
    statistic: float
    pvalue: float
    stationary: bool
    detail: str = ""


def adf_test(series: pd.Series, alpha: float = 0.05) -> StationarityResult:
    """Augmented Dickey-Fuller test (null: a unit root => non-stationary)."""
    from statsmodels.tsa.stattools import adfuller

    x = series.dropna().values
    stat, pvalue, *_ = adfuller(x, autolag="AIC")
    return StationarityResult(
        test="ADF",
        statistic=float(stat),
        pvalue=float(pvalue),
        stationary=pvalue < alpha,  # reject unit root => stationary
        detail="reject H0 (unit root) => stationary" if pvalue < alpha else "fail to reject H0",
    )


def kpss_test(series: pd.Series, alpha: float = 0.05) -> StationarityResult:
    """KPSS test (null: the series *is* stationary)."""
    from statsmodels.tsa.stattools import kpss

    x = series.dropna().values
    stat, pvalue, *_ = kpss(x, regression="c", nlags="auto")
    return StationarityResult(
        test="KPSS",
        statistic=float(stat),
        pvalue=float(pvalue),
        stationary=pvalue >= alpha,  # fail to reject => stationary
        detail="fail to reject H0 (stationary)" if pvalue >= alpha else "reject H0 => non-stationary",
    )


def ljung_box_test(series: pd.Series, lags: int = 10) -> dict[str, float]:
    """Ljung-Box test for autocorrelation up to ``lags`` (null: white noise)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox

    res = acorr_ljungbox(series.dropna(), lags=[lags], return_df=True)
    return {
        "lb_stat": float(res["lb_stat"].iloc[0]),
        "lb_pvalue": float(res["lb_pvalue"].iloc[0]),
    }


def normality_test(series: pd.Series) -> dict[str, float]:
    """Jarque-Bera normality test plus sample skew/kurtosis."""
    from scipy import stats

    x = series.dropna().values
    jb, p = stats.jarque_bera(x)
    return {
        "jarque_bera": float(jb),
        "pvalue": float(p),
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
    }


def hurst_exponent(series: pd.Series, max_lag: int = 50) -> float:
    """Estimate the Hurst exponent via the rescaled-range / variance method.

    H < 0.5 => mean-reverting, H ~ 0.5 => random walk, H > 0.5 => trending.
    """
    x = series.dropna().values
    lags = range(2, min(max_lag, len(x) // 2))
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    tau = np.asarray(tau)
    valid = tau > 0
    if valid.sum() < 2:
        return float("nan")
    poly = np.polyfit(np.log(np.fromiter(lags, float)[valid]), np.log(tau[valid]), 1)
    return float(poly[0])

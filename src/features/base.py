"""Base class and shared primitives for feature generators."""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class FeatureGenerator(abc.ABC):
    """A reusable block of features derived from an OHLCV frame.

    Subclasses implement :meth:`transform`, returning a DataFrame whose columns
    are the new features (aligned to the input index). Generators are pure and
    stateless so they can be reused across tickers and walk-forward folds.
    """

    #: Human-readable group name, used for logging and feature grouping.
    group: str = "base"

    @abc.abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    # -- shared indicator primitives ----------------------------------- #
    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n).mean()

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        prev_close = df["close"].shift(1)
        ranges = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        )
        return ranges.max(axis=1)

    @staticmethod
    def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / n, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / n, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        return 100 - (100 / (1 + rs))

"""Concrete feature-generator groups.

Together these emit ~50 features spanning trend, momentum, volatility, volume,
statistical, lag and calendar families. Indicators are implemented in pure
pandas/NumPy to avoid a native TA-Lib dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.base import FeatureGenerator


class TrendFeatures(FeatureGenerator):
    """SMA, EMA, MACD and ADX trend indicators."""

    group = "trend"

    def __init__(self, windows: list[int]) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        for n in self.windows:
            out[f"sma_{n}"] = self._sma(close, n)
            out[f"ema_{n}"] = self._ema(close, n)
            out[f"close_sma_{n}_ratio"] = close / out[f"sma_{n}"] - 1.0

        # MACD (12/26/9).
        ema12, ema26 = self._ema(close, 12), self._ema(close, 26)
        out["macd"] = ema12 - ema26
        out["macd_signal"] = self._ema(out["macd"], 9)
        out["macd_hist"] = out["macd"] - out["macd_signal"]

        out["adx_14"] = self._adx(df, 14)
        return out

    def _adx(self, df: pd.DataFrame, n: int = 14) -> pd.Series:
        up = df["high"].diff()
        down = -df["low"].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = self._true_range(df)
        atr = tr.ewm(alpha=1 / n, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
        return dx.ewm(alpha=1 / n, adjust=False).mean()


class MomentumFeatures(FeatureGenerator):
    """RSI, ROC, CCI and the stochastic oscillator."""

    group = "momentum"

    def __init__(self, windows: list[int]) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df["close"]
        for n in self.windows:
            out[f"rsi_{n}"] = self._rsi(close, n)
        out["roc_10"] = close.pct_change(10)
        out["cci_20"] = self._cci(df, 20)
        k, d = self._stochastic(df, 14, 3)
        out["stoch_k_14"] = k
        out["stoch_d_14"] = d
        return out

    @staticmethod
    def _cci(df: pd.DataFrame, n: int) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(n).mean()
        mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - sma) / (0.015 * mad.replace(0.0, np.nan))

    @staticmethod
    def _stochastic(df: pd.DataFrame, n: int, d: int) -> tuple[pd.Series, pd.Series]:
        low_n = df["low"].rolling(n).min()
        high_n = df["high"].rolling(n).max()
        k = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0.0, np.nan)
        return k, k.rolling(d).mean()


class VolatilityFeatures(FeatureGenerator):
    """ATR, historical/realised volatility, rolling std and Bollinger width."""

    group = "volatility"

    def __init__(self, windows: list[int]) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        log_ret = np.log(df["close"]).diff()
        tr = self._true_range(df)
        for n in self.windows:
            out[f"atr_{n}"] = tr.ewm(alpha=1 / n, adjust=False).mean()
            out[f"hist_vol_{n}"] = log_ret.rolling(n).std() * np.sqrt(252)
            out[f"rolling_std_{n}"] = df["close"].pct_change().rolling(n).std()
            ma = df["close"].rolling(n).mean()
            sd = df["close"].rolling(n).std()
            out[f"bb_width_{n}"] = (4 * sd) / ma.replace(0.0, np.nan)
        return out


class VolumeFeatures(FeatureGenerator):
    """VWAP, OBV, Money Flow Index and a volume ratio."""

    group = "volume"

    def __init__(self, windows: list[int]) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        out["vwap_20"] = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        out["obv"] = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()
        out["mfi_14"] = self._mfi(df, 14)
        for n in self.windows:
            out[f"volume_ratio_{n}"] = df["volume"] / df["volume"].rolling(n).mean()
        return out

    @staticmethod
    def _mfi(df: pd.DataFrame, n: int) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3
        mf = tp * df["volume"]
        pos = mf.where(tp.diff() > 0, 0.0).rolling(n).sum()
        neg = mf.where(tp.diff() < 0, 0.0).rolling(n).sum()
        mr = pos / neg.replace(0.0, np.nan)
        return 100 - (100 / (1 + mr))


class StatisticalFeatures(FeatureGenerator):
    """Rolling mean, variance, skewness, kurtosis and autocorrelation."""

    group = "statistical"

    def __init__(self, windows: list[int]) -> None:
        self.windows = windows

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change()
        for n in self.windows:
            out[f"roll_mean_{n}"] = ret.rolling(n).mean()
            out[f"roll_var_{n}"] = ret.rolling(n).var()
            out[f"roll_skew_{n}"] = ret.rolling(n).skew()
            out[f"roll_kurt_{n}"] = ret.rolling(n).kurt()
            out[f"autocorr_{n}"] = ret.rolling(n).apply(
                lambda x: pd.Series(x).autocorr(lag=1), raw=False
            )
        return out


class LagFeatures(FeatureGenerator):
    """Lagged returns, RSI and volume to expose short-horizon memory."""

    group = "lag"

    def __init__(self, lags: list[int]) -> None:
        self.lags = lags

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ret = df["close"].pct_change()
        rsi = self._rsi(df["close"], 14)
        for lag in self.lags:
            out[f"lag_ret_{lag}"] = ret.shift(lag)
            out[f"lag_rsi_{lag}"] = rsi.shift(lag)
            out[f"lag_volume_{lag}"] = df["volume"].pct_change().shift(lag)
        return out


class TimeFeatures(FeatureGenerator):
    """Calendar features: day-of-week, week, month, quarter, holiday proximity."""

    group = "time"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        idx = df.index
        out = pd.DataFrame(index=idx)
        out["day_of_week"] = idx.dayofweek
        out["week_of_year"] = idx.isocalendar().week.astype(int).values
        out["month"] = idx.month
        out["quarter"] = idx.quarter
        out["is_month_end"] = idx.is_month_end.astype(int)
        out["is_holiday_adjacent"] = self._holiday_adjacent(idx)
        return out

    @staticmethod
    def _holiday_adjacent(idx: pd.DatetimeIndex) -> pd.Series:
        """1 if the next/prev calendar day skips a US market holiday/weekend gap."""
        try:
            from pandas.tseries.holiday import USFederalHolidayCalendar

            holidays = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
            near = pd.Series(0, index=idx)
            holset = set(holidays.normalize())
            for offset in (1, -1):
                shifted = (idx + pd.Timedelta(days=offset)).normalize()
                near |= pd.Series(shifted.isin(holset).astype(int), index=idx)
            return near
        except Exception:  # noqa: BLE001
            return pd.Series(0, index=idx)

"""Concrete market-data providers.

Only :class:`YahooFinanceProvider` is implemented today. Stubs for Polygon,
Finnhub and AlphaVantage document the extension point and raise a clear
``NotImplementedError`` until wired up.
"""

from __future__ import annotations

import pandas as pd

from src.data.base import OHLCV_COLUMNS, DataProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)


class YahooFinanceProvider(DataProvider):
    """Historical OHLCV via the ``yfinance`` library (free, no API key)."""

    name = "yfinance"

    def _download_one(
        self, ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        import yfinance as yf  # lazy import — optional at framework import time

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if raw.empty:
            raise ValueError("empty response")

        # yfinance may return a MultiIndex (column, ticker) for single symbols.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
        frame = raw.rename(columns=rename)
        if "adj_close" not in frame.columns:  # auto_adjust edge case
            frame["adj_close"] = frame["close"]

        frame = frame[OHLCV_COLUMNS].copy()
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frame.index.name = "date"
        return frame


class _UnimplementedProvider(DataProvider):
    """Shared base for providers that are planned but not yet implemented."""

    def _download_one(self, ticker, start, end, interval):  # noqa: ANN001
        raise NotImplementedError(
            f"Provider '{self.name}' is not implemented yet. "
            "Implement `_download_one` to normalise its API to the OHLCV schema."
        )


class PolygonProvider(_UnimplementedProvider):
    name = "polygon"


class FinnhubProvider(_UnimplementedProvider):
    name = "finnhub"


class AlphaVantageProvider(_UnimplementedProvider):
    name = "alphavantage"


_REGISTRY: dict[str, type[DataProvider]] = {
    "yfinance": YahooFinanceProvider,
    "polygon": PolygonProvider,
    "finnhub": FinnhubProvider,
    "alphavantage": AlphaVantageProvider,
}


def get_provider(name: str) -> DataProvider:
    """Factory: instantiate a provider by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown provider '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]()

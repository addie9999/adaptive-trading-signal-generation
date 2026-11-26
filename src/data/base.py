"""Provider-agnostic contracts for market data."""

from __future__ import annotations

import abc

import pandas as pd

# Canonical column schema used everywhere downstream. Providers must normalise
# their raw output into exactly these columns (lower-case, snake_case).
OHLCV_COLUMNS: list[str] = ["open", "high", "low", "close", "adj_close", "volume"]


class DataProvider(abc.ABC):
    """Abstract base class for a historical market-data source.

    Concrete implementations (Yahoo, Polygon, Finnhub, AlphaVantage) only need
    to implement :meth:`_download_one`, returning a frame indexed by a
    timezone-naive ``DatetimeIndex`` with the :data:`OHLCV_COLUMNS` schema.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def _download_one(
        self, ticker: str, start: str, end: str | None, interval: str
    ) -> pd.DataFrame:
        """Download a single ticker and normalise it to the canonical schema."""

    def download(
        self,
        tickers: str | list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Download one or more tickers.

        Returns a mapping ``ticker -> OHLCV DataFrame``. Tickers that fail to
        download are skipped with a logged warning rather than aborting the run.
        """
        from src.utils.logging import get_logger

        logger = get_logger(__name__)
        if isinstance(tickers, str):
            tickers = [tickers]

        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                frame = self._download_one(ticker, start, end, interval)
                self._validate(frame, ticker)
                out[ticker] = frame
                logger.info("Downloaded %s rows for %s via %s", len(frame), ticker, self.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to download %s via %s: %s", ticker, self.name, exc)
        return out

    @staticmethod
    def _validate(frame: pd.DataFrame, ticker: str) -> None:
        missing = set(OHLCV_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{ticker}: provider returned frame missing columns {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(f"{ticker}: provider frame is not indexed by datetime")

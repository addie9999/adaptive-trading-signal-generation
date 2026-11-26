"""High-level market-data loader with local parquet caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.data.providers import get_provider
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataLoader:
    """Fetch, cache and persist OHLCV market data.

    Downloaded frames are cached on disk (parquet) keyed by a hash of the
    request parameters, so repeated research runs do not re-hit the provider.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.provider = get_provider(config.data.provider)
        self.cache_dir = Path(config.data.cache_dir)
        self.processed_dir = Path(config.data.processed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _cache_path(self, ticker: str) -> Path:
        d = self.config.data
        key = f"{d.provider}|{ticker}|{d.start}|{d.end}|{d.interval}"
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        return self.cache_dir / f"{ticker}_{d.interval}_{digest}.parquet"

    def load(self, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Return ``ticker -> OHLCV`` frames, using the cache where possible."""
        tickers = tickers or self.config.data.tickers
        out: dict[str, pd.DataFrame] = {}
        to_download: list[str] = []

        for ticker in tickers:
            path = self._cache_path(ticker)
            if self.config.data.use_cache and path.exists():
                out[ticker] = pd.read_parquet(path)
                logger.info("Loaded %s from cache (%s rows)", ticker, len(out[ticker]))
            else:
                to_download.append(ticker)

        if to_download:
            fetched = self.provider.download(
                to_download,
                start=self.config.data.start,
                end=self.config.data.end,
                interval=self.config.data.interval,
            )
            for ticker, frame in fetched.items():
                frame.to_parquet(self._cache_path(ticker))
                out[ticker] = frame

        return out

    def load_one(self, ticker: str) -> pd.DataFrame:
        """Convenience accessor returning a single ticker's frame."""
        data = self.load([ticker])
        if ticker not in data:
            raise ValueError(f"No data available for {ticker}")
        return data[ticker]

    def save_processed(self, ticker: str, frame: pd.DataFrame) -> Path:
        """Persist a processed/feature frame to the processed directory."""
        path = self.processed_dir / f"{ticker}_processed.parquet"
        frame.to_parquet(path)
        logger.info("Saved processed dataset for %s -> %s", ticker, path)
        return path

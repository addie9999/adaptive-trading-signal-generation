"""Market data acquisition layer.

Exposes a provider-agnostic :class:`MarketDataLoader` plus the
:class:`DataProvider` interface that concrete providers implement.
"""

from src.data.base import OHLCV_COLUMNS, DataProvider
from src.data.loader import MarketDataLoader
from src.data.providers import YahooFinanceProvider, get_provider

__all__ = [
    "DataProvider",
    "OHLCV_COLUMNS",
    "MarketDataLoader",
    "YahooFinanceProvider",
    "get_provider",
]

"""A vectorised long/flat/short backtester with realistic frictions.

Design choices
--------------
* Signals are acted on at the *next* bar's open-to-close return to avoid
  look-ahead (a signal generated at close of *t* takes effect over *t+1*).
* Commission and slippage are charged in basis points on every change of
  position (turnover), so flipping long->short costs two units of turnover.
* Outputs a trade log, an equity curve and a full performance report.

For research that needs a fuller event model (limit orders, partial fills),
swap this for VectorBT/Backtesting.py behind the same ``BacktestResult`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics.metrics import trading_metrics
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest artefacts."""

    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0)


class BacktestEngine:
    """Turn a target-position series into a costed equity curve."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.backtesting

    def run(
        self,
        prices: pd.Series,
        positions: pd.Series,
    ) -> BacktestResult:
        """Backtest a target-position series.

        Parameters
        ----------
        prices:
            Close price series.
        positions:
            Desired position in ``[-1, 1]`` for each bar (1 = fully long,
            -1 = fully short, 0 = flat). Applied to the *next* bar's return.
        """
        prices = prices.astype(float)
        positions = positions.reindex(prices.index).fillna(0.0).clip(-1, 1)

        # Act on next-bar returns: shift positions forward by one.
        asset_ret = prices.pct_change().fillna(0.0)
        applied_pos = positions.shift(1).fillna(0.0)
        gross = applied_pos * asset_ret

        # Costs proportional to turnover (|Δposition|).
        turnover = applied_pos.diff().abs().fillna(applied_pos.abs())
        cost_rate = (self.cfg.commission_bps + self.cfg.slippage_bps) / 1e4
        costs = turnover * cost_rate

        net_ret = (gross - costs) * self.cfg.position_size_fraction
        equity = self.cfg.initial_cash * (1 + net_ret).cumprod()

        trades = self._trade_log(prices, applied_pos, net_ret)
        metrics = trading_metrics(net_ret)
        metrics["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        metrics["n_trades"] = int(len(trades))
        metrics["turnover"] = float(turnover.sum())

        logger.info(
            "Backtest done: total_return=%.2f%% sharpe=%.2f maxDD=%.2f%% trades=%d",
            metrics["total_return"] * 100, metrics["sharpe"],
            metrics["max_drawdown"] * 100, metrics["n_trades"],
        )
        return BacktestResult(equity, net_ret, trades, metrics)

    @staticmethod
    def _trade_log(
        prices: pd.Series, positions: pd.Series, net_ret: pd.Series
    ) -> pd.DataFrame:
        """Reconstruct discrete trades from position changes."""
        records: list[dict] = []
        entry_idx = None
        entry_price = None
        entry_pos = 0.0
        for ts, pos in positions.items():
            if entry_idx is None and pos != 0:
                entry_idx, entry_price, entry_pos = ts, prices.loc[ts], pos
            elif entry_idx is not None and pos != entry_pos:
                exit_price = prices.loc[ts]
                pnl = entry_pos * (exit_price / entry_price - 1.0)
                records.append(
                    {
                        "entry_time": entry_idx,
                        "exit_time": ts,
                        "direction": "long" if entry_pos > 0 else "short",
                        "entry_price": round(float(entry_price), 4),
                        "exit_price": round(float(exit_price), 4),
                        "return": round(float(pnl), 5),
                    }
                )
                if pos != 0:  # immediately re-entered (flip)
                    entry_idx, entry_price, entry_pos = ts, exit_price, pos
                else:
                    entry_idx = None
        return pd.DataFrame(records)

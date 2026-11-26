"""Matplotlib plotting helpers that save figures to the reports directory.

All functions use a non-interactive backend and return the saved path so they
can be logged as MLflow artifacts.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

_REGIME_COLORS = {
    "bull": "#2ca02c",
    "bear": "#d62728",
    "sideways": "#7f7f7f",
    "high_volatility": "#ff7f0e",
}


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_regimes(
    prices: pd.Series, regimes: pd.Series, out: str = "reports/regimes.png"
) -> Path:
    """Overlay detected regimes as shaded spans on the price series."""
    path = Path(out)
    _ensure_dir(path)
    aligned = regimes.reindex(prices.index).ffill()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(prices.index, prices.values, color="black", lw=1.0, label="Close")
    for label, color in _REGIME_COLORS.items():
        mask = aligned == label
        ax.fill_between(
            prices.index, prices.min(), prices.max(), where=mask.values,
            color=color, alpha=0.15, label=label,
        )
    ax.set_title("Market Regimes (HMM)")
    ax.legend(loc="upper left", ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved regime plot -> %s", path)
    return path


def plot_equity_curve(
    equity: pd.Series, benchmark: pd.Series | None = None, out: str = "reports/equity.png"
) -> Path:
    path = Path(out)
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(equity.index, equity.values, label="Strategy", color="#1f77b4")
    if benchmark is not None:
        scaled = benchmark / benchmark.iloc[0] * equity.iloc[0]
        ax.plot(scaled.index, scaled.values, label="Buy & Hold", color="grey", alpha=0.7)
    ax.set_title("Equity Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved equity curve -> %s", path)
    return path


def plot_drawdown(equity: pd.Series, out: str = "reports/drawdown.png") -> Path:
    path = Path(out)
    _ensure_dir(path)
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    ax.set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved drawdown plot -> %s", path)
    return path


def plot_walk_forward(results: pd.DataFrame, out: str = "reports/walk_forward.png") -> Path:
    """Bar chart of per-fold RMSE and directional accuracy."""
    path = Path(out)
    _ensure_dir(path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(results["fold"], results["rmse"], color="#1f77b4")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Walk-Forward Validation by Fold")
    ax2.bar(results["fold"], results["directional_accuracy"], color="#2ca02c")
    ax2.axhline(0.5, color="black", ls="--", lw=0.8)
    ax2.set_ylabel("Directional Acc.")
    ax2.set_xlabel("Fold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved walk-forward plot -> %s", path)
    return path

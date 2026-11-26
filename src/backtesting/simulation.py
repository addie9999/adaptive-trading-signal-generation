"""Historical path scenarios from circular blocks of cost-adjusted returns."""
import numpy as np


def bootstrap_report(returns, *, paths: int, block_size: int, seed: int) -> dict:
    """Resample paths of the observed length. Drawdowns are positive loss magnitudes."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Provide a finite one-dimensional series of at least two net returns.")
    if np.any(values <= -1):
        raise ValueError("Each net return must be greater than -100% for log compounding.")
    if not 1 <= block_size < len(values):
        raise ValueError("Block size must be positive and shorter than the return history.")
    if paths < 1:
        raise ValueError("Number of simulation paths must be positive.")
    n = len(values)
    logs = np.log1p(values)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_size)
    blocks = (n + block_size - 1) // block_size
    totals = np.empty(paths)
    drawdowns = np.empty(paths)
    for i in range(paths):
        starts = rng.integers(n, size=blocks)
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        wealth = np.cumsum(logs[indices])
        peaks = np.maximum(0, np.maximum.accumulate(wealth))
        totals[i] = np.expm1(wealth[-1])
        drawdowns[i] = -np.expm1((wealth - peaks).min())

    def percentiles(sample):
        return dict(zip(("p05", "p50", "p95"), np.quantile(sample, [.05, .50, .95]).tolist()))

    return {"method": "circular_block_bootstrap", "paths": paths, "bars_per_path": n,
            "block_size": block_size, "seed": seed, "terminal_return": percentiles(totals),
            "max_drawdown": percentiles(drawdowns), "fraction_ending_below_start": float(np.mean(totals < 0))}

import numpy as np
import pytest

from main import _cmd_backtest, build_parser
from src.backtesting.simulation import bootstrap_report


def test_compounding_and_first_bar_drawdown():
    report = bootstrap_report([-.1] * 4, paths=20, block_size=2, seed=1)
    loss = 1 - .9**4
    assert report["terminal_return"]["p50"] == pytest.approx(-loss)
    assert report["max_drawdown"]["p95"] == pytest.approx(loss)
    assert report["fraction_ending_below_start"] == 1


def test_circular_blocks_against_reference_path():
    values = np.array([.05, -.20, .12, -.08, .03])
    seed = 4
    starts = np.random.default_rng(seed).integers(5, size=3)
    sample = [values[(start + j) % 5] for start in starts for j in range(2)][:5]
    wealth = np.r_[1, np.cumprod(1 + np.array(sample))]
    report = bootstrap_report(values, paths=1, block_size=2, seed=seed)
    assert report["terminal_return"]["p50"] == pytest.approx(wealth[-1]-1)
    assert report["max_drawdown"]["p50"] == pytest.approx(max(1-wealth/np.maximum.accumulate(wealth)))


@pytest.mark.parametrize("values,paths,block", [([], 10, 1), ([0, float("nan")], 5, 1),
                                               ([0, -1], 2, 1), ([0, .1], 0, 1), ([0, .1], 2, 2)])
def test_invalid_scenarios(values, paths, block):
    with pytest.raises(ValueError):
        bootstrap_report(values, paths=paths, block_size=block, seed=2)


def test_backtest_cli_simulation(monkeypatch, synthetic_ohlcv, capsys):
    from src.data import MarketDataLoader
    import src.reporting

    monkeypatch.setattr(MarketDataLoader, "load_one", lambda self, ticker: synthetic_ohlcv)
    monkeypatch.setattr(src.reporting, "plot_equity_curve", lambda *a, **kw: None)
    monkeypatch.setattr(src.reporting, "plot_drawdown", lambda *a: None)
    args = build_parser().parse_args(["backtest", "--ticker", "TEST", "--simulations", "10"])
    _cmd_backtest(args)
    assert '"circular_block_bootstrap"' in capsys.readouterr().out

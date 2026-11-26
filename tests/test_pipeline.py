"""End-to-end pipeline test with the data provider mocked out (no network)."""

from __future__ import annotations

import pytest

from src.alpha import SignalType
from src.pipeline import InferencePipeline, TrainingPipeline


@pytest.fixture
def offline(monkeypatch, synthetic_ohlcv):
    """Patch the loader so pipelines use synthetic data instead of yfinance."""
    from src.data.loader import MarketDataLoader

    monkeypatch.setattr(MarketDataLoader, "load_one", lambda self, ticker: synthetic_ohlcv)
    monkeypatch.setattr(
        MarketDataLoader, "load", lambda self, tickers=None: {"TEST": synthetic_ohlcv}
    )


def test_training_then_inference(config, offline):
    bundle = TrainingPipeline(config).run("TEST")
    assert bundle.ticker == "TEST"
    assert bundle.selector.models                      # at least one model trained
    assert len(bundle.feature_columns) >= 40

    result = InferencePipeline(bundle).predict("TEST")
    assert result.signal.signal in set(SignalType)
    assert result.regime
    assert result.selected_model in {"arima", "xgboost"}
    # Non-HOLD signals must carry a risk decision.
    if result.signal.signal != SignalType.HOLD:
        assert result.risk is not None
        assert result.risk.stop_loss > 0


def test_bundle_save_load(config, offline, tmp_path):
    from src.pipeline import AlphaBundle

    bundle = TrainingPipeline(config).run("TEST")
    path = tmp_path / "bundle.joblib"
    bundle.save(path)

    reloaded = AlphaBundle.load(path)
    out = InferencePipeline(reloaded).predict("TEST")
    assert out.to_dict()["ticker"] == "TEST"

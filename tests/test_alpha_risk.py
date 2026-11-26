"""Tests for alpha signal generation and risk management."""

from __future__ import annotations

from src.alpha import AlphaSignalGenerator, SignalType
from src.risk import RiskManager


def test_signal_thresholds(config):
    gen = AlphaSignalGenerator(config)
    assert gen.generate(0.05).signal == SignalType.BUY
    assert gen.generate(-0.05).signal == SignalType.SELL
    assert gen.generate(0.0).signal == SignalType.HOLD


def test_confidence_bounded(config):
    gen = AlphaSignalGenerator(config)
    sig = gen.generate(10.0)  # extreme return
    assert 0.0 <= sig.confidence <= 1.0


def test_risk_long_levels_make_sense(config):
    rm = RiskManager(config)
    decision = rm.evaluate(SignalType.BUY, price=100.0, atr=2.0, expected_return=0.02)
    assert decision.stop_loss < 100.0 < decision.take_profit
    assert decision.position_size > 0
    assert decision.risk_reward_ratio > 0


def test_risk_short_levels_inverted(config):
    rm = RiskManager(config)
    decision = rm.evaluate(SignalType.SELL, price=100.0, atr=2.0, expected_return=-0.02)
    assert decision.take_profit < 100.0 < decision.stop_loss
    assert decision.position_size < 0


def test_position_capped_by_max_fraction(config):
    rm = RiskManager(config)
    decision = rm.evaluate(SignalType.BUY, price=10.0, atr=0.001, expected_return=0.01)
    max_notional = config.risk.account_equity * config.risk.max_position_fraction
    assert decision.notional <= max_notional + 1e-6

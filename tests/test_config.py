"""Tests for layered configuration loading."""

from __future__ import annotations

import os

from src.utils.config import load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg.project.name == "aqaf"
    assert cfg.data.provider == "yfinance"
    assert cfg.regime.n_states == 4


def test_env_layer_overrides_default():
    cfg = load_config(env="research")
    assert "MSFT" in cfg.data.tickers
    assert cfg.project.log_level == "DEBUG"


def test_explicit_overrides_win():
    cfg = load_config(overrides={"regime": {"n_states": 3}})
    assert cfg.regime.n_states == 3


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("AQAF__PROJECT__SEED", "123")
    cfg = load_config()
    assert cfg.project.seed == 123
    os.environ.pop("AQAF__PROJECT__SEED", None)

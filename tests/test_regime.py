"""Tests for the HMM regime detector."""

from __future__ import annotations

from src.cleaning import DataCleaner
from src.regime import HMMRegimeDetector, RegimeLabel


def test_regime_fit_and_labels(config, synthetic_ohlcv):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    detector = HMMRegimeDetector(config).fit(clean)
    out = detector.predict(clean)

    assert {"state", "regime"} <= set(out.columns)
    valid = {r.value for r in RegimeLabel}
    assert set(out["regime"].unique()) <= valid
    assert detector.current_regime(clean) in valid


def test_regime_save_load(config, synthetic_ohlcv, tmp_path):
    clean = DataCleaner(config).clean(synthetic_ohlcv)
    detector = HMMRegimeDetector(config).fit(clean)
    path = tmp_path / "regime.joblib"
    detector.save(str(path))

    reloaded = HMMRegimeDetector.load(str(path), config)
    assert reloaded.current_regime(clean) == detector.current_regime(clean)

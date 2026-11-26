"""Tests for the model interface, persistence and registry."""

from __future__ import annotations

from src.cleaning import DataCleaner
from src.features import FeatureEngineer
from src.models import available_models, build_model


def _xy(config, ohlcv):
    clean = DataCleaner(config).clean(ohlcv)
    return FeatureEngineer(config).build_dataset(clean)


def test_registry_lists_all_models():
    assert set(available_models()) == {"arima", "xgboost", "lstm", "transformer"}


def test_xgboost_fit_predict_evaluate(config, synthetic_ohlcv):
    X, y = _xy(config, synthetic_ohlcv)
    model = build_model("xgboost", config).fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    metrics = model.evaluate(X, y)
    assert metrics["rmse"] >= 0


def test_xgboost_save_load_roundtrip(config, synthetic_ohlcv, tmp_path):
    X, y = _xy(config, synthetic_ohlcv)
    model = build_model("xgboost", config).fit(X, y)
    path = tmp_path / "xgb.joblib"
    model.save(str(path))

    from src.models.xgboost import XGBoostForecaster

    reloaded = XGBoostForecaster.load(str(path))
    import numpy as np

    np.testing.assert_allclose(model.predict(X), reloaded.predict(X))


def test_arima_fit_predict(config, synthetic_ohlcv):
    X, y = _xy(config, synthetic_ohlcv)
    model = build_model("arima", config).fit(X.iloc[:500], y.iloc[:500])
    preds = model.predict(X.iloc[500:520])
    assert len(preds) == 20

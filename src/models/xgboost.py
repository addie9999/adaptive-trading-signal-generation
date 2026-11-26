"""Gradient-boosted-tree forecaster (XGBoost regressor).

Predicts the forward return from the engineered tabular feature matrix. This
is the framework's default production model: fast to train, fast to serve, no
heavy runtime dependency, and natively supported by SHAP's TreeExplainer.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULTS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostForecaster:
        from xgboost import XGBRegressor

        cfg = {**_DEFAULTS, **self.params}
        self._model = XGBRegressor(**cfg)
        self._model.fit(X.values, y.values)
        self._record_metadata(X, y)
        logger.info("XGBoost fitted on %d samples, %d features", len(y), X.shape[1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._model.predict(X.values), dtype=float)

    def feature_importances(self) -> pd.Series:
        """Gain-based feature importance as a named, sorted Series."""
        self._check_fitted()
        names = self.metadata.feature_names if self.metadata else None
        return pd.Series(self._model.feature_importances_, index=names).sort_values(
            ascending=False
        )

    @property
    def sklearn_model(self):  # used by SHAP TreeExplainer
        self._check_fitted()
        return self._model

    def save(self, path: str) -> None:
        self._check_fitted()
        joblib.dump({"model": self._model, "metadata": self.metadata, "params": self.params}, path)

    @classmethod
    def load(cls, path: str) -> XGBoostForecaster:
        blob = joblib.load(path)
        obj = cls(blob["params"])
        obj._model = blob["model"]
        obj.metadata = blob["metadata"]
        obj._fitted = True
        return obj

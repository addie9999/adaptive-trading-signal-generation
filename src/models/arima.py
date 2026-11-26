"""ARIMA forecaster (statsmodels SARIMAX backend).

ARIMA models the forward-return *series* directly and ignores the engineered
feature matrix beyond using its index/length. Predictions are produced by
one-step-ahead forecasting, which keeps the model honest under walk-forward
evaluation.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ARIMAForecaster(BaseForecaster):
    name = "arima"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.order = tuple(self.params.get("order", (5, 1, 0)))
        self.seasonal_order = tuple(self.params.get("seasonal_order", (0, 0, 0, 0)))
        self._result = None
        self._train_endog: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ARIMAForecaster:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._train_endog = y.astype(float).reset_index(drop=True)
        model = SARIMAX(
            self._train_endog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False)
        self._record_metadata(X, y)
        logger.info("ARIMA%s fitted on %d samples", self.order, len(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """One-step-ahead forecasts for ``len(X)`` future steps.

        The fitted state is rolled forward without re-estimation (a fast,
        leakage-free approximation suitable for walk-forward test windows).
        """
        self._check_fitted()
        steps = len(X)
        forecast = self._result.forecast(steps=steps)
        return np.asarray(forecast, dtype=float)

    def save(self, path: str) -> None:
        self._check_fitted()
        joblib.dump(
            {
                "result": self._result,
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> ARIMAForecaster:
        blob = joblib.load(path)
        obj = cls({"order": blob["order"], "seasonal_order": blob["seasonal_order"]})
        obj._result = blob["result"]
        obj.metadata = blob["metadata"]
        obj._fitted = True
        return obj

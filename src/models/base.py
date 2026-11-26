"""Common forecasting-model interface.

Every model — statistical (ARIMA), tree-based (XGBoost) or deep
(LSTM/Transformer) — implements the same five-method contract so the adaptive
selector, walk-forward engine and serving layer can treat them uniformly:

    fit(X, y) -> self
    predict(X) -> np.ndarray
    save(path) / load(path)
    evaluate(X, y) -> dict[str, float]

Targets are forward *returns*; predictions are forward-return estimates.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.analytics.metrics import forecast_metrics


@dataclass
class ModelMetadata:
    """Lightweight provenance attached to every fitted model."""

    name: str
    params: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    trained_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    n_train_samples: int = 0


class BaseForecaster(abc.ABC):
    """Abstract base class for all forecasters."""

    name: str = "base"
    #: Tree/linear models consume a tabular feature matrix; sequence models
    #: (LSTM/Transformer) build their own windows from the ordered series.
    requires_sequences: bool = False

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self.metadata: ModelMetadata | None = None
        self._fitted = False

    # -- core contract -------------------------------------------------- #
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseForecaster:
        ...

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abc.abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> BaseForecaster:
        ...

    # -- shared behaviour ----------------------------------------------- #
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return RMSE / MAE / directional accuracy on a holdout set."""
        preds = self.predict(X)
        y_true = np.asarray(y, float)
        # Sequence models emit fewer predictions than rows (warm-up window);
        # align to the tail so lengths match.
        if len(preds) != len(y_true):
            y_true = y_true[-len(preds):]
        return forecast_metrics(y_true, preds)

    def _record_metadata(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.metadata = ModelMetadata(
            name=self.name,
            params=self.params,
            feature_names=list(X.columns),
            n_train_samples=len(y),
        )
        self._fitted = True

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} model is not fitted; call fit() first")

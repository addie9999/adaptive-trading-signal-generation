"""Weighted-average ensemble forecaster.

Wraps several fitted forecasters and blends their predictions. Weights can be
supplied directly or learned as inverse-RMSE weights on a validation set,
implementing the "weighted model averaging" bonus requirement.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src.analytics.metrics import rmse
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleForecaster(BaseForecaster):
    name = "ensemble"

    def __init__(self, models: list[BaseForecaster], weights: list[float] | None = None) -> None:
        super().__init__({})
        if not models:
            raise ValueError("EnsembleForecaster requires at least one model")
        self.models = models
        self.weights = self._normalise(weights or [1.0] * len(models))

    @staticmethod
    def _normalise(weights: list[float]) -> np.ndarray:
        w = np.asarray(weights, float)
        total = w.sum()
        return w / total if total > 0 else np.full_like(w, 1 / len(w))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> EnsembleForecaster:
        for m in self.models:
            if not getattr(m, "_fitted", False):
                m.fit(X, y)
        self._record_metadata(X, y)
        return self

    def fit_weights(self, X: pd.DataFrame, y: pd.Series) -> EnsembleForecaster:
        """Learn inverse-RMSE weights on a validation window."""
        rmses = []
        yt = np.asarray(y, float)
        for m in self.models:
            preds = m.predict(X)
            yy = yt[-len(preds):] if len(preds) != len(yt) else yt
            rmses.append(rmse(yy, preds))
        inv = 1.0 / (np.asarray(rmses) + 1e-9)
        self.weights = self._normalise(inv.tolist())
        logger.info(
            "Ensemble weights: %s",
            {m.name: round(w, 3) for m, w in zip(self.models, self.weights, strict=False)},
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = [np.asarray(m.predict(X), float) for m in self.models]
        min_len = min(len(p) for p in preds)
        stacked = np.vstack([p[-min_len:] for p in preds])
        return np.average(stacked, axis=0, weights=self.weights)

    def save(self, path: str) -> None:
        joblib.dump({"weights": self.weights, "names": [m.name for m in self.models]}, path)

    @classmethod
    def load(cls, path: str) -> EnsembleForecaster:  # pragma: no cover
        raise NotImplementedError(
            "Reload member models individually, then reconstruct the ensemble."
        )

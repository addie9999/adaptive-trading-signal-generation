"""Adaptive model selection: pick the best forecaster *per market regime*.

The selector trains every enabled model, then evaluates each one on the test
rows belonging to each regime. For every regime it records the winning model
(by a configurable metric). At runtime, given the current regime, it routes
prediction to that regime's champion model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.metrics import directional_accuracy, forecast_metrics
from src.models import BaseForecaster, build_model
from src.models.base import BaseForecaster as _BF
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Metrics where a *lower* value is better (so selection inverts the comparison).
_LOWER_IS_BETTER = {"rmse", "mae"}


@dataclass
class RegimeModelScore:
    """Per-regime, per-model evaluation record."""

    regime: str
    model: str
    metrics: dict[str, float] = field(default_factory=dict)
    n_samples: int = 0


class AdaptiveModelSelector:
    """Train models and learn the best one for each detected regime."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cfg = config.adaptive
        self.models: dict[str, _BF] = {}
        self.scores: list[RegimeModelScore] = []
        self.best_by_regime: dict[str, str] = {}
        self.fallback_model: str | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> AdaptiveModelSelector:
        """Train all enabled models on the full training window."""
        for name in self.config.models.enabled:
            try:
                model = build_model(name, self.config)
                model.fit(X, y)
                self.models[name] = model
                logger.info("Trained model '%s'", name)
            except Exception as exc:  # noqa: BLE001 - skip unavailable backends
                logger.warning("Skipping model '%s': %s", name, exc)
        if not self.models:
            raise RuntimeError("No models could be trained (check optional deps)")
        return self

    def evaluate_by_regime(
        self, X: pd.DataFrame, y: pd.Series, regimes: pd.Series
    ) -> pd.DataFrame:
        """Score every model within each regime on a holdout window.

        ``regimes`` is the per-row regime label aligned to ``X``/``y``.
        Populates :attr:`best_by_regime` and returns a tidy score table.
        """
        metric = self.cfg.selection_metric
        rows: list[RegimeModelScore] = []
        aligned = regimes.reindex(X.index).ffill()

        for regime in sorted(aligned.dropna().unique()):
            mask = (aligned == regime).values
            if mask.sum() < self.cfg.min_samples_per_regime:
                logger.debug("Regime '%s' has too few samples (%d)", regime, mask.sum())
                continue
            Xr, yr = X[mask], y[mask]
            for name, model in self.models.items():
                preds = model.predict(Xr)
                yt = np.asarray(yr, float)
                if len(preds) != len(yt):
                    yt = yt[-len(preds):]
                m = forecast_metrics(yt, preds)
                m["directional_accuracy"] = directional_accuracy(yt, preds)
                rows.append(RegimeModelScore(regime, name, m, int(mask.sum())))

        self.scores = rows
        self._select_best(metric)
        return self._scores_to_frame(rows)

    def _select_best(self, metric: str) -> None:
        by_regime: dict[str, list[RegimeModelScore]] = {}
        for s in self.scores:
            by_regime.setdefault(s.regime, []).append(s)

        lower = metric in _LOWER_IS_BETTER
        for regime, scores in by_regime.items():
            best = (min if lower else max)(scores, key=lambda s: s.metrics.get(metric, np.nan))
            self.best_by_regime[regime] = best.model
            logger.info(
                "Regime '%s' -> best model '%s' (%s=%.4f)",
                regime, best.model, metric, best.metrics.get(metric, float("nan")),
            )
        # Global fallback = most frequently winning model.
        if self.best_by_regime:
            counts = pd.Series(list(self.best_by_regime.values())).value_counts()
            self.fallback_model = str(counts.index[0])

    @staticmethod
    def _scores_to_frame(rows: list[RegimeModelScore]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"regime": r.regime, "model": r.model, "n": r.n_samples, **r.metrics} for r in rows]
        )

    # ------------------------------------------------------------------ #
    def select_model(self, regime: str) -> BaseForecaster:
        """Return the champion model for ``regime`` (or the global fallback)."""
        name = self.best_by_regime.get(regime, self.fallback_model)
        if name is None or name not in self.models:
            name = next(iter(self.models))
        return self.models[name]

    def predict(self, regime: str, X: pd.DataFrame) -> np.ndarray:
        """Predict using the regime's champion model."""
        return self.select_model(regime).predict(X)

"""Walk-forward (rolling / expanding) validation for time-series models.

Random train/test splits leak future information into the past and are invalid
for sequential financial data. This module generates ordered, non-overlapping
test windows with an optional *purge/embargo* gap to prevent label leakage from
overlapping forward-return targets.

    |----- train -----||purge||-- test --|
                                 |----- train -----||purge||-- test --|
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.analytics.metrics import forecast_metrics
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardSplit:
    """Integer index ranges for one walk-forward fold."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


class WalkForwardValidator:
    """Generate folds and evaluate a model factory across them."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.validation

    # ------------------------------------------------------------------ #
    def split(self, n_samples: int) -> Iterator[WalkForwardSplit]:
        """Yield ordered train/test folds over ``n_samples`` rows."""
        train, test, step, purge = (
            self.cfg.train_period,
            self.cfg.test_period,
            self.cfg.step,
            self.cfg.purge,
        )
        expanding = self.cfg.scheme == "expanding"

        fold = 0
        start = 0
        while True:
            train_end = start + train
            test_start = train_end + purge
            test_end = test_start + test
            if test_end > n_samples:
                break
            train_start = 0 if expanding else start
            yield WalkForwardSplit(
                fold=fold,
                train_idx=np.arange(train_start, train_end),
                test_idx=np.arange(test_start, test_end),
            )
            fold += 1
            start += step

        if fold == 0:
            logger.warning(
                "No walk-forward folds produced: need >= %d rows, have %d",
                train + purge + test,
                n_samples,
            )

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        model_factory,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Fit a fresh model per fold and collect out-of-sample metrics.

        ``model_factory`` is a zero-arg callable returning an unfitted model
        implementing the :class:`BaseForecaster` interface.
        """
        records: list[dict] = []
        for sp in self.split(len(X)):
            Xtr, ytr = X.iloc[sp.train_idx], y.iloc[sp.train_idx]
            Xte, yte = X.iloc[sp.test_idx], y.iloc[sp.test_idx]
            model = model_factory()
            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            yt = np.asarray(yte, float)
            if len(preds) != len(yt):
                yt = yt[-len(preds):]
            metrics = forecast_metrics(yt, preds)
            metrics.update(
                fold=sp.fold,
                train_start=str(X.index[sp.train_idx[0]]),
                test_start=str(X.index[sp.test_idx[0]]),
                test_end=str(X.index[sp.test_idx[-1]]),
                n_test=len(sp.test_idx),
            )
            records.append(metrics)
            logger.info(
                "Fold %d | test %s..%s | RMSE=%.5f DA=%.3f",
                sp.fold, metrics["test_start"][:10], metrics["test_end"][:10],
                metrics["rmse"], metrics["directional_accuracy"],
            )
        return pd.DataFrame.from_records(records)

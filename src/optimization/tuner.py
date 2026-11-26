"""Optuna-based hyperparameter optimization for the XGBoost forecaster.

Uses a *time-series-aware* objective: each trial is scored by walk-forward RMSE
rather than a random split, keeping the search honest for sequential data.
"""

from __future__ import annotations

import pandas as pd

from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OptunaXGBTuner:
    """Search XGBoost hyperparameters with Optuna + walk-forward scoring."""

    def __init__(self, config: AppConfig, n_trials: int = 40) -> None:
        self.config = config
        self.n_trials = n_trials
        self.best_params: dict | None = None

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        from src.models.xgboost import XGBoostForecaster
        from src.validation import WalkForwardValidator

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        validator = WalkForwardValidator(self.config)
        results = validator.evaluate(lambda: XGBoostForecaster(params), X, y)
        if results.empty:
            return float("inf")
        return float(results["rmse"].mean())

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> dict:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.project.seed),
        )
        study.optimize(
            lambda t: self._objective(t, X, y), n_trials=self.n_trials, show_progress_bar=False
        )
        self.best_params = study.best_params
        logger.info("Best XGB params (RMSE=%.5f): %s", study.best_value, self.best_params)
        return self.best_params

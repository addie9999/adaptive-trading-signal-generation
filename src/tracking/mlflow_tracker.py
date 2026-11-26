"""Thin MLflow wrapper for experiment tracking and the model registry.

Designed to degrade gracefully: when ``tracking.enabled`` is false (or MLflow
is unavailable) every call becomes a no-op, so research code can always be
wrapped in tracking without branching.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Context-manager based MLflow tracker."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.tracking
        self._mlflow = None
        self._active = self.cfg.enabled
        if self._active:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.cfg.tracking_uri)
                mlflow.set_experiment(self.cfg.experiment_name)
                self._mlflow = mlflow
            except Exception as exc:  # noqa: BLE001
                logger.warning("MLflow unavailable, tracking disabled: %s", exc)
                self._active = False

    # ------------------------------------------------------------------ #
    @contextmanager
    def run(self, run_name: str, tags: dict[str, str] | None = None):
        """Open an MLflow run (or a no-op context when disabled)."""
        if not self._active:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name, tags=tags) as run:
            logger.info("MLflow run started: %s (%s)", run_name, run.info.run_id)
            yield run

    # ------------------------------------------------------------------ #
    def log_params(self, params: dict[str, Any]) -> None:
        if self._active:
            self._mlflow.log_params(_flatten(params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._active:
            clean = {k: float(v) for k, v in metrics.items() if _is_number(v)}
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self._active and Path(path).exists():
            self._mlflow.log_artifact(str(path))

    def log_dataframe(self, df, name: str) -> None:
        if not self._active:
            return
        tmp = Path("artifacts") / name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=True)
        self._mlflow.log_artifact(str(tmp))

    def log_model(self, model, artifact_path: str, registered_name: str | None = None) -> None:
        """Log a fitted forecaster, optionally registering it in the registry."""
        if not self._active:
            return
        try:
            local = Path("artifacts") / f"{artifact_path}.joblib"
            local.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(local))
            self._mlflow.log_artifact(str(local), artifact_path="model")
            if registered_name:
                self._register(local, registered_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log model: %s", exc)

    def _register(self, local_path: Path, name: str) -> None:
        # File-based stores support the model registry only with a DB backend;
        # we record the registration intent as a tag for portability.
        self._mlflow.set_tag("registered_model_name", name)
        logger.info("Recorded model registration intent: %s", name)


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)

"""Typed, layered configuration management.

Configuration is assembled by deep-merging YAML layers
(``default.yaml`` -> environment file -> explicit overrides) and is then
validated into strongly-typed Pydantic models. This gives the rest of the
codebase autocomplete-friendly, validated access to settings instead of
free-form dictionaries.

Example
-------
>>> cfg = load_config(env="research")
>>> cfg.data.tickers
['AAPL', 'MSFT', 'SPY', 'JPM', 'XOM']
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
ENV_PREFIX = "AQAF__"


# --------------------------------------------------------------------------- #
# Typed sections
# --------------------------------------------------------------------------- #
class ProjectConfig(BaseModel):
    name: str = "aqaf"
    seed: int = 42
    artifacts_dir: str = "artifacts"
    log_level: str = "INFO"


class DataConfig(BaseModel):
    provider: str = "yfinance"
    tickers: list[str] = Field(default_factory=lambda: ["AAPL"])
    start: str = "2015-01-01"
    end: str | None = None
    interval: str = "1d"
    cache_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    use_cache: bool = True


class CleaningConfig(BaseModel):
    fill_method: str = "ffill"
    drop_duplicates: bool = True
    outlier_method: str = "iqr"
    iqr_k: float = 3.0
    isolation_contamination: float = 0.01
    scaler: str = "robust"
    scale_features: bool = True


class FeaturesConfig(BaseModel):
    trend_windows: list[int] = Field(default_factory=lambda: [10, 20, 50, 200])
    momentum_windows: list[int] = Field(default_factory=lambda: [14])
    volatility_windows: list[int] = Field(default_factory=lambda: [14, 20])
    volume_windows: list[int] = Field(default_factory=lambda: [20])
    stat_windows: list[int] = Field(default_factory=lambda: [20])
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 10])
    target_horizon: int = 1


class RegimeConfig(BaseModel):
    n_states: int = 4
    covariance_type: str = "full"
    n_iter: int = 200
    features: list[str] = Field(
        default_factory=lambda: ["log_return", "rolling_vol_20", "volume_z"]
    )
    random_state: int = 42


class ModelsConfig(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["arima", "xgboost", "lstm", "transformer"])
    arima: dict[str, Any] = Field(default_factory=dict)
    xgboost: dict[str, Any] = Field(default_factory=dict)
    lstm: dict[str, Any] = Field(default_factory=dict)
    transformer: dict[str, Any] = Field(default_factory=dict)


class AdaptiveConfig(BaseModel):
    selection_metric: str = "directional_accuracy"
    min_samples_per_regime: int = 60


class AlphaConfig(BaseModel):
    buy_threshold: float = 0.0015
    sell_threshold: float = -0.0015
    confidence_scaler: float = 50.0


class RiskConfig(BaseModel):
    account_equity: float = 100_000.0
    risk_per_trade: float = 0.01
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    max_position_fraction: float = 0.25
    kelly_fraction_cap: float = 0.5
    sizing_method: str = "atr"


class ValidationConfig(BaseModel):
    scheme: str = "rolling"
    train_period: int = 756
    test_period: int = 252
    step: int = 252
    purge: int = 5


class BacktestingConfig(BaseModel):
    initial_cash: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 2.0
    position_size_fraction: float = 0.95


class DriftConfig(BaseModel):
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05
    adwin_delta: float = 0.002
    rolling_window: int = 60
    retrain_on_drift: bool = True


class TrackingConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "aqaf-research"


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_artifact_dir: str = "artifacts/models"


class DatabaseConfig(BaseModel):
    url: str = "postgresql+psycopg2://aqaf:aqaf@localhost:5432/aqaf"
    echo: bool = False


class AppConfig(BaseModel):
    """Root configuration object — the single typed handle to all settings."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    adaptive: AdaptiveConfig = Field(default_factory=AdaptiveConfig)
    alpha: AlphaConfig = Field(default_factory=AlphaConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    backtesting: BacktestingConfig = Field(default_factory=BacktestingConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: str) -> Any:
    """Best-effort coercion of an env-var string into a Python scalar/list."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _env_overrides() -> dict[str, Any]:
    """Translate ``AQAF__SECTION__KEY=value`` env vars into a nested dict."""
    overrides: dict[str, Any] = {}
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        path = env_key[len(ENV_PREFIX) :].lower().split("__")
        cursor = overrides
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _coerce(raw)
    return overrides


def load_config(
    env: str | None = None,
    overrides: dict[str, Any] | None = None,
    config_dir: str | Path | None = None,
) -> AppConfig:
    """Assemble and validate the application configuration.

    Layering order (later wins): ``default.yaml`` -> ``{env}.yaml`` ->
    environment variables (``AQAF__*``) -> explicit ``overrides`` dict.

    Parameters
    ----------
    env:
        Optional environment name (``"research"``/``"production"``). Falls
        back to the ``AQAF_ENV`` environment variable.
    overrides:
        Highest-priority explicit overrides, primarily for tests.
    config_dir:
        Directory containing the YAML layers (defaults to ``configs/``).
    """
    directory = Path(config_dir) if config_dir else CONFIG_DIR
    merged = _read_yaml(directory / "default.yaml")

    env = env or os.environ.get("AQAF_ENV")
    if env:
        merged = _deep_merge(merged, _read_yaml(directory / f"{env}.yaml"))

    merged = _deep_merge(merged, _env_overrides())
    if overrides:
        merged = _deep_merge(merged, overrides)

    return AppConfig.model_validate(merged)

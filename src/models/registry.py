"""Model registry / factory.

Maps config names to forecaster classes and constructs them with per-model
parameters from :class:`AppConfig`. Deep models are referenced lazily so that
importing the registry never imports torch.
"""

from __future__ import annotations

from src.models.base import BaseForecaster
from src.utils.config import AppConfig

# name -> "module:ClassName"  (lazy to avoid importing torch at module import)
_REGISTRY: dict[str, str] = {
    "arima": "src.models.arima:ARIMAForecaster",
    "xgboost": "src.models.xgboost:XGBoostForecaster",
    "lstm": "src.models.lstm:LSTMForecaster",
    "transformer": "src.models.transformer:TransformerForecaster",
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def _resolve(name: str) -> type[BaseForecaster]:
    import importlib

    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {available_models()}")
    module_path, cls_name = _REGISTRY[name].split(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def build_model(name: str, config: AppConfig) -> BaseForecaster:
    """Instantiate a forecaster by name using parameters from ``config``."""
    cls = _resolve(name)
    params = getattr(config.models, name, {}) or {}
    return cls(params=dict(params))

"""Model-serving layer: load bundles and produce predictions.

The service maintains an in-memory cache of loaded :class:`AlphaBundle`
objects keyed by ticker, lazily loading them from the artifact directory.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline import AlphaBundle, InferencePipeline, PredictionOutput, TrainingPipeline
from src.utils.config import AppConfig, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Stateful holder of trained bundles for the API process."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.artifact_dir = Path(self.config.api.model_artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, InferencePipeline] = {}

    # ------------------------------------------------------------------ #
    def _bundle_path(self, ticker: str) -> Path:
        return self.artifact_dir / f"{ticker.upper()}_bundle.joblib"

    def is_trained(self, ticker: str) -> bool:
        return ticker.upper() in self._cache or self._bundle_path(ticker).exists()

    def train(self, ticker: str) -> Path:
        """Train and persist a bundle for ``ticker``."""
        bundle = TrainingPipeline(self.config).run(ticker)
        path = bundle.save(self._bundle_path(ticker))
        self._cache[ticker.upper()] = InferencePipeline(bundle)
        return path

    def _get_pipeline(self, ticker: str) -> InferencePipeline:
        key = ticker.upper()
        if key in self._cache:
            return self._cache[key]
        path = self._bundle_path(ticker)
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model for {ticker}. Train it first (POST /train)."
            )
        pipeline = InferencePipeline(AlphaBundle.load(path))
        self._cache[key] = pipeline
        return pipeline

    def predict(self, ticker: str) -> PredictionOutput:
        return self._get_pipeline(ticker).predict(ticker)

    def loaded_tickers(self) -> list[str]:
        disk = {p.stem.replace("_bundle", "") for p in self.artifact_dir.glob("*_bundle.joblib")}
        return sorted(set(self._cache) | disk)

"""End-to-end orchestration tying every module into two pipelines.

``TrainingPipeline``  : data -> clean -> features -> regime -> multi-model
                        training -> adaptive selection -> persisted bundle.
``InferencePipeline`` : load bundle -> for a ticker produce regime, selected
                        model, alpha signal and risk decision.

The persisted :class:`AlphaBundle` is the single artefact the FastAPI service
loads, keeping training and serving perfectly consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.adaptive import AdaptiveModelSelector
from src.alpha import AlphaSignal, AlphaSignalGenerator, SignalType
from src.cleaning import DataCleaner
from src.data import MarketDataLoader
from src.features import FeatureEngineer
from src.features.engineer import TARGET_COL
from src.regime import HMMRegimeDetector
from src.risk import RiskDecision, RiskManager
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


@dataclass
class AlphaBundle:
    """Everything needed to serve predictions for one ticker."""

    ticker: str
    config: AppConfig
    regime_detector: HMMRegimeDetector
    selector: AdaptiveModelSelector
    feature_columns: list[str]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved alpha bundle for %s -> %s", self.ticker, path)
        return path

    @staticmethod
    def load(path: str | Path) -> AlphaBundle:
        return joblib.load(path)


# --------------------------------------------------------------------------- #
@dataclass
class PredictionOutput:
    """The full decision returned to an API caller."""

    ticker: str
    as_of: str
    regime: str
    selected_model: str
    signal: AlphaSignal
    risk: RiskDecision | None
    last_price: float

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "as_of": self.as_of,
            "market_regime": self.regime,
            "selected_model": self.selected_model,
            "signal": self.signal.signal.value,
            "confidence": round(self.signal.confidence, 4),
            "expected_return": round(self.signal.expected_return, 6),
            "expected_holding_period": self.signal.expected_holding_period,
            "stop_loss": self.risk.stop_loss if self.risk else None,
            "take_profit": self.risk.take_profit if self.risk else None,
            "position_size": self.risk.position_size if self.risk else None,
            "risk_reward_ratio": self.risk.risk_reward_ratio if self.risk else None,
            "last_price": round(self.last_price, 4),
        }


# --------------------------------------------------------------------------- #
class TrainingPipeline:
    """Train models for a ticker and persist a servable bundle."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        set_global_seed(config.project.seed)
        self.loader = MarketDataLoader(config)
        self.cleaner = DataCleaner(config)
        self.engineer = FeatureEngineer(config)

    def run(self, ticker: str) -> AlphaBundle:
        logger.info("=== Training pipeline for %s ===", ticker)
        raw = self.loader.load_one(ticker)
        clean = self.cleaner.clean(raw)

        # Regime model is fit on the full (clean) history.
        regime_detector = HMMRegimeDetector(self.config).fit(clean)
        regimes = regime_detector.predict(clean)["regime"]

        # Feature matrix + forward-return label.
        full = self.engineer.add_label(self.engineer.build_features(clean))
        feat_cols = self.engineer.feature_columns(full)
        data = full[feat_cols + [TARGET_COL]].dropna()
        X, y = data[feat_cols], data[TARGET_COL]

        # Train/holdout split for regime-conditional model selection.
        split = int(len(X) * 0.7)
        Xtr, Xho = X.iloc[:split], X.iloc[split:]
        ytr, yho = y.iloc[:split], y.iloc[split:]

        selector = AdaptiveModelSelector(self.config).fit(Xtr, ytr)
        scores = selector.evaluate_by_regime(Xho, yho, regimes)
        if not scores.empty:
            logger.info("Per-regime model scores:\n%s", scores.to_string(index=False))

        return AlphaBundle(
            ticker=ticker,
            config=self.config,
            regime_detector=regime_detector,
            selector=selector,
            feature_columns=feat_cols,
        )


# --------------------------------------------------------------------------- #
class InferencePipeline:
    """Serve a signal for a ticker from a trained :class:`AlphaBundle`."""

    def __init__(self, bundle: AlphaBundle) -> None:
        self.bundle = bundle
        self.config = bundle.config
        self.loader = MarketDataLoader(self.config)
        self.cleaner = DataCleaner(self.config)
        self.engineer = FeatureEngineer(self.config)
        self.signal_gen = AlphaSignalGenerator(self.config)
        self.risk = RiskManager(self.config)

    def predict(self, ticker: str | None = None) -> PredictionOutput:
        ticker = ticker or self.bundle.ticker
        raw = self.loader.load_one(ticker)
        clean = self.cleaner.clean(raw)

        regime = self.bundle.regime_detector.current_regime(clean)
        features = self.engineer.build_features(clean)
        X = features[self.bundle.feature_columns].dropna()
        if X.empty:
            raise ValueError(f"Insufficient data to build features for {ticker}")
        last_row = X.iloc[[-1]]

        model = self.bundle.selector.select_model(regime)
        # Sequence models need a window of recent rows; tabular need just one.
        x_input = X.iloc[-max(60, 1):] if getattr(model, "requires_sequences", False) else last_row
        expected_return = float(np.asarray(model.predict(x_input)).ravel()[-1])

        signal = self.signal_gen.generate(
            expected_return, regime=regime, model=model.name
        )

        last_price = float(clean["close"].iloc[-1])
        risk_decision = None
        if signal.signal != SignalType.HOLD:
            atr = self._latest_atr(features)
            risk_decision = self.risk.evaluate(
                signal=signal.signal,
                price=last_price,
                atr=atr,
                expected_return=expected_return,
                win_probability=signal.confidence,
            )

        return PredictionOutput(
            ticker=ticker,
            as_of=str(clean.index[-1].date()),
            regime=regime,
            selected_model=model.name,
            signal=signal,
            risk=risk_decision,
            last_price=last_price,
        )

    def _latest_atr(self, features: pd.DataFrame) -> float:
        atr_cols = [c for c in features.columns if c.startswith("atr_")]
        if atr_cols:
            val = features[atr_cols[0]].dropna()
            if not val.empty:
                return float(val.iloc[-1])
        # Fallback: 2% of price as a coarse volatility proxy.
        return float(features["close"].iloc[-1] * 0.02)

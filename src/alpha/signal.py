"""Convert model return forecasts into actionable trading signals.

The mapping is threshold-based on the *expected forward return*:

    expected_return >  buy_threshold  -> BUY
    expected_return <  sell_threshold -> SELL
    otherwise                         -> HOLD

A bounded confidence score is derived from the forecast magnitude via a tanh
squashing function, and an expected holding period is inferred from the
configured forecast horizon.
"""

from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass

from src.utils.config import AppConfig


class SignalType(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class AlphaSignal:
    """A single, fully-described trading signal."""

    signal: SignalType
    expected_return: float
    confidence: float            # 0..1
    expected_holding_period: int  # bars
    regime: str | None = None
    model: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        return d


class AlphaSignalGenerator:
    """Stateless mapping from a predicted return to an :class:`AlphaSignal`."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.alpha
        self.horizon = config.features.target_horizon

    def generate(
        self,
        expected_return: float,
        regime: str | None = None,
        model: str | None = None,
    ) -> AlphaSignal:
        if expected_return > self.cfg.buy_threshold:
            signal = SignalType.BUY
        elif expected_return < self.cfg.sell_threshold:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        confidence = self._confidence(expected_return)
        return AlphaSignal(
            signal=signal,
            expected_return=float(expected_return),
            confidence=confidence,
            expected_holding_period=self.horizon,
            regime=regime,
            model=model,
        )

    def _confidence(self, expected_return: float) -> float:
        """Squash forecast magnitude into a calibrated 0..1 confidence."""
        return float(math.tanh(abs(expected_return) * self.cfg.confidence_scaler))

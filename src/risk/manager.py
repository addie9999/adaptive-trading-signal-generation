"""Position sizing and stop/target placement.

Supported sizing methods (configurable):

* ``atr``        — risk a fixed % of equity, sized so the ATR stop distance
                   equals that risk budget.
* ``volatility`` — inverse-volatility (vol-target) sizing.
* ``kelly``      — fractional Kelly from forecast edge and win probability.
* ``fixed``      — a fixed fraction of equity.

All methods are clamped by ``max_position_fraction`` for sanity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.alpha.signal import SignalType
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """The risk layer's output for one signal."""

    position_size: float       # number of shares/units (signed by direction)
    notional: float            # absolute dollar exposure
    stop_loss: float           # price level
    take_profit: float         # price level
    risk_reward_ratio: float
    risk_amount: float         # dollars at risk to the stop
    method: str

    def to_dict(self) -> dict:
        return asdict(self)


class RiskManager:
    """Compute sizing and protective levels from price, ATR and edge."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.risk

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        signal: SignalType,
        price: float,
        atr: float,
        expected_return: float = 0.0,
        win_probability: float = 0.5,
        equity: float | None = None,
    ) -> RiskDecision:
        """Produce a :class:`RiskDecision` for a (non-HOLD) signal."""
        equity = equity if equity is not None else self.cfg.account_equity
        direction = 1 if signal == SignalType.BUY else -1

        stop, target = self._levels(price, atr, direction)
        size = self._size(price, atr, expected_return, win_probability, equity)
        size *= direction

        per_share_risk = abs(price - stop)
        risk_amount = abs(size) * per_share_risk
        reward = abs(target - price)
        rr = reward / per_share_risk if per_share_risk > 0 else 0.0

        return RiskDecision(
            position_size=round(size, 4),
            notional=round(abs(size) * price, 2),
            stop_loss=round(stop, 4),
            take_profit=round(target, 4),
            risk_reward_ratio=round(rr, 3),
            risk_amount=round(risk_amount, 2),
            method=self.cfg.sizing_method,
        )

    # ------------------------------------------------------------------ #
    def _levels(self, price: float, atr: float, direction: int) -> tuple[float, float]:
        stop = price - direction * self.cfg.atr_stop_mult * atr
        target = price + direction * self.cfg.atr_target_mult * atr
        return stop, target

    def _size(
        self,
        price: float,
        atr: float,
        expected_return: float,
        win_probability: float,
        equity: float,
    ) -> float:
        method = self.cfg.sizing_method
        if method == "atr":
            frac_units = self._atr_size(equity, atr)
        elif method == "volatility":
            frac_units = self._vol_size(equity, price, atr)
        elif method == "kelly":
            frac_units = self._kelly_size(equity, price, expected_return, win_probability)
        elif method == "fixed":
            frac_units = (equity * self.cfg.max_position_fraction) / price
        else:
            raise ValueError(f"Unknown sizing_method '{method}'")

        cap_units = (equity * self.cfg.max_position_fraction) / price
        return max(0.0, min(frac_units, cap_units))

    def _atr_size(self, equity: float, atr: float) -> float:
        """Units such that an ATR-stop loss equals ``risk_per_trade`` of equity."""
        risk_budget = equity * self.cfg.risk_per_trade
        per_unit_risk = self.cfg.atr_stop_mult * atr
        return risk_budget / per_unit_risk if per_unit_risk > 0 else 0.0

    def _vol_size(self, equity: float, price: float, atr: float) -> float:
        """Inverse-volatility sizing using ATR as the volatility proxy."""
        vol = atr / price if price > 0 else 0.0
        if vol <= 0:
            return 0.0
        target_vol = self.cfg.risk_per_trade
        notional = equity * min(target_vol / vol, self.cfg.max_position_fraction)
        return notional / price

    def _kelly_size(
        self, equity: float, price: float, expected_return: float, win_probability: float
    ) -> float:
        """Fractional Kelly from edge and win probability, capped for safety."""
        p = min(max(win_probability, 0.0), 1.0)
        b = abs(expected_return) if expected_return != 0 else 1e-4
        # Kelly fraction f* = p - (1-p)/b  for odds b.
        kelly = p - (1 - p) / b
        kelly = max(0.0, kelly) * self.cfg.kelly_fraction_cap
        kelly = min(kelly, self.cfg.max_position_fraction)
        return (equity * kelly) / price if price > 0 else 0.0

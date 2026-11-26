"""Drift detectors and an orchestrating monitor.

Three complementary signals:

* **Data drift** — Population Stability Index (PSI) and the two-sample
  Kolmogorov-Smirnov test compare a reference (training) distribution against
  a current (live) window, per feature.
* **Concept drift** — ADWIN (via ``river``) monitors the model's error stream
  for distributional change in the input->output relationship.
* **Performance drift** — rolling RMSE / Sharpe / win-rate degradation.

:class:`DriftMonitor` aggregates these into a single retrain decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.analytics.metrics import rmse, sharpe_ratio, win_rate
from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Data drift
# --------------------------------------------------------------------------- #
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two samples. >0.2 commonly signals a meaningful shift."""
    reference = np.asarray(reference, float)
    current = np.asarray(current, float)
    quantiles = np.quantile(reference, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=quantiles)[0] / max(len(reference), 1)
    cur_pct = np.histogram(current, bins=quantiles)[0] / max(len(current), 1)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_drift(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Two-sample KS test; returns ``(statistic, p_value)``."""
    from scipy.stats import ks_2samp

    stat, p = ks_2samp(np.asarray(reference, float), np.asarray(current, float))
    return float(stat), float(p)


def adwin_concept_drift(errors: np.ndarray, delta: float = 0.002) -> bool:
    """Run ADWIN over an error stream; ``True`` if a change point is detected."""
    try:
        from river.drift import ADWIN
    except Exception:  # noqa: BLE001 - optional dependency
        logger.warning("river not installed; skipping ADWIN concept-drift check")
        return False

    detector = ADWIN(delta=delta)
    drifted = False
    for e in np.asarray(errors, float):
        detector.update(float(e))
        if detector.drift_detected:
            drifted = True
    return drifted


# --------------------------------------------------------------------------- #
# Aggregated monitor
# --------------------------------------------------------------------------- #
@dataclass
class DriftReport:
    """Aggregated drift findings and the resulting retrain decision."""

    data_drift_features: dict[str, float] = field(default_factory=dict)
    ks_flags: dict[str, float] = field(default_factory=dict)
    concept_drift: bool = False
    performance: dict[str, float] = field(default_factory=dict)
    retrain_recommended: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "data_drift_features": self.data_drift_features,
            "ks_flags": self.ks_flags,
            "concept_drift": self.concept_drift,
            "performance": self.performance,
            "retrain_recommended": self.retrain_recommended,
            "reasons": self.reasons,
        }


class DriftMonitor:
    """Combine data, concept and performance drift into a retrain decision."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.drift

    def check(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        errors: np.ndarray | None = None,
        recent_returns: pd.Series | None = None,
    ) -> DriftReport:
        report = DriftReport()
        common = [c for c in reference.columns if c in current.columns]

        # --- data drift (PSI + KS) per feature ---
        for col in common:
            ref, cur = reference[col].dropna().values, current[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue
            psi = population_stability_index(ref, cur)
            if psi > self.cfg.psi_threshold:
                report.data_drift_features[col] = round(psi, 4)
            _, p = ks_drift(ref, cur)
            if p < self.cfg.ks_pvalue_threshold:
                report.ks_flags[col] = round(p, 4)

        # --- concept drift (ADWIN over error stream) ---
        if errors is not None and len(errors) > 0:
            report.concept_drift = adwin_concept_drift(errors, self.cfg.adwin_delta)

        # --- performance drift (rolling stats) ---
        if recent_returns is not None and len(recent_returns) > 0:
            report.performance = {
                "rolling_sharpe": round(sharpe_ratio(recent_returns), 3),
                "rolling_win_rate": round(win_rate(recent_returns), 3),
            }
        if errors is not None and len(errors) > 0:
            report.performance["rolling_rmse"] = round(rmse(np.zeros_like(errors), errors), 5)

        self._decide(report)
        return report

    def _decide(self, report: DriftReport) -> None:
        if report.data_drift_features:
            report.reasons.append(
                f"PSI breach on {len(report.data_drift_features)} feature(s)"
            )
        if report.ks_flags:
            report.reasons.append(f"KS shift on {len(report.ks_flags)} feature(s)")
        if report.concept_drift:
            report.reasons.append("ADWIN detected concept drift")
        report.retrain_recommended = bool(report.reasons) and self.cfg.retrain_on_drift
        if report.retrain_recommended:
            logger.warning("Drift detected -> retraining recommended: %s", report.reasons)

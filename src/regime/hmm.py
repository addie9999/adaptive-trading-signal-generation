"""Gaussian HMM regime detector.

A Gaussian HMM is fit on a small set of regime-descriptive features
(log returns, rolling volatility, volume z-score). The latent states are then
*mapped* to human-readable regime labels purely from their fitted statistics —
no hand-written rules classify the market, satisfying the requirement to rely
on the HMM for regime discovery.

State -> label mapping heuristic
--------------------------------
* The state with the highest mean volatility -> ``HIGH_VOLATILITY``.
* Among the rest, ranked by mean return: highest -> ``BULL``, lowest ->
  ``BEAR``, the remainder -> ``SIDEWAYS``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import AppConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeLabel(str, enum.Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class RegimeModelState:
    """Serialisable fitted-model artefacts."""

    feature_names: list[str]
    state_to_label: dict[int, str] = field(default_factory=dict)
    state_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    scaler: object = None  # fitted StandardScaler for the regime features


class HMMRegimeDetector:
    """Fit, label and predict market regimes with a Gaussian HMM."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.regime
        self.model = None  # hmmlearn GaussianHMM, set on fit
        self.state: RegimeModelState | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def build_regime_features(ohlcv: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
        """Derive the canonical regime feature block from an OHLCV frame."""
        out = pd.DataFrame(index=ohlcv.index)
        out["log_return"] = np.log(ohlcv["close"]).diff()
        out[f"rolling_vol_{vol_window}"] = out["log_return"].rolling(vol_window).std()
        vol = ohlcv["volume"].replace(0, np.nan)
        out["volume_z"] = (
            (vol - vol.rolling(vol_window).mean()) / vol.rolling(vol_window).std()
        )
        return out.dropna()

    # ------------------------------------------------------------------ #
    def fit(self, ohlcv: pd.DataFrame) -> HMMRegimeDetector:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        feats = self.build_regime_features(ohlcv)
        # Standardise: regime features differ by orders of magnitude (returns
        # ~1e-2 vs volume z-scores ~1), which makes a full covariance matrix
        # numerically singular. Standardising keeps the Cholesky well-posed.
        scaler = StandardScaler()
        x = scaler.fit_transform(feats[self.cfg.features].values)

        self.model = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.random_state,
            min_covar=1e-3,
        )
        self.model.fit(x)
        hidden = self.model.predict(x)

        self.state = RegimeModelState(
            feature_names=list(self.cfg.features), scaler=scaler
        )
        self._label_states(feats, hidden)
        logger.info(
            "HMM fitted: %d states mapped to %s",
            self.cfg.n_states,
            dict(self.state.state_to_label),
        )
        return self

    def _label_states(self, feats: pd.DataFrame, hidden: np.ndarray) -> None:
        assert self.state is not None
        vol_col = next(c for c in feats.columns if c.startswith("rolling_vol"))
        stats: dict[int, dict[str, float]] = {}
        for s in range(self.cfg.n_states):
            mask = hidden == s
            if mask.sum() == 0:
                stats[s] = {"mean_return": 0.0, "mean_vol": 0.0, "freq": 0.0}
                continue
            stats[s] = {
                "mean_return": float(feats["log_return"][mask].mean()),
                "mean_vol": float(feats[vol_col][mask].mean()),
                "freq": float(mask.mean()),
            }
        self.state.state_stats = stats

        # Highest-volatility state -> HIGH_VOLATILITY.
        hv = max(stats, key=lambda s: stats[s]["mean_vol"])
        labels = {hv: RegimeLabel.HIGH_VOLATILITY.value}

        remaining = sorted(
            (s for s in stats if s != hv),
            key=lambda s: stats[s]["mean_return"],
            reverse=True,
        )
        if remaining:
            labels[remaining[0]] = RegimeLabel.BULL.value
        if len(remaining) > 1:
            labels[remaining[-1]] = RegimeLabel.BEAR.value
        for s in remaining[1:-1]:
            labels[s] = RegimeLabel.SIDEWAYS.value
        # Any unassigned (e.g. n_states==2) default to sideways.
        for s in stats:
            labels.setdefault(s, RegimeLabel.SIDEWAYS.value)
        self.state.state_to_label = labels

    # ------------------------------------------------------------------ #
    def predict(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return a frame with hidden ``state`` and mapped ``regime`` columns."""
        if self.model is None or self.state is None:
            raise RuntimeError("HMMRegimeDetector must be fit before predict()")
        feats = self.build_regime_features(ohlcv)
        x = self.state.scaler.transform(feats[self.cfg.features].values)
        hidden = self.model.predict(x)
        out = pd.DataFrame(index=feats.index)
        out["state"] = hidden
        out["regime"] = [self.state.state_to_label[s] for s in hidden]
        return out

    def current_regime(self, ohlcv: pd.DataFrame) -> str:
        """Most recent regime label — what the live system acts on."""
        return str(self.predict(ohlcv)["regime"].iloc[-1])

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib

        joblib.dump({"model": self.model, "state": self.state, "cfg": self.cfg}, path)
        logger.info("Saved regime model -> %s", path)

    @classmethod
    def load(cls, path: str, config: AppConfig) -> HMMRegimeDetector:
        import joblib

        blob = joblib.load(path)
        obj = cls(config)
        obj.model = blob["model"]
        obj.state = blob["state"]
        return obj

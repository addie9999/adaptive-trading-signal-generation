"""SHAP explainability for tree-based forecasters.

Produces both global (summary) and local (force / waterfall) explanations and
persists them to the artifacts directory. Uses the fast ``TreeExplainer`` for
XGBoost; other model families can be supported via ``KernelExplainer`` at the
cost of speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """Compute and save SHAP explanations for a fitted model."""

    def __init__(self, model, output_dir: str = "artifacts/shap") -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._explainer = None
        self._shap_values = None
        self._X: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, max_samples: int = 500) -> ShapExplainer:
        """Build the explainer and compute SHAP values on a sample of ``X``."""
        import shap

        self._X = X.iloc[-max_samples:] if len(X) > max_samples else X
        estimator = getattr(self.model, "sklearn_model", self.model)
        try:
            self._explainer = shap.TreeExplainer(estimator)
        except Exception:  # noqa: BLE001 - fall back to model-agnostic explainer
            logger.info("TreeExplainer unavailable; using model-agnostic Explainer")
            self._explainer = shap.Explainer(self.model.predict, self._X)
        self._shap_values = self._explainer(self._X)
        return self

    # ------------------------------------------------------------------ #
    def global_importance(self) -> pd.Series:
        """Mean absolute SHAP value per feature (global importance)."""
        self._require_fitted()
        vals = np.abs(self._shap_values.values).mean(axis=0)
        return pd.Series(vals, index=self._X.columns).sort_values(ascending=False)

    def summary_plot(self, filename: str = "shap_summary.png") -> Path:
        import matplotlib.pyplot as plt
        import shap

        self._require_fitted()
        shap.summary_plot(self._shap_values, self._X, show=False)
        return self._save_current_fig(plt, filename)

    def waterfall_plot(self, row: int = -1, filename: str = "shap_waterfall.png") -> Path:
        import matplotlib.pyplot as plt
        import shap

        self._require_fitted()
        shap.plots.waterfall(self._shap_values[row], show=False)
        return self._save_current_fig(plt, filename)

    def force_plot(self, row: int = -1, filename: str = "shap_force.html") -> Path:
        """Save an interactive HTML force plot for a single prediction."""
        import shap

        self._require_fitted()
        fp = shap.plots.force(self._shap_values[row], matplotlib=False)
        path = self.output_dir / filename
        shap.save_html(str(path), fp)
        logger.info("Saved SHAP force plot -> %s", path)
        return path

    def save_report(self) -> dict[str, Path]:
        """Generate and persist the standard explanation bundle."""
        paths = {
            "summary": self.summary_plot(),
            "waterfall": self.waterfall_plot(),
        }
        importance = self.global_importance()
        csv_path = self.output_dir / "global_importance.csv"
        importance.to_csv(csv_path, header=["mean_abs_shap"])
        paths["importance_csv"] = csv_path
        logger.info("SHAP report saved to %s", self.output_dir)
        return paths

    # ------------------------------------------------------------------ #
    def _save_current_fig(self, plt, filename: str) -> Path:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP plot -> %s", path)
        return path

    def _require_fitted(self) -> None:
        if self._shap_values is None:
            raise RuntimeError("ShapExplainer.fit() must be called first")

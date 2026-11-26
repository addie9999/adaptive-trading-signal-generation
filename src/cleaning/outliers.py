"""Outlier detection strategies returning a boolean mask of inliers."""

from __future__ import annotations

import abc

import numpy as np
import pandas as pd


class OutlierDetector(abc.ABC):
    """Strategy interface: ``True`` marks a row to keep (an inlier)."""

    @abc.abstractmethod
    def inlier_mask(self, frame: pd.DataFrame) -> pd.Series:
        ...


class IQROutlierDetector(OutlierDetector):
    """Flag rows where any numeric column lies outside ``k`` * IQR fences.

    Robust to the heavy tails typical of financial returns; ``k`` controls how
    aggressive the trimming is (3.0 is conservative, 1.5 is the textbook box).
    """

    def __init__(self, k: float = 3.0, columns: list[str] | None = None) -> None:
        self.k = k
        self.columns = columns

    def inlier_mask(self, frame: pd.DataFrame) -> pd.Series:
        cols = self.columns or frame.select_dtypes("number").columns.tolist()
        mask = pd.Series(True, index=frame.index)
        for col in cols:
            q1, q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                continue
            lower, upper = q1 - self.k * iqr, q3 + self.k * iqr
            mask &= frame[col].between(lower, upper)
        return mask


class IsolationForestOutlierDetector(OutlierDetector):
    """Multivariate anomaly detection via scikit-learn's Isolation Forest."""

    def __init__(
        self,
        contamination: float = 0.01,
        columns: list[str] | None = None,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.columns = columns
        self.random_state = random_state

    def inlier_mask(self, frame: pd.DataFrame) -> pd.Series:
        from sklearn.ensemble import IsolationForest

        cols = self.columns or frame.select_dtypes("number").columns.tolist()
        x = frame[cols].replace([np.inf, -np.inf], np.nan).dropna()
        model = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        pred = model.fit_predict(x)  # +1 inlier, -1 outlier
        mask = pd.Series(True, index=frame.index)
        mask.loc[x.index] = pred == 1
        return mask

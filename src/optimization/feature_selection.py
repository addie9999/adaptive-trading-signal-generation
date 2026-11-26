"""Feature-selection utilities: RFE and permutation importance."""

from __future__ import annotations

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def recursive_feature_elimination(
    X: pd.DataFrame, y: pd.Series, n_features: int = 20
) -> list[str]:
    """Select ``n_features`` via scikit-learn RFE over a gradient-boosted base."""
    from sklearn.feature_selection import RFE
    from xgboost import XGBRegressor

    base = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1)
    selector = RFE(base, n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X.values, y.values)
    selected = X.columns[selector.support_].tolist()
    logger.info("RFE selected %d/%d features", len(selected), X.shape[1])
    return selected


def permutation_importance_selection(
    X: pd.DataFrame, y: pd.Series, top_k: int = 20, n_repeats: int = 5
) -> pd.Series:
    """Rank features by permutation importance on a held-out tail split."""
    from sklearn.inspection import permutation_importance
    from xgboost import XGBRegressor

    split = int(len(X) * 0.8)
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=-1)
    model.fit(Xtr.values, ytr.values)
    result = permutation_importance(
        model, Xte.values, yte.values, n_repeats=n_repeats, random_state=42
    )
    importance = pd.Series(result.importances_mean, index=X.columns).sort_values(
        ascending=False
    )
    return importance.head(top_k)

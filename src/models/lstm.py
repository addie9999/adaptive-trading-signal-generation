"""LSTM sequence forecaster (PyTorch backend, lazily imported).

Consumes overlapping windows of the engineered feature matrix and regresses
the forward return. Requires the optional ``[deep]`` extra (``torch``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models._torch_utils import _TorchTrainerMixin, make_sequences
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULTS = dict(
    sequence_length=30,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    epochs=30,
    batch_size=64,
    lr=1e-3,
)


try:
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self, n_features: int, p: dict) -> None:
            super().__init__()
            self._init_args = (n_features, p)
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=p["hidden_size"],
                num_layers=p["num_layers"],
                dropout=p["dropout"] if p["num_layers"] > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Sequential(
                nn.Linear(p["hidden_size"], p["hidden_size"] // 2),
                nn.ReLU(),
                nn.Linear(p["hidden_size"] // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])  # last timestep

except ImportError:
    pass


def _build_net(n_features: int, p: dict):
    return LSTMNet(n_features, p)


class LSTMForecaster(BaseForecaster, _TorchTrainerMixin):
    name = "lstm"
    requires_sequences = True

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.p = {**_DEFAULTS, **self.params}
        self._net = None
        self._device = None
        self._n_features = None

    def _log_epoch(self, epoch, epochs, loss):
        logger.info("LSTM epoch %d/%d - mse=%.6f", epoch, epochs, loss)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMForecaster:
        windows, targets = make_sequences(X, y, self.p["sequence_length"])
        self._n_features = X.shape[1]
        net = _build_net(self._n_features, self.p)
        self._net, self._device = self._train_loop(
            net, windows, targets, self.p["epochs"], self.p["batch_size"], self.p["lr"]
        )
        self._record_metadata(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        windows, _ = make_sequences(X, None, self.p["sequence_length"])
        return self._predict_batch(self._net, windows, self._device)

    def save(self, path: str) -> None:
        import torch

        self._check_fitted()
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "p": self.p,
                "n_features": self._n_features,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> LSTMForecaster:
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(blob["p"])
        obj._n_features = blob["n_features"]
        obj._net = _build_net(obj._n_features, obj.p)
        obj._net.load_state_dict(blob["state_dict"])
        obj._device = torch.device("cpu")
        obj.metadata = blob["metadata"]
        obj._fitted = True
        return obj

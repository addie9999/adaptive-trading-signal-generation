"""Transformer-encoder sequence forecaster (PyTorch backend, lazily imported).

A compact encoder-only Transformer with sinusoidal positional encoding applied
to windows of the engineered feature matrix. Requires the optional ``[deep]``
extra (``torch``).
"""

from __future__ import annotations

import math

import joblib  # noqa: F401  (kept for symmetry; torch handles persistence)
import numpy as np
import pandas as pd

from src.models._torch_utils import _TorchTrainerMixin, make_sequences
from src.models.base import BaseForecaster
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULTS = dict(
    sequence_length=30,
    d_model=64,
    nhead=4,
    num_layers=2,
    dropout=0.1,
    epochs=30,
    batch_size=64,
    lr=1e-3,
)


try:
    import torch
    import torch.nn as nn

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 512) -> None:
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class TransformerNet(nn.Module):
        def __init__(self, n_features: int, p: dict) -> None:
            super().__init__()
            self._init_args = (n_features, p)
            self.input_proj = nn.Linear(n_features, p["d_model"])
            self.pos = PositionalEncoding(p["d_model"])
            layer = nn.TransformerEncoderLayer(
                d_model=p["d_model"],
                nhead=p["nhead"],
                dim_feedforward=p["d_model"] * 4,
                dropout=p["dropout"],
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=p["num_layers"])
            self.head = nn.Sequential(
                nn.LayerNorm(p["d_model"]),
                nn.Linear(p["d_model"], 1),
            )

        def forward(self, x):
            h = self.pos(self.input_proj(x))
            h = self.encoder(h)
            return self.head(h[:, -1, :])  # pool last position

except ImportError:
    pass


def _build_net(n_features: int, p: dict):
    return TransformerNet(n_features, p)


class TransformerForecaster(BaseForecaster, _TorchTrainerMixin):
    name = "transformer"
    requires_sequences = True

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.p = {**_DEFAULTS, **self.params}
        self._net = None
        self._device = None
        self._n_features = None

    def _log_epoch(self, epoch, epochs, loss):
        logger.info("Transformer epoch %d/%d - mse=%.6f", epoch, epochs, loss)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerForecaster:
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
    def load(cls, path: str) -> TransformerForecaster:
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

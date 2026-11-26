"""Shared helpers for the PyTorch-based sequence models (LSTM, Transformer).

Kept separate so ``import src.models`` never imports torch; these helpers are
only touched when a deep model is actually instantiated and fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequences(
    X: pd.DataFrame, y: pd.Series | None, seq_len: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a tabular frame into overlapping ``(N, seq_len, n_features)`` windows.

    Window ending at row *t* predicts the target at row *t*. The first
    ``seq_len - 1`` rows cannot form a full window and are dropped, so the
    output has ``len(X) - seq_len + 1`` samples.
    """
    values = np.asarray(X.values, dtype=np.float32)
    n, _ = values.shape
    if n < seq_len:
        raise ValueError(f"Need >= {seq_len} rows to build sequences, got {n}")
    windows = np.stack([values[i : i + seq_len] for i in range(n - seq_len + 1)])
    targets = None
    if y is not None:
        targets = np.asarray(y.values, dtype=np.float32)[seq_len - 1 :]
    return windows, targets


class _TorchTrainerMixin:
    """Mixin providing a generic train loop shared by LSTM and Transformer."""

    def _train_loop(self, net, windows, targets, epochs, batch_size, lr):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        ds = TensorDataset(
            torch.from_numpy(windows), torch.from_numpy(targets).unsqueeze(-1)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        net.train()
        for epoch in range(epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                self._log_epoch(epoch + 1, epochs, total / len(ds))
        return net, device

    def _predict_batch(self, net, windows, device):
        import torch

        net.eval()
        with torch.no_grad():
            x = torch.from_numpy(windows).to(device)
            return net(x).cpu().numpy().ravel()

    def _log_epoch(self, epoch, epochs, loss):  # overridden to use module logger
        pass

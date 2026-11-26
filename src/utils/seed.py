"""Reproducibility helpers — seed every RNG the framework may touch."""

from __future__ import annotations

import os
import random

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if available) PyTorch / TensorFlow RNGs.

    Deep-learning libraries are imported lazily so the core framework does not
    depend on them being installed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass

    try:  # pragma: no cover - optional dependency
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:  # noqa: BLE001
        pass

    logger.debug("Global seed set to %d", seed)

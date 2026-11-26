"""Centralised logging configuration.

A single :func:`setup_logging` call configures the root logger with a
consistent format; modules obtain child loggers via :func:`get_logger`.
Logging is idempotent — repeated setup calls do not duplicate handlers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure the root logger.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"INFO"``) or numeric level.
    log_file:
        Optional path; when supplied, logs are also written to this file.
    """
    global _CONFIGURED

    root = logging.getLogger()
    numeric = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(numeric)

    if not _CONFIGURED:
        formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        root.addHandler(stream)

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

        _CONFIGURED = True

    # Tame noisy third-party loggers.
    for noisy in ("matplotlib", "urllib3", "yfinance", "hmmlearn", "py4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, ensuring logging is configured."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)

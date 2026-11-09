"""Logging and reproducibility helpers.

A single :func:`get_logger` gives every module a consistently formatted logger
that plays well with Kubernetes log collection (plain stdout, no files by
default). :func:`set_seed` centralises deterministic seeding.
"""

from __future__ import annotations

import logging
import os
import random

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("MARSWORLD_LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger (idempotent across calls)."""
    _configure_root()
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass

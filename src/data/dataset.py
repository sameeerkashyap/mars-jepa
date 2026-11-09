"""Torch datasets and dataloaders over generated trajectories.

A :class:`TrajectoryDataset` yields ``(context, target, forcing)`` tuples where

- ``context`` is the field at time ``t``            shape ``(C, H, W)``
- ``target``  is the field at time ``t + rollout``  shape ``(rollout, C, H, W)``
- ``forcing`` is the per-trajectory forcing vector  shape ``(forcing_dim,)``

Splitting into train/val/test and sub-sampling the training set (for the
data-efficiency experiments) is handled by :func:`make_dataloaders`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


class TrajectoryDataset(Dataset):
    """Expose (context, future-rollout, forcing) samples from trajectories.

    Parameters
    ----------
    fields:
        Array ``(N, L, C, H, W)``.
    forcing:
        Array ``(N, forcing_dim)``.
    rollout_steps:
        Number of future steps to return as the prediction target.
    """

    def __init__(self, fields: np.ndarray, forcing: np.ndarray, rollout_steps: int = 1):
        assert fields.ndim == 5, "expected (N, L, C, H, W)"
        self.fields = fields
        self.forcing = forcing
        self.rollout = rollout_steps
        self.L = fields.shape[1]
        if self.rollout >= self.L:
            raise ValueError("rollout_steps must be < trajectory_length")

        # Enumerate valid (trajectory, start-time) index pairs.
        self.index = [
            (n, t)
            for n in range(fields.shape[0])
            for t in range(self.L - self.rollout)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, t = self.index[i]
        context = torch.from_numpy(self.fields[n, t])                     # (C,H,W)
        target = torch.from_numpy(self.fields[n, t + 1 : t + 1 + self.rollout])  # (R,C,H,W)
        forcing = torch.from_numpy(self.forcing[n])                       # (F,)
        return context, target, forcing


def _split_indices(n: int, train_f: float, val_f: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * train_f))
    n_val = int(round(n * val_f))
    return (
        perm[:n_train],
        perm[n_train : n_train + n_val],
        perm[n_train + n_val :],
    )


def make_dataloaders(cfg: Config, npz_path: str | Path):
    """Build train/val/test dataloaders from a generated ``.npz`` dataset.

    Applies the configured split and ``data_fraction`` subsampling of the
    training trajectories (for data-efficiency experiments).
    """
    data = np.load(npz_path)
    fields, forcing = data["fields"], data["forcing"]

    train_idx, val_idx, test_idx = _split_indices(
        fields.shape[0], cfg.data.train_fraction, cfg.data.val_fraction, cfg.data.seed
    )

    # Data-efficiency subsampling on the TRAIN split only.
    if cfg.data.data_fraction < 1.0:
        keep = max(1, int(round(len(train_idx) * cfg.data.data_fraction)))
        train_idx = train_idx[:keep]

    def _ds(idx):
        return TrajectoryDataset(fields[idx], forcing[idx], cfg.train.rollout_steps)

    common = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    train_loader = DataLoader(_ds(train_idx), shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(_ds(val_idx), shuffle=False, **common)
    test_loader = DataLoader(_ds(test_idx), shuffle=False, **common)
    return train_loader, val_loader, test_loader

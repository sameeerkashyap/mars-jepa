"""Shared normalization base for the model arms."""

from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model):
    """Number of TRAINABLE parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


class BaseSeqModel(nn.Module):
    """Base class holding TRAIN-only normalization stats as buffers.

    Subclasses implement `rollout`.  All models operate on PHYSICAL inputs and
    return PHYSICAL T_s predictions (Kelvin); normalization happens internally.
    """

    needs_norm = True
    stochastic = False

    def __init__(self, stats, n_cell):
        super().__init__()
        self.n_cell = n_cell
        for k, v in stats.items():
            self.register_buffer(k, torch.tensor(float(v)))

    # --- normalization helpers (use registered buffers) ---
    def nTs(self, x):
        return (x - self.T_s_mean) / self.T_s_std

    def dTs(self, x):
        return x * self.T_s_std + self.T_s_mean

    def nS(self, x):
        return (x - self.S_mean) / self.S_std

    def nFa(self, x):
        return (x - self.F_a_mean) / self.F_a_std

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        raise NotImplementedError


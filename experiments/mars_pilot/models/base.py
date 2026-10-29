"""Common base + factory for the model arms."""

from __future__ import annotations

import numpy as np
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


def build_model(arm, stats, n_cell, arch, seed=0):
    """Factory.  `arm` in {gru, rssm, transformer, hybrid, linear, ebm_only,
    gru_noforce, jepa}."""
    torch.manual_seed(seed)
    if arm == "gru":
        from .gru import GRUModel
        return GRUModel(stats, n_cell, arch["gru"])
    if arm == "rssm":
        from .rssm import RSSMModel
        return RSSMModel(stats, n_cell, arch["rssm"])
    if arm == "transformer":
        from .transformer import TransformerModel
        return TransformerModel(stats, n_cell, arch["transformer"])
    if arm == "hybrid":
        from .hybrid import HybridModel
        return HybridModel(stats, n_cell, arch["hybrid"])
    if arm == "linear":
        from .linear import LinearARX
        return LinearARX(stats, n_cell, arch["linear"])
    if arm == "ebm_only":
        from .hybrid import EBMOnly
        return EBMOnly(stats, n_cell, arch["hybrid"])
    if arm == "gru_noforce":
        from .gru import GRUNoForce
        return GRUNoForce(stats, n_cell, arch["gru"])
    if arm == "jepa":
        from .jepa import JEPAModel
        return JEPAModel(stats, n_cell, arch["jepa"])
    raise ValueError(f"unknown arm {arm}")

"""Latent dynamics predictor.

Given the current latent ``z_t`` and the environmental forcing ``a_t`` (wind,
source strength, etc.), predict the next-step latent ``z_hat_(t+dt)``. This is
the learned *dynamics* of the world model and the component conditioned on
interventions for the intervention-response experiments.

A residual MLP is used so the predictor defaults to an identity map, which is a
strong prior for small time steps.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """Residual MLP predicting the next latent from (latent, forcing).

    Parameters
    ----------
    latent_dim:
        Size of the latent state.
    forcing_dim:
        Size of the forcing/action conditioning vector.
    hidden:
        Hidden width of the MLP.
    """

    def __init__(self, latent_dim: int, forcing_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + forcing_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        """Predict next latent given ``z`` ``(B, D)`` and ``forcing`` ``(B, F)``."""
        delta = self.net(torch.cat([z, forcing], dim=-1))
        return z + delta

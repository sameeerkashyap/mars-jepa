"""Context / target encoder for gridded Martian physical states.

A compact convolutional encoder maps a field ``(C, H, W)`` to a latent vector
``z`` of dimension ``latent_dim``. The same class is instantiated twice in the
PI-JEPA model: once as the trainable *context* encoder and once as the
EMA-updated *target* encoder (see :class:`marsworld.models.jepa.PIJEPA`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Downsampling CNN producing a global latent embedding.

    Parameters
    ----------
    in_channels:
        Input field channels.
    hidden_channels:
        Base channel width (doubled at each downsampling stage).
    latent_dim:
        Output embedding size.
    """

    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        c = hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1),  # H/2
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),        # H/4
            nn.GroupNorm(8, 2 * c),
            nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),    # H/8
            nn.GroupNorm(8, 4 * c),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(4 * c, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of fields ``(B, C, H, W)`` to latents ``(B, D)``."""
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)

"""Context / target encoder for gridded Martian physical states.

A compact convolutional encoder maps a field ``(C, H, W)`` to a latent vector
``z`` of dimension ``latent_dim``. The same class is instantiated twice in the
PI-JEPA model: once as the trainable *context* encoder and once as the
EMA-updated *target* encoder (see :class:`marsworld.models.jepa.PIJEPA`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm whose group count always divides ``channels``.

    Using ``gcd(channels, max_groups)`` avoids the silent constraint that
    ``channels`` be a multiple of 8 (which previously crashed for e.g.
    ``hidden_channels=12``).
    """
    groups = math.gcd(channels, max_groups) or 1
    return nn.GroupNorm(groups, channels)


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
            _group_norm(c),
            nn.SiLU(),
            nn.Conv2d(c, 2 * c, 3, stride=2, padding=1),        # H/4
            _group_norm(2 * c),
            nn.SiLU(),
            nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=1),    # H/8
            _group_norm(4 * c),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(4 * c, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of fields ``(B, C, H, W)`` to latents ``(B, D)``."""
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)

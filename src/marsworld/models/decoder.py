"""Physical-state decoder: latent -> interpretable field.

The decoder reconstructs a physical field ``(C, H, W)`` from a latent vector.
Decoding into *interpretable* physical space is what makes the physics loss
possible: PDE residuals and conservation diagnostics are computed on decoded
fields, not on arbitrary latent coordinates (per the research plan).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from marsworld.models.encoder import _group_norm


class ConvDecoder(nn.Module):
    """Upsampling CNN mapping a latent vector back to a field.

    Parameters
    ----------
    out_channels:
        Number of output field channels.
    hidden_channels:
        Base channel width (matched to the encoder).
    latent_dim:
        Input embedding size.
    out_size:
        Target spatial resolution (H == W).
    """

    def __init__(self, out_channels: int, hidden_channels: int, latent_dim: int, out_size: int):
        super().__init__()
        if out_size % 8 != 0:
            raise ValueError(
                f"out_size={out_size} must be divisible by 8 (the encoder "
                "downsamples by 2 three times). Previously this silently "
                "produced a mismatched output resolution."
            )
        c = hidden_channels
        self.out_size = out_size
        self.start = out_size // 8
        self.fc = nn.Linear(latent_dim, 4 * c * self.start * self.start)
        self.net = nn.Sequential(
            _group_norm(4 * c),
            nn.SiLU(),
            nn.ConvTranspose2d(4 * c, 2 * c, 4, stride=2, padding=1),  # x2
            _group_norm(2 * c),
            nn.SiLU(),
            nn.ConvTranspose2d(2 * c, c, 4, stride=2, padding=1),      # x4
            _group_norm(c),
            nn.SiLU(),
            nn.ConvTranspose2d(c, out_channels, 4, stride=2, padding=1),  # x8
        )
        self._c = c

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latents ``(B, D)`` to fields ``(B, C, H, W)``."""
        h = self.fc(z).view(-1, 4 * self._c, self.start, self.start)
        return self.net(h)

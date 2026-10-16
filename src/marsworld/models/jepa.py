"""The PI-JEPA world model assembling encoder, predictor and decoder.

Architecture (single-step; multi-step rollout handled by the trainer)::

    x_t ---(context encoder)---> z_t --+
                                        |-- predictor(z_t, a_t) --> z_hat_(t+1)
    x_(t+1) -(target encoder, EMA)-> z~_(t+1)              |
                                        JEPA loss: ||z_hat - sg(z~)||
                                                      |
                                        decoder(z_hat) --> x_hat_(t+1)
                                        forecast loss + PDE/conservation loss

The target encoder is a momentum (EMA) copy of the context encoder; its
parameters receive no gradient (stop-gradient targets), following JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from marsworld.config import ModelConfig
from marsworld.models.encoder import ConvEncoder
from marsworld.models.decoder import ConvDecoder
from marsworld.models.predictor import LatentPredictor


@dataclass
class JEPAOutput:
    """Container for a single prediction step's tensors."""

    z_context: torch.Tensor      # (B, D) encoded current state
    z_pred: torch.Tensor         # (B, D) predicted next latent
    z_target: torch.Tensor       # (B, D) EMA target latent (stop-grad)
    x_pred: torch.Tensor         # (B, C, H, W) decoded predicted field


class PIJEPA(nn.Module):
    """Physics-Informed Joint Embedding Predictive Architecture.

    Parameters
    ----------
    cfg:
        :class:`marsworld.config.ModelConfig` instance.
    grid_size:
        Spatial resolution of the fields (needed by the decoder).
    """

    def __init__(self, cfg: ModelConfig, grid_size: int):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = ConvEncoder(cfg.in_channels, cfg.hidden_channels, cfg.latent_dim)
        self.target_encoder = ConvEncoder(cfg.in_channels, cfg.hidden_channels, cfg.latent_dim)
        self.predictor = LatentPredictor(cfg.latent_dim, cfg.forcing_dim, cfg.predictor_hidden)
        self.decoder = ConvDecoder(cfg.in_channels, cfg.hidden_channels, cfg.latent_dim, grid_size)

        # Initialise target encoder as a frozen copy of the context encoder.
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # --- core forward paths ------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a field with the trainable context encoder."""
        return self.context_encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a field with the frozen EMA target encoder (stop-grad)."""
        return self.target_encoder(x)

    def predict_latent(self, z: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        """Advance a latent one step under the given forcing."""
        return self.predictor(z, forcing)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent back to an interpretable physical field."""
        return self.decoder(z)

    def forward(self, x_t: torch.Tensor, x_next: torch.Tensor, forcing: torch.Tensor) -> JEPAOutput:
        """Run a single predictive step and return all tensors for the loss."""
        z_context = self.encode(x_t)
        z_pred = self.predict_latent(z_context, forcing)
        z_target = self.encode_target(x_next)
        x_pred = self.decode(z_pred)
        return JEPAOutput(z_context=z_context, z_pred=z_pred, z_target=z_target, x_pred=x_pred)

    # --- EMA update --------------------------------------------------------
    @torch.no_grad()
    def update_target_encoder(self, decay: float | None = None) -> None:
        """Momentum-update the target encoder toward the context encoder."""
        m = self.cfg.ema_decay if decay is None else decay
        for tgt, src in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tgt.mul_(m).add_(src, alpha=1.0 - m)

    def num_parameters(self) -> Dict[str, int]:
        """Return trainable parameter counts (for compute-matched reporting)."""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "context_encoder": count(self.context_encoder),
            "predictor": count(self.predictor),
            "decoder": count(self.decoder),
            "total_trainable": count(self),
        }

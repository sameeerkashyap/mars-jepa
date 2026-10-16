"""Composite objective combining JEPA, forecast and physics terms.

Each term is weighted so that ablations (plan's models A/B/C) are produced by
flipping weights in config:

- model A (simple field predictor): jepa_weight=0, physics disabled
- model B (JEPA latent predictor):  physics disabled
- model C (physics-informed JEPA):  all terms active
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from marsworld.config import PhysicsConfig, TrainConfig
from marsworld.models.jepa import JEPAOutput
from marsworld.physics.losses import pde_residual_loss, mass_conservation_loss


@dataclass
class CompositeLoss:
    """Weighted sum of JEPA alignment, forecast and physics losses."""

    train_cfg: TrainConfig
    physics_cfg: PhysicsConfig

    def __call__(
        self,
        out: JEPAOutput,
        x_t: torch.Tensor,
        x_next_true: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss terms.

        Returns a dict with ``total`` and every component (detached scalars for
        the components are kept attached for logging convenience; only ``total``
        is used for backprop).
        """
        # JEPA alignment: predicted latent vs stop-grad target latent.
        jepa = F.smooth_l1_loss(out.z_pred, out.z_target.detach())

        # Forecast reconstruction in physical space.
        forecast = F.mse_loss(out.x_pred, x_next_true)

        total = self.train_cfg.jepa_weight * jepa + self.train_cfg.forecast_weight * forecast
        terms = {"jepa": jepa, "forecast": forecast}

        if self.physics_cfg.enabled:
            pde = pde_residual_loss(
                x_t, out.x_pred,
                diffusivity=self.physics_cfg.diffusivity,
                dt=self.physics_cfg.dt,
            )
            cons = mass_conservation_loss(x_t, out.x_pred)
            total = total + self.physics_cfg.pde_weight * pde
            total = total + self.physics_cfg.conservation_weight * cons
            terms["pde"] = pde
            terms["conservation"] = cons

        terms["total"] = total
        return terms

"""Differentiable physics losses evaluated on decoded physical fields.

These implement the physics-informed term of the objective

    L = lambda_JEPA L_JEPA + lambda_forecast L_forecast + lambda_PDE L_PDE

The PDE residual penalises decoded predictions that violate the discretised
governing equation (here the diffusion operator, matching the synthetic
temperature solver). The conservation loss penalises spurious creation/loss of
the integrated quantity when there is no net source.

All operators use periodic (torch.roll) boundaries to match the solver.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _laplacian(f: torch.Tensor, dx: float) -> torch.Tensor:
    """5-point periodic Laplacian for ``(B, C, H, W)`` tensors."""
    return (
        torch.roll(f, 1, -2) + torch.roll(f, -1, -2)
        + torch.roll(f, 1, -1) + torch.roll(f, -1, -1)
        - 4.0 * f
    ) / dx ** 2


def pde_residual_loss(
    x_t: torch.Tensor,
    x_next_pred: torch.Tensor,
    diffusivity: float,
    dt: float,
    dx: float = None,  # type: ignore[assignment]
) -> torch.Tensor:
    """Penalise violation of the (diffusion) governing equation.

    Residual: ``(x_next - x_t)/dt - kappa * laplacian(x_t)``.

    Note the advection term is *not* included here because the predictor is
    conditioned on the (unknown-to-this-loss) wind; the diffusion residual is a
    conservative physical prior that is always valid. This mirrors the plan's
    guidance to use a justified, stable discretisation.

    Parameters
    ----------
    x_t:
        Current field ``(B, C, H, W)``.
    x_next_pred:
        Decoded predicted next field ``(B, C, H, W)``.
    """
    if dx is None:
        dx = 1.0  # unit cell spacing, matching the solver default
    dT_dt = (x_next_pred - x_t) / dt
    residual = dT_dt - diffusivity * _laplacian(x_t, dx)
    return residual.pow(2).mean()


def mass_conservation_loss(x_t: torch.Tensor, x_next_pred: torch.Tensor) -> torch.Tensor:
    """Penalise change in the spatially-integrated quantity.

    For a closed periodic domain with diffusion-only transport, the domain
    integral is conserved. Advection on a periodic grid also conserves the
    integral, so this is a valid soft constraint for the temperature proxy.
    """
    integral_t = x_t.mean(dim=(-1, -2))
    integral_next = x_next_pred.mean(dim=(-1, -2))
    return F.mse_loss(integral_next, integral_t)

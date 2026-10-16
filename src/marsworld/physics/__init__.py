"""Physics-informed diagnostics and losses computed on decoded fields."""

from marsworld.physics.losses import pde_residual_loss, mass_conservation_loss

__all__ = ["pde_residual_loss", "mass_conservation_loss"]

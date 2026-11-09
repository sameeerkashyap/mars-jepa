"""Physics-informed diagnostics and losses computed on decoded fields."""

from physics.losses import mass_conservation_loss, pde_residual_loss

__all__ = ["pde_residual_loss", "mass_conservation_loss"]

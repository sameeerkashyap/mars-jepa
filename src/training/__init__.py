"""Training loop and composite loss for the PI-JEPA world model."""

from training.losses import CompositeLoss
from training.trainer import Trainer

__all__ = ["CompositeLoss", "Trainer"]

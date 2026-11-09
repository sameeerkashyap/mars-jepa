"""Data generation and loading for MarsWorld.

Exposes the synthetic PDE solver and the torch dataset wrapper.
"""

from data.dataset import TrajectoryDataset, make_dataloaders
from data.solver import HeatDiffusionSolver, generate_dataset

__all__ = [
    "HeatDiffusionSolver",
    "generate_dataset",
    "TrajectoryDataset",
    "make_dataloaders",
]

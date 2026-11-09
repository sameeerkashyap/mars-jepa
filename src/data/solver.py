"""Synthetic ground-truth solver for the single-parameter proof of concept.

The initial MarsWorld parameter is **surface temperature**. We generate
reproducible trajectories from a 2-D heat (diffusion + advection + forcing)
equation on a periodic grid:

    dT/dt = kappa * laplacian(T) - u . grad(T) + S(x, y, t)

where ``S`` is a controllable insolation-like source term. This gives a
*known, reproducible source of truth* (Stage A of the research plan's data
strategy) and lets us compute exact PDE residuals for the physics loss.

The solver is deliberately framework-light (pure NumPy) so data generation can
run on a cheap CPU pod, decoupled from GPU training.

The same interface (``generate_dataset``) can later be swapped for a dust
advection-diffusion solver by implementing another ``*Solver`` class; the
rest of the pipeline is parameter-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class HeatDiffusionSolver:
    """Finite-difference solver for 2-D advection-diffusion with forcing.

    Parameters
    ----------
    grid_size:
        Number of cells per side of the square periodic domain.
    dt:
        Explicit time step. Must satisfy the stability condition relative to
        ``diffusivity`` and grid spacing (checked in :meth:`__post_init__`).
    diffusivity:
        Thermal diffusivity ``kappa``.
    dx:
        Grid spacing in cell units (defaults to ``1.0``, i.e. one cell).
    """

    grid_size: int = 32
    dt: float = 0.1
    diffusivity: float = 0.15
    dx: float = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dx is None:
            self.dx = 1.0
        # Explicit-scheme stability (von Neumann) for 2-D diffusion.
        stable_dt = self.dx ** 2 / (4.0 * max(self.diffusivity, 1e-8))
        if self.dt > stable_dt:
            raise ValueError(
                f"dt={self.dt} violates stability limit {stable_dt:.4g}. "
                "Reduce dt or diffusivity, or increase grid_size."
            )

    # --- differential operators (periodic boundaries) --------------------
    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """5-point periodic Laplacian."""
        return (
            np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1)
            - 4.0 * f
        ) / self.dx ** 2

    def grad(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Central-difference periodic gradient (df/dx, df/dy)."""
        fx = (np.roll(f, -1, 1) - np.roll(f, 1, 1)) / (2 * self.dx)
        fy = (np.roll(f, -1, 0) - np.roll(f, 1, 0)) / (2 * self.dx)
        return fx, fy

    def step(
        self,
        T: np.ndarray,
        u: Tuple[float, float],
        source: np.ndarray,
    ) -> np.ndarray:
        """Advance the temperature field one explicit Euler step."""
        fx, fy = self.grad(T)
        advection = u[0] * fx + u[1] * fy
        dT = self.diffusivity * self.laplacian(T) - advection + source
        return T + self.dt * dT


def _random_source(rng: np.random.Generator, grid: int, amp: float) -> np.ndarray:
    """Build a smooth Gaussian 'insolation' source term."""
    yy, xx = np.mgrid[0:grid, 0:grid] / grid
    cx, cy = rng.uniform(0.2, 0.8, size=2)
    sigma = rng.uniform(0.1, 0.25)
    bump = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return amp * bump


def generate_dataset(
    n_trajectories: int,
    trajectory_length: int,
    grid_size: int,
    dt: float,
    diffusivity: float,
    seed: int = 0,
    substeps: int = 8,
) -> Dict[str, np.ndarray]:
    """Generate a batch of reproducible trajectories with varied forcing.

    Each trajectory varies: initial field, advection wind ``u``, and source
    amplitude -- these constitute the controllable *forcing* ``a_t`` that the
    predictor is conditioned on, enabling the forcing-generalisation and
    intervention-response experiments.

    Returns
    -------
    dict with keys:
        ``fields``  float32 array ``(N, L, 1, H, W)`` temperature trajectories
        ``forcing`` float32 array ``(N, 2)`` per-trajectory wind (ux, uy)
        ``source_amp`` float32 array ``(N,)`` source amplitude
    """
    np.random.default_rng(seed)
    solver = HeatDiffusionSolver(grid_size=grid_size, dt=dt, diffusivity=diffusivity)

    fields = np.empty((n_trajectories, trajectory_length, 1, grid_size, grid_size), np.float32)
    forcing = np.empty((n_trajectories, 2), np.float32)

    for n in range(n_trajectories):
        traj_rng = np.random.default_rng(seed * 100003 + n)
        # Smooth random initial condition.
        noise = traj_rng.normal(size=(grid_size, grid_size))
        T = _smooth(noise, iters=8)
        T = (T - T.mean()) / (T.std() + 1e-6)

        u = traj_rng.uniform(-0.5, 0.5, size=2)
        amp = traj_rng.uniform(0.0, 0.3)
        source = _random_source(traj_rng, grid_size, amp)

        fields[n, 0, 0] = T
        for t in range(1, trajectory_length):
            # Advance `substeps` internal steps per stored frame so that the
            # frame-to-frame change is large enough to be a non-trivial target.
            for _ in range(max(1, substeps)):
                T = solver.step(T, (float(u[0]), float(u[1])), source)
            fields[n, t, 0] = T

        forcing[n] = u

    return {
        "fields": fields,
        "forcing": forcing,
        "meta": np.array([grid_size, trajectory_length, dt, diffusivity], np.float32),
    }


def _smooth(f: np.ndarray, iters: int = 4) -> np.ndarray:
    """Cheap box smoothing to create spatially-correlated initial conditions."""
    for _ in range(iters):
        f = (
            f
            + np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1)
        ) / 5.0
    return f


def save_dataset(data: Dict[str, np.ndarray], path: str | Path) -> None:
    """Persist a generated dataset to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)

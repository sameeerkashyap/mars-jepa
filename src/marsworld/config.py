"""Typed configuration objects for the MarsWorld training framework.

Configuration is defined with dataclasses so that every knob is discoverable,
type-checked, and documentable. A YAML file (see ``configs/default.yaml``) is
parsed into these dataclasses via :func:`load_config`, which keeps the training
entry points free of ad-hoc dictionary access.

The design intentionally separates concerns:

- :class:`DataConfig`    how synthetic trajectories are generated/loaded
- :class:`ModelConfig`   architecture dimensions of the PI-JEPA model
- :class:`PhysicsConfig` which physical constraints are active and their weights
- :class:`TrainConfig`   optimisation and schedule
- :class:`Config`        the top-level container tying them together
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DataConfig:
    """Controls synthetic data generation and loading.

    Attributes
    ----------
    parameter:
        Physical field to model. ``"temperature"`` is the supported proof of
        concept; ``"dust"`` is reserved for the next milestone.
    grid_size:
        Spatial resolution of the square field (``grid_size`` x ``grid_size``).
    n_trajectories:
        Number of independent trajectories to simulate.
    trajectory_length:
        Number of time steps per trajectory (including the initial state).
    dt:
        Solver time step (dimensionless / normalised units).
    substeps:
        Number of solver integration steps taken between two *stored* frames.
        Increasing this enlarges the per-frame change so the task is not
        dominated by persistence (``x_{t+1} ~= x_t``), making the forecasting
        benchmark meaningful. Does not affect numerical stability (each internal
        step still uses ``dt``).
    diffusivity:
        Base thermal diffusivity used by the solver.
    seed:
        Base RNG seed for reproducible trajectory generation.
    data_dir:
        Directory where generated ``.npz`` shards are written / read.
    train_fraction / val_fraction:
        Split ratios; the remainder is used for the test set.
    data_fraction:
        Fraction of the *training* trajectories actually used, enabling the
        data-efficiency experiments described in the research plan.
    """

    parameter: str = "temperature"
    grid_size: int = 32
    n_trajectories: int = 256
    trajectory_length: int = 24
    dt: float = 0.1
    substeps: int = 8
    diffusivity: float = 0.15
    seed: int = 0
    data_dir: str = "data"
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    data_fraction: float = 1.0


@dataclass
class ModelConfig:
    """Architecture dimensions of the PI-JEPA model.

    Attributes
    ----------
    in_channels:
        Number of input field channels (1 for a single parameter).
    hidden_channels:
        Width of the convolutional encoder/decoder.
    latent_dim:
        Dimensionality of the JEPA latent vector ``z_t``.
    forcing_dim:
        Number of scalar forcing/action inputs conditioning the predictor.
    predictor_hidden:
        Hidden width of the latent dynamics predictor MLP.
    ema_decay:
        Momentum used to update the target encoder (stop-gradient EMA).
    predict_residual:
        If True, the decoder predicts the field *increment* and the next field is
        ``x_t + decode(z)``. This makes "do nothing" equal to persistence and is
        essential for small time steps (see audit notes). Default True.
    """

    in_channels: int = 1
    hidden_channels: int = 32
    latent_dim: int = 128
    forcing_dim: int = 2
    predictor_hidden: int = 256
    ema_decay: float = 0.996
    predict_residual: bool = True


@dataclass
class PhysicsConfig:
    """Weights and switches for the physics-informed objective.

    Setting ``pde_weight`` and ``conservation_weight`` to 0 recovers a pure
    JEPA baseline (model B in the research plan). Enabling them recovers the
    physics-informed model (model C).
    """

    enabled: bool = True
    pde_weight: float = 0.1
    conservation_weight: float = 0.05
    diffusivity: float = 0.15
    dt: float = 0.1
    substeps: int = 8  # must match DataConfig.substeps so the residual uses the
    #                    true frame spacing (effective dt = dt * substeps)

    @property
    def frame_dt(self) -> float:
        """Effective time between two stored frames."""
        return self.dt * max(1, self.substeps)


@dataclass
class TrainConfig:
    """Optimisation schedule and loss weights."""

    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-5
    jepa_weight: float = 1.0
    forecast_weight: float = 1.0
    rollout_steps: int = 1
    num_workers: int = 4
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    log_every: int = 10
    checkpoint_dir: str = "checkpoints"
    seed: int = 0


@dataclass
class Config:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dict (useful for logging / checkpoints)."""
        return asdict(self)


def _merge(section_cls, overrides: Dict[str, Any]):
    """Instantiate ``section_cls`` from a (possibly partial) override dict."""
    if not overrides:
        return section_cls()
    valid = {f for f in section_cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown keys for {section_cls.__name__}: {sorted(unknown)}")
    return section_cls(**overrides)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file.

    Missing sections/keys fall back to the dataclass defaults, so a YAML file
    only needs to specify what it wants to override.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    return Config(
        data=_merge(DataConfig, raw.get("data", {})),
        model=_merge(ModelConfig, raw.get("model", {})),
        physics=_merge(PhysicsConfig, raw.get("physics", {})),
        train=_merge(TrainConfig, raw.get("train", {})),
    )

"""End-to-end smoke tests that run on CPU in seconds.

These guard the public contracts of each component: solver stability, dataset
shapes, model forward pass, loss computation, and a single training step.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from marsworld.config import Config, DataConfig, ModelConfig, PhysicsConfig, TrainConfig
from marsworld.data.solver import generate_dataset
from marsworld.data.dataset import TrajectoryDataset
from marsworld.models.jepa import PIJEPA
from marsworld.training.losses import CompositeLoss
from marsworld.physics.losses import pde_residual_loss, mass_conservation_loss


GRID = 16


def _tiny_data():
    return generate_dataset(
        n_trajectories=4, trajectory_length=6, grid_size=GRID,
        dt=0.1, diffusivity=0.15, seed=1,
    )


def test_solver_shapes_and_finiteness():
    data = _tiny_data()
    assert data["fields"].shape == (4, 6, 1, GRID, GRID)
    assert np.isfinite(data["fields"]).all()
    assert data["forcing"].shape == (4, 2)


def test_solver_stability_guard():
    from marsworld.data.solver import HeatDiffusionSolver

    with pytest.raises(ValueError):
        HeatDiffusionSolver(grid_size=GRID, dt=10.0, diffusivity=5.0)


def test_dataset_sample_shapes():
    data = _tiny_data()
    ds = TrajectoryDataset(data["fields"], data["forcing"], rollout_steps=2)
    ctx, tgt, forcing = ds[0]
    assert ctx.shape == (1, GRID, GRID)
    assert tgt.shape == (2, 1, GRID, GRID)
    assert forcing.shape == (2,)


def test_model_forward_and_param_count():
    cfg = ModelConfig(hidden_channels=8, latent_dim=32, predictor_hidden=64)
    model = PIJEPA(cfg, grid_size=GRID)
    x = torch.randn(2, 1, GRID, GRID)
    out = model(x, x, torch.randn(2, cfg.forcing_dim))
    assert out.z_pred.shape == (2, 32)
    assert out.x_pred.shape == (2, 1, GRID, GRID)
    counts = model.num_parameters()
    assert counts["total_trainable"] > 0


def test_physics_losses_nonnegative():
    x = torch.randn(2, 1, GRID, GRID)
    xn = x + 0.01 * torch.randn_like(x)
    assert pde_residual_loss(x, xn, 0.15, 0.1).item() >= 0
    assert mass_conservation_loss(x, xn).item() >= 0


def test_ema_target_update_changes_target_encoder():
    cfg = ModelConfig(hidden_channels=8, latent_dim=32, predictor_hidden=64)
    model = PIJEPA(cfg, grid_size=GRID)
    # Perturb context encoder so EMA has something to move toward.
    with torch.no_grad():
        for p in model.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    before = next(model.target_encoder.parameters()).clone()
    model.update_target_encoder(decay=0.5)
    after = next(model.target_encoder.parameters())
    assert not torch.equal(before, after)


def test_single_training_step_reduces_or_runs():
    cfg = Config(
        data=DataConfig(grid_size=GRID),
        model=ModelConfig(hidden_channels=8, latent_dim=32, predictor_hidden=64),
        physics=PhysicsConfig(enabled=True),
        train=TrainConfig(device="cpu"),
    )
    model = PIJEPA(cfg.model, grid_size=GRID)
    loss_fn = CompositeLoss(cfg.train, cfg.physics)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(4, 1, GRID, GRID)
    xn = x + 0.01 * torch.randn_like(x)
    forcing = torch.randn(4, cfg.model.forcing_dim)

    out = model(x, xn, forcing)
    terms = loss_fn(out, x, xn)
    assert "total" in terms and "pde" in terms and "jepa" in terms
    opt.zero_grad()
    terms["total"].backward()
    opt.step()
    assert torch.isfinite(terms["total"]).all()

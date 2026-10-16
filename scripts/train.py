#!/usr/bin/env python
"""Train the PI-JEPA world model.

Usage
-----
    python scripts/train.py --config configs/default.yaml

Expects the dataset produced by ``scripts/generate_data.py`` to exist at
``<data_dir>/<parameter>.npz``. Designed to be the container entrypoint for a
Kubernetes training Job.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from marsworld.config import load_config
from marsworld.data.dataset import make_dataloaders
from marsworld.data.solver import generate_dataset, save_dataset
from marsworld.models.jepa import PIJEPA
from marsworld.training.trainer import Trainer
from marsworld.utils.logging import get_logger, set_seed

logger = get_logger("train")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--auto-generate",
        action="store_true",
        help="Generate the dataset on the fly if it does not exist.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.train.seed)

    npz_path = Path(cfg.data.data_dir) / f"{cfg.data.parameter}.npz"
    if not npz_path.exists():
        if not args.auto_generate:
            raise FileNotFoundError(
                f"{npz_path} not found. Run scripts/generate_data.py first "
                "or pass --auto-generate."
            )
        logger.info("dataset missing; auto-generating -> %s", npz_path)
        data = generate_dataset(
            cfg.data.n_trajectories, cfg.data.trajectory_length, cfg.data.grid_size,
            cfg.data.dt, cfg.data.diffusivity, cfg.data.seed,
        )
        save_dataset(data, npz_path)

    train_loader, val_loader, test_loader = make_dataloaders(cfg, npz_path)
    logger.info("batches | train=%d val=%d test=%d",
                len(train_loader), len(val_loader), len(test_loader))

    model = PIJEPA(cfg.model, grid_size=cfg.data.grid_size)
    trainer = Trainer(model, cfg)
    trainer.fit(train_loader, val_loader)

    test_metrics = trainer.evaluate(test_loader)
    logger.info("[test] %s", " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()

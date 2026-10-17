#!/usr/bin/env python
"""Generate synthetic Mars temperature trajectories.

Usage
-----
    python scripts/generate_data.py --config configs/default.yaml

Writes a single compressed ``.npz`` shard into ``<data_dir>/<parameter>.npz``.
This is the CPU-only Stage A step and can run as a separate Kubernetes Job
before training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from marsworld.config import load_config
from marsworld.data.solver import generate_dataset, save_dataset
from marsworld.utils.logging import get_logger, set_seed

logger = get_logger("generate_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.data.seed)

    logger.info("generating %d trajectories of length %d (grid=%d)",
                cfg.data.n_trajectories, cfg.data.trajectory_length, cfg.data.grid_size)

    data = generate_dataset(
        n_trajectories=cfg.data.n_trajectories,
        trajectory_length=cfg.data.trajectory_length,
        grid_size=cfg.data.grid_size,
        dt=cfg.data.dt,
        diffusivity=cfg.data.diffusivity,
        seed=cfg.data.seed,
        substeps=cfg.data.substeps,
    )

    out = Path(cfg.data.data_dir) / f"{cfg.data.parameter}.npz"
    save_dataset(data, out)
    logger.info("wrote dataset -> %s (fields shape=%s)", out, data["fields"].shape)


if __name__ == "__main__":
    main()

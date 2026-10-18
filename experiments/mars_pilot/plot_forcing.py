"""
Plot F_a(t) for every planned trajectory, computed from the schedule DEFINITIONS
(deterministic from seed) -- no data files are opened, so no test-split values
are ever inspected.

One small-multiples figure per split (each subplot = one trajectory).
Writes figures/forcing/<family>_<split>.png and prints the forcing definitions.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_config
from gen_data import (SPLITS, RECORD_YEARS, build_tasks, make_schedule,
                      FORCING_DEFINITIONS)

OUT = "figures/forcing"
os.makedirs(OUT, exist_ok=True)


def main():
    print(FORCING_DEFINITIONS)
    cfg = load_config()
    year = cfg["planet"]["year_sols"]
    total = RECORD_YEARS * year
    t = np.linspace(0.0, total, 400)

    tasks = build_tasks(cfg, OUT, RECORD_YEARS)
    # group tasks by split
    by_split = {}
    for tk in tasks:
        by_split.setdefault(f"{tk['family']}/{tk['split']}", []).append(tk)

    for key, group in by_split.items():
        ncol = 5
        nrow = int(np.ceil(len(group) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 1.6 * nrow),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, tk in enumerate(group):
            rng = np.random.default_rng(tk["seed"])
            fa, _ = make_schedule(tk["shape"], tk["magnitude"], total, rng)
            y = [fa(float(ti)) for ti in t]
            ax = axes[i // ncol][i % ncol]
            ax.axis("on")
            ax.plot(t / year, y, lw=1.2)
            ax.set_title(f"{tk['seed']} {tk['shape']}\nA={tk['magnitude']:.2f}",
                         fontsize=6)
            ax.tick_params(labelsize=5)
            ax.set_ylim(-0.5, 2 * 5 + 0.5)
        fig.suptitle(f"F_a(t) -- {key}  ({len(group)} trajectories)", fontsize=10)
        fig.supxlabel("time (Mars years)", fontsize=8)
        fig.supylabel("F_a (W/m^2)", fontsize=8)
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
        fname = os.path.join(OUT, key.replace("/", "_") + ".png")
        fig.savefig(fname, dpi=110)
        plt.close(fig)
        print(f"wrote {fname}  ({len(group)} trajectories)")


if __name__ == "__main__":
    main()

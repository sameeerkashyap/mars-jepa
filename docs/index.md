# MarsWorld

MarsWorld studies whether models with memory and physical structure can predict
Mars-like climate responses to new heating schedules.

Start with [what we model and which data we use](DATA_AND_MODELING.md).

- [Architecture](architecture.md): the spatial PI-JEPA world model.
- [Heating pilot](pilot/RUNME.md): reference physics, trajectory data, training,
  and architecture comparisons.
- [Paper revision](pilot/PAPER_REVISION_PLAN.md): evidence review and experiments.
- [Research plan](Mars_PI_JEPA_Research_Plan.md): the original spatial-model plan.
- [Training](training.md): spatial PI-JEPA commands and loss settings.
- [Cluster jobs](kubernetes.md): container and Kubernetes execution.

## Repository map

| Folder | Purpose |
|---|---|
| `src/marsworld/models/pi_jepa/` | Spatial world-model architecture |
| `src/marsworld/models/latent_jepa/` | Latitude-band latent world model |
| `benchmarks/` | GRU, RSSM, Transformer, hybrid, and linear comparisons |
| `src/marsworld/physics/` | Reference physics, simplified model, physical losses |
| `src/marsworld/data/` | Dataset readers and spatial generator code |
| `data/synthetic/` | Generated arrays by experiment |
| `data/arco_mars/` | Future historical Mars records |
| `data/specification/` | Physical settings, splits, hashes, thresholds |
| `scripts/` | Training, generation, evaluation, plots, diagnostics, utilities |
| `tests/` | Spatial and pilot verification |

Install from the repository root with `python -m pip install -e ".[dev,docs,benchmark]"`.

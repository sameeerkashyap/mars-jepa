# MarsWorld PI-JEPA

A compact, modular **Physics-Informed Joint Embedding Predictive Architecture
(PI-JEPA)** for predicting Martian climate fields. The long-term goal is a world
model that holds a learned **embedding representation of the Mars environment**
and can be conditioned on interventions.

This repository starts deliberately small: a **single physical parameter
(surface temperature)**, a reproducible synthetic PDE solver, and a clean
training pipeline that runs identically on a laptop or a Kubernetes GPU pod.
Additional parameters (e.g. dust optical depth) slot in without restructuring.

!!! note "Scope discipline"
    Good performance on this single-parameter proof of concept does **not**
    establish multi-century climate prediction or terraforming control. Claims
    must match the evaluation performed. See the
    [research plan](https://example.invalid) in `Mars_PI_JEPA_Research_Plan.md`.

## What it does

```text
 x_t  --(context encoder)-->  z_t --+
                                     |-- predictor(z_t, forcing a_t) --> z_hat_(t+1)
 x_(t+1) -(target enc, EMA)-> z~_(t+1)          |
                                     JEPA loss ||z_hat - sg(z~)||
                                                |
                                     decoder(z_hat) --> x_hat_(t+1)
                                     forecast loss + PDE/conservation loss
```

## Quick start

```bash
# install (editable) + docs tooling
pip install -e ".[docs,dev]"

# generate synthetic trajectories
python scripts/generate_data.py --config configs/default.yaml

# train the physics-informed model
python scripts/train.py --config configs/default.yaml

# train the no-physics ablation (model B)
python scripts/train.py --config configs/baseline_no_physics.yaml

# serve these docs
mkdocs serve
```

## Repository map

| Path | Purpose |
| --- | --- |
| `src/marsworld/config.py` | Typed configuration dataclasses + YAML loader |
| `src/marsworld/data/` | Synthetic PDE solver and torch datasets |
| `src/marsworld/models/` | Encoder, predictor, decoder, PI-JEPA model |
| `src/marsworld/physics/` | PDE residual and conservation losses |
| `src/marsworld/training/` | Composite loss and training loop |
| `src/marsworld/utils/` | Logging and reproducibility helpers |
| `scripts/` | `generate_data.py`, `train.py` entry points |
| `configs/` | YAML experiment configurations |
| `k8s/` | Dockerfile + Kubernetes Jobs/PVC |

Explore each component under **Components** in the navigation; those pages pull
API docs directly from the source docstrings.

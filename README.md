# MarsWorld PI-JEPA

A modular **Physics-Informed Joint Embedding Predictive Architecture (PI-JEPA)**
for Martian climate dynamics. The long-term aim is a world model that learns an
**embedding representation of the Mars environment** and can be conditioned on
interventions.

This repo bootstraps that program with a single physical parameter — **surface
temperature** — a reproducible synthetic PDE solver, and a training pipeline that
runs identically on a laptop or a Kubernetes GPU pod. New parameters (e.g. dust
optical depth) slot in without restructuring.

> ⚠️ **Scope:** strong results on this proof of concept do **not** establish
> multi-century climate prediction or terraforming control. See
> `Mars_PI_JEPA_Research_Plan.md` for the full research framing and claim
> discipline.

## Layout

```
src/marsworld/
  config.py          typed config dataclasses + YAML loader
  data/              synthetic solver + torch datasets
  models/            encoder, predictor, decoder, PI-JEPA
  physics/           PDE residual + conservation losses
  training/          composite loss + trainer
  utils/             logging + seeding
scripts/             generate_data.py, train.py
configs/             experiment YAMLs
k8s/                 Dockerfile + Jobs + PVC
docs/                MkDocs site (per-component API docs)
tests/               CPU smoke tests
```

## Quick start

```bash
pip install -e ".[docs,dev]"

python scripts/generate_data.py --config configs/default.yaml   # Stage A data
python scripts/train.py        --config configs/default.yaml     # train model C
python scripts/train.py --config configs/baseline_no_physics.yaml  # model B

pytest -q          # smoke tests
mkdocs serve       # browse the docs
```

## Models (ablation ladder)

| Model | Config | JEPA | Physics |
| --- | --- | --- | --- |
| A simple predictor | set `jepa_weight: 0`, `physics.enabled: false` | – | – |
| B JEPA predictor | `configs/baseline_no_physics.yaml` | ✓ | – |
| C physics-informed JEPA | `configs/default.yaml` | ✓ | ✓ |

## Kubernetes

Two-stage batch pipeline (CPU data gen → GPU training) over a shared PVC. See
[`k8s/README.md`](k8s/README.md) or the **Kubernetes** docs page.

## Documentation

Full architecture notes and auto-generated per-component API docs live in
`docs/` and build with MkDocs Material + mkdocstrings:

```bash
mkdocs build --strict
```

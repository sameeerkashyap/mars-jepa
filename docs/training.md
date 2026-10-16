# Training guide

## Local run

```bash
pip install -e ".[dev]"
python scripts/generate_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
```

`train.py --auto-generate` will synthesise the dataset on the fly if it is
missing, which is convenient for a one-shot container run.

## Configurations

| File | Purpose |
| --- | --- |
| `configs/default.yaml` | Full physics-informed model (model C) |
| `configs/baseline_no_physics.yaml` | JEPA-only ablation (model B) |

Override any field by editing a YAML file; keys map 1:1 to the dataclasses in
`marsworld.config`.

## Experiments from the research plan

- **Multi-step rollout** — set `train.rollout_steps > 1`; the trainer
  autoregresses in latent space and applies forecast/physics losses at each step.
- **Data efficiency** — set `data.data_fraction` to `0.1`, `0.25`, `0.5`, `1.0`
  and compare test error; only the training split is subsampled.
- **Physics ablation** — toggle `physics.enabled` and adjust `pde_weight` /
  `conservation_weight`.
- **OOD forcing** — generate a separate dataset with held-out wind/source ranges
  and evaluate a trained checkpoint against it.

## Outputs

Checkpoints (`best.pt`, `last.pt`) are written to `train.checkpoint_dir` and
contain the model/optimizer state, the full config, and validation metrics.

## Reproducibility

`set_seed` seeds Python, NumPy and Torch. Data generation seeds per-trajectory,
so a dataset is fully determined by `data.seed` and the generation parameters.

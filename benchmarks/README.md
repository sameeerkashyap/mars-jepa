# Comparison architectures

Each architecture has its own directory: `gru`, `rssm`, `transformer`, `hybrid`,
and `linear`. `registry.py` builds them and the main latent JEPA model through the
same temperature rollout interface. `metrics.py` defines common scoring masks.

World-model implementations live in `src/marsworld/models/`.
`pi_jepa/no_physics.yaml` is the spatial model's physics-off comparison.

```bash
python -m scripts.pilot.train --arm gru --physics off --device cpu
python -m scripts.pilot.train --arm hybrid --physics off --device cpu
python -m scripts.pilot.sweep --help
```

Run from the repository root after installation. Keep data and evaluation settings
equal across methods. The registered grid is in
`data/specification/pilot/config_phase2.yaml`.

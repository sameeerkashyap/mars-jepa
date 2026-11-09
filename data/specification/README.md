# Data specifications

`pilot/` records the heating experiment:

- `config.yaml`: reference simulator parameters.
- `config_phase2.yaml`: protocol, model sizes, and data fractions.
- `PREREG.md`: hypotheses and dated design changes.
- `thresholds.yaml`: decision thresholds and statistical procedure.
- `ebm_params.json`: fitted simplified-physics parameters.
- `DATA_SHA256.txt`: trajectory hashes with paths relative to the repository root.

The manifest is in `data/synthetic/pilot/manifest.json`. Its file paths are relative
to its data directory. Data contents and hashes were preserved during the move;
checksum path prefixes were updated.

Spatial prototype settings are in `configs/pi_jepa/default.yaml` and the
comparison config `benchmarks/pi_jepa/no_physics.yaml`.

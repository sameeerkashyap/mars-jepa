# Run the Mars heating pilot

Run commands from the repository root after installing the project:

```bash
python -m pip install -e ".[dev,benchmark]"
```

## Locations

- Settings and split rules: `data/specification/pilot/`.
- Trajectory root: `data/synthetic/pilot/`.
- World model: `src/models/latent_jepa/`.
- Comparison architectures: `benchmarks/`.
- Reference physics: `src/physics/`.
- Commands and utilities: `scripts/pilot/`.
- New results and runs: `outputs/pilot/`.

Read [the data plan](../DATA_AND_MODELING.md) before choosing an experiment.

## Recover or generate arrays

The manifest lists 210 trajectories. Full arrays are not bundled. For exact
reproduction, recover them and check hashes from the repository root:

```bash
shasum -a 256 -c data/specification/pilot/DATA_SHA256.txt
```

To generate the manifest's 8–10 W/m² test design, use a new output directory:

```bash
python -m scripts.pilot.gen_data --out data/synthetic/pilot_new --workers 2 --tier2-mag 8 10
```

The historical generator default is 7.5–10 W/m². Pass bounds explicitly. Record
the new manifest, configuration, and hashes. Supply the new data root to training
with `--data-dir data/synthetic/pilot_new`. Full generation takes time.

For a small pipeline check:

```bash
python -m scripts.pilot.gen_data --smoke --out data/synthetic/pilot_smoke --workers 1
```

Smoke data are for checking the pipeline.

## Train and evaluate

```bash
python -m scripts.pilot.train --arm jepa --physics off --device cpu
python -m scripts.pilot.train --arm gru --physics off --device cpu
python -m scripts.pilot.train --arm gru --physics A --device cpu
python -m scripts.pilot.train --arm hybrid --physics off --device cpu
python -m scripts.pilot.eval --help
```

Use `--data-dir` for a new data root. Training also accepts `--epochs`,
`--max-batches`, `--fraction`, `--seed`, and `--out`.

Arms: `jepa`, `gru`, `gru_noforce`, `rssm`, `transformer`, `hybrid`, `linear`,
and `ebm_only`. The sweep also supports a GRU ensemble.

## Compare and verify

```bash
python -m scripts.pilot.calibrate --help
python -m scripts.pilot.sweep --help
python -m scripts.pilot.analyze --help
python -m pytest -q
python -m scripts.pilot.regression_check
```

The sweep uses a fixed physics weight. The separate train command supports
validation selection with `--select-lambda`. Record the protocol used.

Earlier diagnostic documents are historical records. Their short filenames refer
to the former flat pilot directory. Use the locations above for current commands.
Design history is preserved in `data/specification/pilot/PREREG.md`.

# Small-data TorchRL training

This trains a 66,480-parameter temperature forecaster with TorchRL's DreamerV3
categorical prior, posterior, block GRU, and KL loss. It uses 30% of the training
trajectories, subset seed 0, initialization seed 0, and five epochs. On the real
pilot dataset this means 30 training trajectories and all 20 validation
trajectories. Each example has 40 observed sols and a 120-sol forecast.

The model receives surface temperature, sunlight, and heating. Sunlight is a
supplied environmental driver. Only heating would be controllable in a future
policy. Ice, ground temperature, dust, and evaluation labels stay outside the
model inputs. Training does not load test splits.

## Environment

Run from the repository root with Python 3.12 and uv. Keep the environment,
caches, and temporary files in this project:

```sh
rtk proxy mkdir -p .cache/uv .cache/tmp
rtk proxy env UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" \
  uv venv --python 3.12 .venv-torchrl
rtk proxy env UV_CACHE_DIR="$PWD/.cache/uv" TMPDIR="$PWD/.cache/tmp" \
  uv pip install --python .venv-torchrl/bin/python \
  -r k8s/nautilus/requirements.txt -e '.[dreamer,dev]'
```

The optional `dreamer` dependencies pin PyTorch 2.10.0, TorchRL 0.14.0, and
TensorDict 0.14.2. This release contains the DreamerV3 components without the
newer nightly-PyTorch requirement. Existing model arms do not import TorchRL.

## Train on the pilot data

Recover the full pilot dataset for local training, or use its existing location
on Nautilus: `/results/pilot-20261109/data`. The local manifest alone does not
contain the temperature arrays. Set `--data-dir` to the recovered dataset:

```sh
rtk proxy .venv-torchrl/bin/python -u -m scripts.pilot.small_study train \
  --config data/specification/pilot/torchrl_small.yaml \
  --variant torchrl_dreamer --seed 0 --epochs 5 --device cpu \
  --data-dir /absolute/path/to/pilot \
  --out outputs/pilot/torchrl-small/seed0
```

Use a fresh output directory for a new experiment. Running the same command
resumes its matching checkpoint after an interruption. Configuration, data, and
device checks are enforced; Nautilus also pins and checks the source archive.
Use a fresh directory after local code changes. Normalization is fitted on the selected
training trajectories only. The best validation epoch is exported as `model.pt`;
`latest.pt` holds the optimizer and random state for recovery. Training curves,
normalization, settings, and status are saved beside them.

## Verify the pipeline with a tiny fixture

This generated fixture checks execution and recovery. Its scores do not measure
Mars forecast skill:

```sh
rtk proxy .venv-torchrl/bin/python -m tests.pilot.fakedata outputs/pilot/torchrl-smoke/data
rtk proxy .venv-torchrl/bin/python -u -m scripts.pilot.small_study train \
  --config data/specification/pilot/torchrl_small.yaml \
  --variant torchrl_dreamer --seed 0 --epochs 5 --device cpu \
  --data-dir "$PWD/outputs/pilot/torchrl-smoke/data" \
  --out outputs/pilot/torchrl-smoke/run
rtk proxy env TMPDIR="$PWD/.cache/tmp" .venv-torchrl/bin/python -m pytest \
  tests/pilot/test_torchrl_dreamer.py tests/pilot/test_small_study.py \
  -q --basetemp=.cache/pytest-torchrl
```

## Training objective and comparison

The shared trainer minimizes open-loop future-temperature mean squared error.
The TorchRL arm also learns to reconstruct temperatures from observation-updated
latent states. Its KL terms fit the dynamics prior to the posterior and
regularize the posterior, with weights 0.5 and 0.1 and one free nat. All
temperature losses use training-subset normalization. Transition to temperature
at index `t` uses sunlight and heating from `t-1`.

Future temperatures enter only the training loss. Forecasting advances the prior
without observing them. Validation uses deterministic categorical modes;
`sample=True` draws stochastic trajectories. The normal evaluation loader can
reload this arm using its saved settings and weights.

This is a forecast-only adaptation of DreamerV3 components. It has no actor,
critic, reward model, or policy training. The earlier JEPA study is preserved;
this extra arm is exploratory. Five-epoch fixture or pilot results do not
establish an advantage over models trained with a larger budget.

For Nautilus, the existing small-study Job accepts `MARSWORLD_CONFIG`,
`MARSWORLD_TORCH_VERSION`, and `MARSWORLD_EXTRA_REQUIREMENTS`. Use this config,
version `2.10.0`, and `k8s/nautilus/requirements-torchrl.txt` in a fresh study
directory with its own source archive. Jobs must have no runtime deadline.

Component references: [TorchRL prior and posterior](https://github.com/pytorch/rl/blob/v0.14.0/torchrl/modules/models/model_based.py)
and [categorical KL loss](https://github.com/pytorch/rl/blob/v0.14.0/torchrl/objectives/dreamer_v3.py).

## Three-seed comparison on Nautilus

The comparison uses the same 30 training trajectories, all 20 validation
trajectories, and early stopping as the completed JEPA study. It trains seeds
0, 1, and 2 from scratch for up to 20 epochs. The five-epoch pilot stays intact.

```sh
rtk proxy env TMPDIR="$PWD/.cache/tmp" UV_CACHE_DIR="$PWD/.cache/uv" \
  .venv-torchrl/bin/python -u -m scripts.pilot.torchrl_comparison \
  --state-dir outputs/pilot/torchrl-comparison-v1 \
  --study-id mars-torchrl-match-v1
```

The launcher requires the saved baseline and pilot state files under
`outputs/pilot/small-study-v2` and `outputs/pilot/torchrl-small-v1`. It checks the
training settings and data fingerprints against the baseline. It carries forward
the compute charges and runs at most two jobs at once. Jobs have no time limit.
Repeat this command to collect existing jobs after a launcher interruption.
Completed tasks are reused.

All three training jobs must finish before test evaluation starts. The launcher
writes `FROZEN.json` with the best validation checkpoint and file hashes for
each seed. Evaluation checks these hashes before loading either test tier.
It evaluates each seed once, then writes `report.json` and `report.md` in the
state directory. Reports include paired differences against the saved GRU,
JEPA without latent loss, JEPA, and linear results for the South-pole winter
heating-response error. Negative differences favor TorchRL.

This is an exploratory comparison on one fixed training subset. The baseline
used PyTorch 2.5.1; TorchRL requires 2.10.0. Three seeds do not support the
original ten-seed study's hypothesis claims.

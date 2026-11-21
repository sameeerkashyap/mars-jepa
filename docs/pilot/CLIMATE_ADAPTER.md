# Response-control Dreamer adapter

This implements the compatibility and data boundary in issue 9. It does not
admit a climate actor experiment or claim forecast or control skill.

## Public interfaces

```python
from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
from benchmarks.dreamer_control.climate import ClimateVectorEnv, climate_imagination
from benchmarks.dreamer_control.data import generate_dataset, load_dataset

learner = DreamerLearner(LearnerConfig(obs_dim=133),
                         imagination_env_factory=climate_imagination)
env = ClimateVectorEnv()
episode = learner.collect_episode(env, seed=5000)
learner.update(model_only=True)
learner.save("trusted-local.pt")
restored = DreamerLearner.load("trusted-local.pt",
                               imagination_env_factory=climate_imagination)

# Run generation only after the separate cost admission.
# generate_dataset("new-directory", split="train", train_conditions=12)
# manifest, episodes = load_dataset("new-directory", split="train")
# for episode in episodes:
#     learner.add_episode(**episode["policy"])
```

`ClimateVectorEnv` wraps only `ResponseHeatingControl`. Its 133 fields use the
fixed order and scaling in [the original task](HEATING_CONTROL_TASK.md).
The [response task](HEATING_RESPONSE_TASK.md) keeps the matched no-heat world
private. Neither hidden state nor response/reference temperatures enter the
policy vector or daily observable records. The private oracle supplies reward
labels, separate response targets, and score diagnostics. Trusted simulator state dictionaries
remain separate from replay.

A raw scalar Normal sample is clipped to [-1,1]. The adapter maps it to requested
heat with `5*(clipped+1)`, then projects against actual remaining heat. Replay
uses `applied_heat/5-1`. The raw sample remains in `requested_actions`; physical
requested/applied heat and daily observations are returned in step info.

## Exact imagined constraints

`climate_imagination` retains the learner's shared world model. It seeds public
remaining sols and heat from `sample['next','observation']` in the exact flattened
batch/time order used by TorchRL. Pending starts are transient and consumed at
reset. A fixed task-constants buffer is saved with the learner. Fresh resets use
120 sols and 300 W/m²·sol. No learned observation decoder can refill that heat.

Each imagined step projects heat before calling the learned prior. The root
action stays raw so REINFORCE scores its actual Normal sample. Time advances by
ten sols and heat debits the projected amount. Empty heat permits zero-heat
steps until time ends. At zero remaining time, latent state and public context
freeze, heat and reward are zero, and continuation is zero. TorchRL may still
compute padded fixed-horizon slots; they add no task interval or return.

Before the known endpoint, continuation retains the learned recipe prediction.
Its training target still includes `1-1/333` by default. The wrapper masks the
known endpoint; it does not replace the learned prediction with exact one.
There is no additional discount in the wrapper. This retains the existing
learner objective, which is distinct from the reported undiscounted daily-MSE
score. A later pilot must state this objective choice explicitly. Changing it
requires a declared learner setting and validation; this adapter does not
silently change the recipe.

## Versioned episodes

`generate_dataset` creates a new directory and refuses to overwrite one. It
writes four matched schedules per condition: zero, constant 2.5 W/m², a seeded
three-interval full-budget pulse, and a seeded feasible random schedule. The
schedule seed stream is `SeedSequence([condition_seed, schedule_index, 1])`.
One spun-up paired checkpoint is reused within a condition. It is never stored
in policy data.

Training defaults to nested groups 5000–5011 or 5000–5023. Validation uses
5100–5107. Explicit subsets support small cost fixtures. Every schedule for a
condition stays in its split. Conditions 5200–5219 are declared and sealed;
9000–9999 remain reserved. The generator and loader reject test access before
opening data. Existing forecast datasets and loaders are unchanged.

Each compressed episode stores 13 policy observations, 12 raw/applied actions,
integrated rewards, termination flags, physical requested/applied heat, and
120 daily temperature/sunlight/time observations. A separate
`oracle_daily_response_T_s[12,10,64]` array stores heated-minus-matched-no-heat
response targets. The zero schedule supplies that matched no-heat trajectory
from the same initial paired checkpoint, without another simulator run. The
loader returns targets only as `episode['oracle_diagnostics']['daily_response_T_s']`.
They can score prior forecasts at 10/30/60/120 sols; they are never policy or
replay inputs. The manifest stores condition
and schedule identity, separate oracle score diagnostics, resolved config,
source/config/task hashes, a hash of each file, and a hash of the complete
manifest. It records wall and process CPU cost, one serial CPU allocation,
zero GPU use, and paired spin-up reuse.

`load_dataset(path, split=...)` checks file and manifest hashes, exact current
source/task/config identity, split membership, complete schedule groups,
shapes/dtypes, finite values, action/observation alignment, per-interval reward
from the private response target, integrated score,
and the heat/time contract. NumPy loading disables pickle. It returns
`(manifest, episodes)`. Only `episode['policy']` is accepted by `add_episode`.
Daily observations and oracle diagnostics occupy separate dictionaries. No
normalizer or model is fitted by generation or loading.

CLI, after admission:

```sh
rtk env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /path/to/python -m benchmarks.dreamer_control.data --out new-directory --split train --train-conditions 12
```

## Checked scope and cost

Focused tests cover real clipping and exhausted heat, private-field exclusion,
public-vector shape, replay action alignment, exact time/context flattening,
mixed endpoints, projection before the prior, terminal latent freeze, nonzero
learned reward masking, continuation masking, raw action retention, actor-only
gradients, two physical model-only updates, deterministic physical evaluation,
and exact checkpoint continuation. A synthetic fixture checks full loss/update
recovery. No climate actor is trained in these checks.

The one-condition/four-schedule data fixture measured 4.968 wall seconds and
4.933 process CPU seconds. With one CPU core this is 0.001380 allocated
core-hours and 0.001370 process CPU-hours; GPU use is zero. This includes one
spin-up and four paired schedules. Linear estimates for 12 and 24 conditions
are about 60 and 120 wall seconds on this machine. They are estimates, not
admission for a production run. Test reruns and learner checks are separate.
No full training/validation generation, final-test access, Nautilus job, forecast
validation run, or production policy training was performed for this adapter.

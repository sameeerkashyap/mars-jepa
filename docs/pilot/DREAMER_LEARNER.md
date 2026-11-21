# Eager vector DreamerV3 learner

`benchmarks.dreamer_control` is a complete CPU learner. It is separate from
`benchmarks.torchrl_dreamer`, which remains a forecast benchmark.

The network composition adapts the official TorchRL v0.14.0 recipe at commit
`cc3fe0f02fc599ab367acbd75d1195139c6f7cef`. The annotated release tag object is
`32b9c057509005a7486c64175c73bd66b46269d8`; it points to that source commit.
Source: `sota-implementations/dreamer_v3/dreamer_v3_agent.py` and `train.py` in
[pytorch/rl](https://github.com/pytorch/rl/tree/cc3fe0f02fc599ab367acbd75d1195139c6f7cef/sota-implementations/dreamer_v3).
The copied composition is under Meta's MIT license. See
`benchmarks/dreamer_control/LICENSE.torchrl`.

## Interface

```python
from benchmarks.dreamer_control import DreamerLearner, LearnerConfig

learner = DreamerLearner(LearnerConfig(obs_dim=observation_size))
learner.collect_episode(vector_env, seed=0)
model_metrics = learner.update(model_only=True)  # Before the forecast readiness gate.
metrics = learner.update()
learner.save("learner.pt")
learner = DreamerLearner.load("learner.pt")
held_out = learner.collect_episode(vector_env, seed=1, deterministic=True, store=False)
```

The vector adapter has two methods:

- `reset(seed=...)` returns one finite float vector of size `obs_dim`.
- `step(raw_action)` receives a tensor of shape `[1]`. It returns
  `(next_vector, reward, terminated, truncated)`. It can add a fifth `info` dict
  with `dynamics_action`, a tensor of shape `[1]` in model action units.

`dynamics_action` is the applied action after physical projection. Collection
stores it in `actions` and uses it as the next policy carry's previous action.
`requested_actions` retains the raw policy sample. Without this field, the
adapter declares that the requested action is also the dynamics action.
The adapter owns unit conversion and bounds. No sunlight action exists.

Use `update(model_only=True)` before the forecast readiness gate. It trains
reconstruction, reward, continuation, and split KL only. It leaves actor,
online/slow critic, and return statistics unchanged. It increments `updates`
but leaves `behavior_updates` unchanged. `update()` also learns behavior and
updates the slow critic. Both use the same optimizer and warmup clock.

`act(vector, carry=None, deterministic=True)` returns `(raw_action, carry)`.
Pass `None` at each reset. The caller must replace `carry['previous_action']`
with the applied model action after an environment projection. `collect_episode`
does this automatically. Deterministic inference uses posterior categorical
modes and the Normal policy mode. It draws no random values and changes no
weights, return statistics, or replay.

`add_episode` accepts `observations[N+1, obs_dim]`, `actions[N,1]`,
`rewards[N,1]`, boolean `terminated[N,1]` and `truncated[N,1]`, plus optional
`requested_actions[N,1]`. Exactly the last transition must end the episode.
It copies inputs and stores only complete episodes. Replay capacity counts
episodes, not transitions. `sample()` selects contiguous chunks from one
episode at a time. It rebuilds the starting latent context from the episode
prefix with current weights and categorical modes. This avoids stale cached
states. Prefix cost is linear in the episode length; this is small for the
12-action task.

Record zero predicts the reset observation from zero carry and zero action.
The next record predicts the next observation from the first applied action.
Each sampled chunk has `sequence_length` records. Episodes shorter than this
length are ineligible; `update()` raises if none is eligible. The default
length is eight records, suitable for a 12-action episode plus reset record.

## Objectives and adaptations

The library supplies the recurrent categorical prior and posterior, symlog
reconstruction loss, two-hot reward and value losses, split categorical KL,
continuation loss, REINFORCE actor loss, entropy bonus, percentile return
normalization, replay value loss, and slow critic regularizer. The combined
library loss keeps posterior starts and critic targets detached for imagined
learning. Replay value learning can train the representation, as in the recipe.

Explicit settings match the recipe: dynamic KL weight 1, representation KL
weight 0.1, uniform mixture 0.01, summed KL free-nats clamp, replay value weight
0.3, slow critic regularizer 1, external gamma 1, lambda 0.95, entropy 0.0003,
and slow target rate 0.02. Continuation targets retain the recipe discount
`1 - 1/333`. This discount is separate from true termination. It is applied
once, through the continuation target and replay return horizon. The task
driver must record any changed horizon. The value estimator stops at true
terminal transitions and retains bootstrapping for truncations.

The compact vector adaptation uses one-layer MLPs, standard GRU recurrence,
one observation reconstruction event, no pixels, synchronous episode
collection, and in-memory tensor replay. It uses the library's DreamerV3
optimizer, including gradient clipping and warmup. No native replay extension,
Hydra, compilation, mixed precision, asynchronous collection, or runtime
downloads are required.

The official continuous policy uses an unbounded Normal with bounded mean.
The sample itself is not tanh squashed. The real adapter must clip the raw
sample before mapping it to physical units. Actor REINFORCE scores the raw
sample. This distinction matters after clipping or budget projection.

## Climate imagination adapter

The concrete [response-control adapter](CLIMATE_ADAPTER.md) supplies exact public
remaining heat and time state for the 133-field response task. It seeds context
from sampled next observations and projects the raw policy action into applied
heat before the learned prior sees it. It advances heat and time without learning
them and zeros reward and continuation after the physical task ends. Actor log-probabilities must still score the raw sample.

The supported hook is
`DreamerLearner(config, imagination_env_factory=factory)`. The factory receives
`(default_dreamer_env, learned_continuation_model)` and returns
`(extended_dreamer_env, context_aware_continuation_model)`. The environment must
subclass `DreamerEnv` and retain the exact shared `world_model`. Its
`prepare_imagination(sample)` method, if present, runs before each loss call.
It can seed context in the same flattened batch/time order used by the
library actor loss. The custom continuation model must include that context
in its input keys and enforce terminal masking. The same factory must be
supplied to `load(..., imagination_env_factory=factory)`.
The adapter must reuse the existing trainable world-model parameters. It can
add deterministic state buffers; those buffers are included in checkpoints.
Adding a new learned adapter head needs a separate optimizer-ownership change.

The default factory is sufficient for the small vector fixture only. Use
`climate_imagination` for response control. Its focused tests check projection,
exact time/heat state, and recovery. Climate performance remains a separate gate.
No climate actor training was run for this adapter.

## Recovery and checked stack

`save` atomically replaces a trusted local checkpoint. It includes resolved
configuration, exact source commit, dependency versions, all loss/module
weights and buffers, online and slow critic, optimizer and warmup state,
return statistics, complete episode replay, counters, and Python/NumPy/torch
random-number states. Recovery requires the same recorded CPU software stack.
Checkpoints use pickle and must come from a trusted source.

Recovery is at an **episode boundary**. No active collector or simulator state
is stored. Start the next episode with its recorded seed. The experiment driver
must save its seed cursor and environment/task configuration beside this state.
An interrupted partial episode must be discarded. Mid-episode exact recovery
needs the simulator state and policy carry and is outside this interface.

Checked locally with Python 3.12.3, torch 2.10.0, torchrl 0.14.0, and
tensordict 0.14.2. The checkpoint records exact runtime versions. Run:

```sh
rtk env PYTHONPATH=src:. /path/to/.venv-torchrl/bin/python -m pytest tests/control/test_learner.py -q
```

The fixture checks collect/replay/update/save/reload, identical next update,
replay alignment and resets, actual versus requested actions, deterministic
evaluation with unchanged RNG/statistics, gradient ownership, split-KL golden
values and stopped gradients, two-hot decoding, hand-worked lambda returns,
terminal versus truncated bootstrap, and slow-target arithmetic. These checks
establish local eager learner compatibility. They do not establish policy skill
or portability to another device or software stack.

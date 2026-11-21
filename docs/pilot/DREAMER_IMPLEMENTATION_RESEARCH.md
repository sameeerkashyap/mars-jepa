# Complete Dreamer for heating control

Research date: 2026-11-21. Decision ticket: [Choose a maintained complete TorchRL Dreamer implementation](https://github.com/sameeerkashyap/mars-jepa/issues/4).

## Decision

Use the official **TorchRL v0.14.0 DreamerV3 recipe** as the implementation reference. Add a separate complete-agent control path around its custom environment factory. Keep the existing forecast arm and its saved results separate. Start with eager execution, one synchronous collector, vector observations, and scalar continuous actions. This is a practical maintained path with a runnable recipe and focused upstream tests. It is not evidence that Dreamer will improve this benchmark.

The official recipe supports vector observations and continuous actions. Its builders cover the world model, actor, value model, continuation head, and real-environment policy. Its training entry point joins collection, replay, losses, optimization, and target updates. It ships a small Pendulum configuration and a larger Walker configuration. The latter follows a named, pinned JAX preset and explicitly distinguishes that preset from the paper protocol. [Recipe README](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/README.md), [training entry point](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/train.py), [network and environment builders](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/dreamer_v3_agent.py).

## Version decision and its limits

Retain Python 3.12, torch **2.10.0**, torchrl **0.14.0**, and tensordict **0.14.2** for the first integration check. The local `.venv-torchrl` successfully imported these exact versions. TorchRL's tagged metadata accepts torch >=2.1.0, requires Python >=3.10, and requires TensorDict >=0.14.2,<0.15.0. Replay's extra requires torch >=2.7.0. Thus the pins satisfy declared requirements. This does **not** prove that the complete learner, extension-backed replay, or accelerated paths work together. [Tagged package metadata](https://github.com/pytorch/rl/blob/v0.14.0/pyproject.toml).

Record the exact upstream commit behind the release tag when preparing the source snapshot. The Git tag API returned object `32b9c057509005a7486c64175c73bd66b46269d8`; resolve an annotated tag to its commit rather than treating the tag-object SHA as the source commit. Pin the recipe source as well as the wheels. The recipe files are needed alongside the installed library.

Do not upgrade merely to obtain a complete agent: this release already provides one. Newer maintained releases exist. TorchRL 0.14.1 requires TensorDict >=0.14.3,<0.15.0; its documented release build targets torch 2.14.1. A future upgrade must use a separately checked lock and must not silently alter the frozen forecast comparison. [Official release notes](https://github.com/pytorch/rl/releases).

Before training, run a tiny custom-environment collect/replay/learner/save/reload check on the exact target CPU or GPU. Check package dependency resolution and extension loading. A failed check is a compatibility blocker, not permission to launch a long run. Keep compilation, CUDA graphs, asynchronous collection, and mixed precision disabled until this works. The upstream smoke test also exercises eager collection and replay. [Upstream objective and integration tests](https://github.com/pytorch/rl/blob/v0.14.0/test/objectives/test_dreamer_v3.py).

## Existing code inventory

The current working tree was inspected without staging it. `benchmarks/torchrl_dreamer/model.py` subclasses `BaseSeqModel`. It uses categorical prior and posterior modules, an observation encoder, a temperature decoder, and split KL terms. Its transition input joins 64 sunlight values with one heating value. Forecasting advances the prior; future temperatures enter a training-only posterior loss. Validation chooses categorical modes.

`tests/pilot/test_torchrl_dreamer.py` checks forcing alignment, deterministic forecasts, component gradients, target isolation, and checkpoint recovery. `docs/pilot/TORCHRL_SMALL.md` explicitly calls the arm forecast-only. `scripts/pilot/torchrl_comparison.py`, its YAML, the registry, and the shared train/eval loaders serve that forecast comparison. None supplies a learned policy.

Missing for a complete control agent: environment interaction and persistent state; scalar actor; reward and continuation heads; online and slow critics; recurrent replay; imagined trajectories; lambda returns; actor/value losses and their gradient boundaries; return normalization; policy evaluation; complete learner/replay checkpoints. Reusing the forecast model alone does not supply these contracts. Do not rename its forecast scores as Dreamer control results.

## Minimal complete-agent contract

The following items are implementation requirements proposed for this project. Numerical task settings remain subject to the task-contract decision.

| Part | Required behavior |
| --- | --- |
| Real environment | One action advances one fixed control interval. Preserve simulator state, absolute sol, and the disturbance schedule across steps. A single full simulation and repeated stepped simulation must agree for the same prescribed actions. |
| Observation | Use only measurements and public task context available when choosing the action. Include time/season and any target or remaining heat budget needed to define the task. Keep hidden ice, ground state, and future disturbance labels out unless the task explicitly exposes them. |
| Action | Actor output has one coordinate. Apply a fixed documented map to permitted heating units. Enforce heating bounds and the remaining physical heat budget in the real environment. Store and train against a consistent raw-action/applied-action convention. |
| Reward | Produce one scalar reward for the transition actually executed. Match the chosen temperature-tracking and heating-cost definition exactly. Train a reward head; evaluating the true simulator reward inside imagination would be a different method. |
| Replay | Store contiguous transitions with stream identity, action, next observation, next reward, terminal/reset flags, and recurrent context. Reset latent state at episode boundaries. A sampled chunk can contain resets only if masks correctly reset recurrence. Do not sample shuffled forecast windows as independent RL steps. |
| World model | Train posterior reconstruction, reward prediction, continuation prediction, and split categorical KL. Imagination starts from detached posterior features and uses the prior without future observations. |
| Behavior learning | Train the actor on imagined lambda returns with explicit REINFORCE and entropy settings. Train the online critic on detached return targets. Keep the recipe's slow-critic regularizer and target update. |
| Recovery | Save model/actor/critic/slow-target weights, optimizer and warmup state, return statistics, RNG states, counters, resolved configuration and source versions. Exact continuation also needs replay and environment state, or a declared episode-boundary restart. |

These components and separation of real posterior learning from latent imagination are supported by the [TorchRL algorithm guide](https://docs.pytorch.org/rl/0.14/reference/dreamer_v3.html). Original [Dreamer](https://arxiv.org/abs/1912.01603) learns behavior through imagined trajectories; [DreamerV2](https://arxiv.org/abs/2010.02193) adds discrete representations. [DreamerV3](https://arxiv.org/abs/2301.04104) emphasizes transformations, normalization, and balancing across tasks. The author-maintained code separately exposes reward, continuation, policy, online value, slow value, and normalization modules. [Pinned original agent](https://github.com/danijar/dreamerv3/blob/e3f02248693a79dc8b0ebd62c93683888ddaccfe/dreamerv3/agent.py).

### Sunlight is not an action

The current 65-coordinate forecast forcing tensor cannot become the policy action space. The smallest stock-recipe adaptation places **current** sunlight and season in the observation vector. The actor controls heating only. The stock latent prior then learns the evolution of the observation context; it does not receive an exact externally supplied future sunlight sequence during imagination. The source builds vector observations separately from the action dimension and uses only latent state, belief, and action for imagined dynamics. [Builders](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/dreamer_v3_agent.py).

If exact known future sunlight must condition every imagined transition, add an explicit clock/driver state that advances independently of the actor, and feed its deterministic schedule into the dynamics. That is a deliberate conditioned-dynamics extension requiring its own alignment and gradient tests. Do not let policy outputs alter sunlight, and do not supply unobserved future storms. The research recommendation is the stock present-context path first; it may model season less accurately than a conditioned model.

### Termination, returns, and normalization

The finite 120-sol task has 12 actions at ten sols per action. If episode completion ends the objective, represent it as terminal for value learning. If an artificial collector cut merely interrupts a continuing objective, represent it as truncation and retain bootstrapping. Do not use `done` indiscriminately for both. Include remaining time in observations for a finite task.

Use the recipe's continuation scaling consistently: its target is non-terminal probability multiplied by `1 - 1 / continuation_horizon`; its configured external gamma is 1.0. Do not apply the same horizon discount twice. For a finite task, choose and record whether this extra discount is retained or disabled. The defaults (continuation horizon 333, imagination length 15) are not Mars task facts. Lambda targets must satisfy `G_t = r_(t+1) + gamma*c_(t+1)*((1-lambda)*V_(t+1) + lambda*G_(t+1))`. Stop at true terminal transitions. [Recipe configuration](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/config.yaml).

Set loss options explicitly. Public constructor defaults can differ from the runnable recipe. In particular, use separate KL with stopped gradients on opposite distributions and clamp the **summed** categorical KL. Keep symlog observation reconstruction, two-hot reward/value heads, percentile return scaling, and slow-target behavior from the chosen recipe unless an adaptation is named. The current normalized-temperature MSE is a separate forecast objective. [Loss implementation](https://github.com/pytorch/rl/blob/v0.14.0/torchrl/objectives/dreamer_v3.py).

## Acceptance checks before any result claim

1. **Environment:** run TorchRL spec checks; compare full versus stepped simulator output; test action endpoints, heat-budget exhaustion, reward timing, terminal/truncation distinction, and independent stream resets.
2. **No leakage:** poison future measurements and hidden state; the action at the current time must stay unchanged. Changing sunlight must affect observations/dynamics without adding an actor coordinate.
3. **Recurrent replay:** use two tiny distinct episodes. Verify transition alignment, no stream mixing, correct state reset, and deterministic latent-context refresh. Short 12-action episodes need a sequence length and sampling rule that are checked explicitly; do not copy the Walker length 64.
4. **Loss arithmetic:** compare a three-step lambda return by hand. Check continuation weights and terminal zero bootstrap. Check two-hot encoding/decoding at zero, between bins, and at support boundaries. Match the upstream KL golden value and stopped-gradient assertions.
5. **Gradient ownership:** isolate each loss. World-model losses train their heads and representation as configured. Actor REINFORCE reaches actor parameters; it must not optimize the world-model or critic parameters. Critic targets and slow-target parameters receive no optimizer gradients. Check slow-target updates numerically. Preserve the selected recipe's replay-value path, which can intentionally train representations.
6. **Recovery and evaluation:** a tiny save/reload run preserves actor outputs, slow targets, and return statistics. A same-stack seeded diagnostic resumes the same next update when full state is saved. Evaluation freezes parameters and statistics and never updates replay from held-out episodes.

Relevant upstream checks include `test_dreamer_v3_reference_kl_fixture_and_gradients`, `test_dreamer_v3_continuation_lambda_and_weights`, `test_dreamer_v3_actor_loss_reinforce`, `test_dreamer_v3_slow_critic_checkpoint_and_online_bootstrap`, replay sequence tests, optimizer-resume tests, and the custom-environment collection smoke test. These are source evidence for test design, not tests run in this research. [Objective and integration tests](https://github.com/pytorch/rl/blob/v0.14.0/test/objectives/test_dreamer_v3.py), [component tests](https://github.com/pytorch/rl/blob/v0.14.0/test/modules/test_dreamer_components.py), [async benchmark test](https://github.com/pytorch/rl/blob/v0.14.0/benchmarks/test_dreamer_v3_benchmark.py).

## Evaluation and remaining decisions

Use a fixed seeded evaluation suite and a declared deterministic policy protocol: categorical posterior modes and actor mode, with exploration off. Check that the upstream state estimator honors this choice; disabling exploration only at the actor is insufficient if posterior sampling remains stochastic. For diagnostics, use synchronous CPU eager execution, fixed RNG states, and strict deterministic PyTorch algorithms so unsupported operations fail visibly. Log versions, device, precision, and backend. Seeded stochastic training is still stochastic; identical results across GPU hardware or compiled/eager backends are not promised. TorchRL explicitly documents that compiled scan changes random draws. [Recipe README](https://github.com/pytorch/rl/blob/v0.14.0/sota-implementations/dreamer_v3/README.md).

A complete Dreamer agent does not require the paper's large model, pixels, asynchronous actors, compilation, or the Walker replay ratio. Retaining the maintained V3 components avoids writing a second RL algorithm. Small dimensions and short sequences are Mars adaptations, and their learning quality remains untested. Call the result **TorchRL DreamerV3 adapted to observed climate vectors**, not a faithful paper reproduction.

Open task choices: reward and heat-budget rules; observations and public future drivers; terminal versus truncation meaning; discount/horizon treatment; sequence length and replay warmup; model size; update ratio and permitted environment interactions; deterministic evaluation state estimator; complete checkpoint scope; and comparison against the frozen forecast study. Resolve these before implementation acceptance or training claims.

Checked: local code and callers named above, tagged official source/configuration/tests, original papers and pinned JAX agent, package metadata, and local version imports. No production code changed, no learner integration test run, and no training started. Main risk: declared package compatibility and upstream test coverage do not establish end-to-end compatibility or control skill on this simulator.

# Small JEPA study

This replaces further work on the 450-configuration sweep. The old data and
177 successful results remain in `/results/pilot-20261109`. The current restart
writes to `/results/mars-jepa-1120-v2`. Earlier small-study checkpoints remain in
`/results/mars-small-jepa-v1`.

## Run and recover

Use Python with PyYAML and an authenticated `nautilus` kubectl context. Keep
the launcher on a durable host. It submits at most two worker Jobs, records
their estimated resource use before submission, and stops expansion on failure.

```sh
rtk proxy python -m scripts.pilot.nautilus_study \
  --state-dir outputs/pilot/small-study-v2 --study-id mars-jepa-1120-v2 --cpu-only
```

Run the same command after a launcher interruption. Keep that state directory:
it holds the exact source archive, image digests, resource ledger, job logs,
decision, and final report. A running Job continues if the launcher disconnects.
Budgets control whether more jobs may start. Running jobs have no runtime limit
and are never stopped because time or budget estimates have been exceeded.
Do not run two launchers for the same study.

The restart uses study ID `mars-jepa-1120-v2` and keeps its local state in
`outputs/pilot/small-study-v2/`. The failed `small-study-v1` state and remote
checkpoints are preserved. `launcher.log` records progress;
`data-preflight.json` records the check of all 210 dataset hashes;
`checks.json` records local test results. Its Python environment is persistent.
To resume after the launcher exits, run from the repository root:

```sh
rtk proxy outputs/pilot/small-study-v1/launcher-env/bin/python -u \
  -m scripts.pilot.nautilus_study --state-dir outputs/pilot/small-study-v2 \
  --study-id mars-jepa-1120-v2 --cpu-only
```

This launcher runs on the local computer. Keep it awake and connected for stage
transitions. Cluster workers continue if the launcher stops; resume with the
same state directory instead of starting a new study.

Use a new `--study-id` when starting with new local state. Remote checkpoints
remain after Kubernetes removes old Job records. Reusing their study ID can
reach incompatible checkpoints; the resume check must keep rejecting them.
The restart retains prior compute charges within the original admission budget.
Local caches and temporary files belong under the repository's `.cache/`;
deployment scripts and records belong under `outputs/pilot/`.

After inspecting a failed GPU benchmark, `--cpu-only` resumes from the successful
CPU benchmarks without retrying the GPU job. This choice persists in the study
state. Failed GPU costs remain charged. It cannot switch a study that already
selected GPU training. The deployed study used this path after a GPU node evicted
the JEPA benchmark for low ephemeral storage; both CPU benchmarks had passed.

The immutable source ConfigMap contains a SHA-256 checked archive of the actual
implementation. Both CPU and GPU runtime images are resolved to digests before
submission. Exact installed packages are saved per task. No Git push is needed.
Submitted Jobs have no runtime deadline, container timer, or timed deletion.
Nautilus currently injects `ttlSecondsAfterFinished: 86400` and restores it after
removal attempts. This cluster policy deletes completed Job records after a day;
it does not limit running work. Persistent logs, checkpoints, and local records
remain available after that cleanup.

## Stages and budgets

1. Time GRU and JEPA on CPU and GPU: seven real training batches (discard the
   first two for the steady estimate), plus full validation. Benchmark admission estimates
   are 4 requested CPU-core-hours and 1 GPU-hour, included in the study totals.
2. Choose GPU only for at least 2x speedup and an affordable full-study forecast.
   Otherwise choose CPU if its forecast fits. Include startup and serial final
   evaluation in the estimate. Admit no further jobs if neither device fits.
3. Fit GRU, JEPA without latent loss, full JEPA, and linear ARX for five epochs,
   seed 0. Score the fixed EBM once. No test data are loaded.
4. Expand only if a neural model's validation RMSE is at least 5% below linear
   ARX. Extend seed 0 and add seeds 1 and 2 for all four learned variants, at
   most 20 epochs. Early stop after three epochs without 0.1% relative MSE
   improvement, after the first five epochs.
5. Freeze the decision and best validation checkpoints. Evaluate both test tiers
   once for every finished variant. If the gate fails, evaluate only seed 0.
   Report each result and paired JEPA differences; do not select a test winner
   and then restart tuning.

Admission budgets: 48 requested CPU-core-hours and 12 GPU-hours. GPU host CPUs
also count against the CPU budget. Pending/startup time is charged conservatively.
Each active job reserves estimated use against the remaining budget. Completed
jobs are charged their elapsed time. Running jobs may exceed estimates; this
blocks additional jobs and never cancels a running job.

Each task uses 2 CPU cores and 8 GiB host RAM. GPU tasks request one GPU. CPU
tasks avoid NVIDIA nodes. Training, validation, peak process memory, CUDA peak
memory, and startup time are recorded. Failed Jobs are not automatically retried.
Inspect their saved logs and termination state before authorizing more work.

## Fixed data and architecture

Use the existing 210-trajectory dataset. Use 30% of the 100 training trajectories,
stratified with subset seed 0, for all initialization seeds. Normalize from
those same 30 trajectories only. Retain all 20 validation trajectories.

Context remains 40 sols and forecast remains 120 sols. Window stride is 120,
giving 16 windows per 2,006-sol trajectory: 480 training and 320 validation
windows. This removes much repeated training work. It is a protocol change.

The pilot JEPA is `models.latent_jepa.JEPAModel`, not the separate convolutional
PI-JEPA. It encodes each 64-band temperature map with a 64→48→48 MLP; a GRU
combines that 48-value embedding with 64 sunlight values and one heat value into
58 memory values. A second GRU advances memory using sunlight and heat alone.
A 58→64→64 decoder returns temperature; a 58→48 projection supplies the latent
loss. The temperature target encoder is a frozen EMA copy, decay 0.99.

Temperature, sunlight, and heat are standardized. Latitude is implicit in band
order. Ice, subsurface temperature, dust, and simulator flux labels remain hidden.
The 48 learned features have no assigned physical meaning. Latent weight 0
removes its supervision without changing model shape; weight 1 is the full JEPA.
No physics loss or uncertainty samples are added.

The simulator stores state and inputs at the start of each interval. Therefore
the transition to temperature at index `t` uses inputs at `t-1`. The GRU already
follows this convention. This study sets `action_offset: -1` for JEPA, linear
ARX, and EBM. Old configurations keep their original convention for compatibility.

## Recovery and interpretation

Training saves `latest.pt` atomically after every epoch. It includes current and
best weights, EMA state, optimizer, epoch, validation curve, early-stop state,
Python/NumPy/Torch/CUDA randomness, shuffle generator, and collocation generator.
The exported `model.pt` is the best validation checkpoint. Resume checks the
source, settings, data manifest, selected observable contents, subset, and device.
An interrupted epoch is repeated from the previous full epoch.

Per-task status, progress logs, resolved settings, and checkpoints are on the
PVC. The local report includes all final scores and charged resource use.
The original tests were already evaluated by the old sweep. This study is
exploratory, has at most three initialization seeds on one fixed dataset, and
does not satisfy the old ten-seed hypothesis tests. Better JEPA results are not
guaranteed. A failed validation gate is a useful result.

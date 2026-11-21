# Frozen TorchRL comparison recovery

Issue: <https://github.com/sameeerkashyap/mars-jepa/issues/3>.

The collector charged the time from launch to local collection. A monitor outage
therefore charged completed jobs for nearly six hours. It also counted that delay
as startup time in later admission estimates.

Collection now charges from pod allocation to the last container finish. It sums
all retained pods for the Job. If pods have expired, retained Job start and finish
times give a conservative bound that includes scheduling. Missing evidence stops
collection without replacing the charge. Corrections retain the old charge,
startup estimate, correction time, and allocation evidence. State saves are
serialized because two collectors can finish together.

Monitor read failures retry reads of the same Job. A resume collects an existing
Job without reserving its cost a second time or launching it again. A failed
worker remains blocked. Saved successful evaluation records can rebuild the
paired report without evaluating test data again.

## Ledger correction on 2026-11-21

The original state is retained at
`outputs/pilot/torchrl-comparison-v1/state-before-accounting-recovery-20261121.json`.
The corrected state retains each original charge in `accounting_corrections`.

| Training seed | Original charged seconds | Corrected bound, seconds | Evidence |
|---|---:|---:|---|
| 0 | 21560.574904 | 533 | Observed Job duration 8m52s, plus one second for display rounding |
| 1 | 21559.350788 | 595 | Observed Job duration 9m54s, plus one second for display rounding |
| 2 | 416.081749 | 411 | Retained Job start 2026-11-21T03:07:20Z, completion 03:14:11Z |

Seed 0 and 1 Job durations were independently observed at about six hours and
again at 24 hours after launch. Their original Job and Pod objects expired under
the old automatic deletion setting. Exact pod allocation timestamps cannot be
recovered. The charges above are conservative duration bounds, not exact CPU use.
Seed 2's raw Job JSON is retained beside the corrected state. Earlier imported
baseline and pilot charges remain unchanged.

This correction restores 23.3316708004 CPU core-hours. Before final evaluation,
the cumulative allowance has 39.7161377773 CPU core-hours and 11.0237286641 GPU-hours
left. Startup bounds become 76.725, 89.554, and 61.650 seconds for seeds 0, 1, and 2.

The existing estimate admits all three CPU evaluations at 6.7264025673 CPU
core-hours. It uses 128 times each fit's slowest of the last three epoch times,
a 1.5 margin, and the largest corrected startup bound. This is a projection;
the final evaluation duration has not yet been measured. No extra benchmark was
needed for admission. The allowance decides launches; it never stops a running Job.

## Frozen evaluation

The correction did not read held-out data, retrain, or change checkpoints. The
archive SHA-256 remains
`a9a81e85467a0a461080312dac5636822bc1b3a3f5f314de5bc3bd5193446b5f`.
The existing `FROZEN.json` bytes and decision were checked before and after the
correction. Original data identity and artifact hashes remain fixed.

The once-only final evaluation resumed with two workers at most. Its persistent
controller uses the root `.venv-torchrl`, the frozen worker archive, and explicit
absolute paths to the original state, baseline, and previous ledger. The log is
`outputs/pilot/torchrl-comparison-v1/recovery-controller-20261121.log`; the PID file
is beside it. Seed 0 and 1 evaluation Jobs were running at handoff. Seed 2 follows
after those finish. No job deadline or timed deletion was added. Keep this ticket
open until the report is saved or a concrete worker failure is reported.

## Checks

35 focused tests pass across `test_small_study.py`,
`test_torchrl_comparison.py`, and `test_torchrl_dreamer.py`. The new accounting and
resume checks failed before their fixes. Coverage includes delayed collection,
transient monitor failure, active-task admission, recovery between charge and
log collection, and report recovery from saved results. Existing source,
checkpoint, data-identity, and no-deadline checks still pass.

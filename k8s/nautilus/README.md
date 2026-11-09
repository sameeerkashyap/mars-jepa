# MarsWorld sweep on Nautilus

This job generates the heating dataset, runs the registered **450 configurations**,
and writes an analysis report. It uses eight CPU workers, 20 training epochs,
three data fractions, ten fitting seeds, and both test tiers. The physics weights,
architectures, and statistical thresholds stay as committed.

The second test uses heating between **8 and 10 W/m²**. All 210 generated
trajectories have three recorded Mars years. Hidden simulator states remain
excluded from ordinary model inputs.

## Configuration

- Context: `nautilus`; namespace: `razvanlab`.
- Job: `marsworld-pilot-sweep`.
- Persistent volume: `marsworld-pilot-sweep`, 50 GiB, `rook-cephfs`.
- Compute: 8 CPU cores, 24 GiB memory; no GPU.
- Local scratch: 8 GiB. Python packages and source are installed on local scratch.
- The node rule avoids NVIDIA GPU nodes, following the
  [Nautilus CPU job guide](https://nrp.ai/documentation/userdocs/running/cpu-only/).
- Source is downloaded from a full Git commit. Python packages are recorded in
  the run's provenance folder; scientific runtime versions are pinned here.

The isolated volume keeps these experiments separate from other Mars projects.

## Start

After committing and pushing the experiment code, set `MARSWORLD_REVISION` in
`sweep-job.yaml` to that full commit. Use a new persistent run directory for a
different experiment.

```bash
kubectl --context nautilus apply -f k8s/nautilus/pvc.yaml
kubectl --context nautilus apply -f k8s/nautilus/sweep-job.yaml
kubectl --context nautilus -n razvanlab logs -f job/marsworld-pilot-sweep
```

The job exits after analysis. Kubernetes can retry it once after a failure.
Each completed model result is saved immediately. A restart skips successful
runs and retries failed runs. Smoke checks must use their own `--output-dir`.

## Outputs

The persistent run directory is `/results/pilot-20261109`:

```text
provenance/                 Source commit, settings, thresholds, installed packages
data/manifest.json          Dataset design and trajectory list
data/SHA256SUMS             Hashes of generated arrays
data/main/                  Training, validation, and two test splits
data/calibration/           Calibration trajectories
output/results.csv          One row per attempt; successful rows feed analysis
output/runs/                Model weights, training information, evaluation metrics
output/logs/                Per-run status and error logs
output/analysis_report.md   Hypothesis results and baseline comparisons
output/figures_phase2/      Error, rollout, and uncertainty figures
SUCCESS                     Written after the full sweep and analysis succeed
```

Watch progress with the job logs. Error rows are retained for diagnosis and
excluded from analysis. The sweep returns a nonzero exit code if any run fails.
Retain the volume after the job completes; it contains the experiment data and
checkpoints.

The 450 configurations vary model-fitting and subset seeds on one generated
dataset. They do not measure variation across independent reference climates.

# RUNME — Phase 2 (ML) pipeline

Exact commands to run each step, with expected output. Run from the project
root. Everything reads `config_phase2.yaml` (ML hyperparameters) and the frozen
`config.yaml` (physics). Hypothesis thresholds live in `thresholds.yaml`.

> **Data:** these commands assume the real datasets are under `data/` (the
> Checkpoint-B owner regenerates them). Every command below accepts
> `--data-dir <dir>` to point elsewhere. A tiny fake fixture with identical
> shapes can be produced with `python3 tests/fakedata.py data_fake` and used via
> `--data-dir data_fake` (this is what the smoke test below uses while `data/`
> is being regenerated).

Device is auto-selected (`cuda` → `mps` → `cpu`); override with `--device cpu`.

---

## 0. Environment

```
pip install -r requirements_phase2.txt   # numpy scipy pyyaml matplotlib pandas pytest torch
```

## 1. Tests (must all pass)

```
python3 -m pytest tests/ -q
```
Expected: `74 passed` (dataio + Checkpoint-A + split tests, Phase-2 tests, the
end-to-end `tests/test_integration.py` on 4 tiny trajectories generated through
`gen_data.py`, and a structure-only `diag_tier2.py` check; ~90 s on CPU). The Phase-2 tests cover: no hidden leakage, eval labels never imported by
train/models, train-only normalization, param counts within 10%, physics residual
~0 on an EBM trajectory (and forcing-aware), open-loop rollouts ignore future
ground truth, determinism, hybrid ice init from context only, calibration cannot
load test splits, and the stratified/nested/deterministic subsampler.

## 2. Train one configuration

```
python3 train.py --arm gru --physics off --fraction 1.0 --seed 0 --out runs/gru_off_f100_s0
```
Expected: prints an `info` JSON (arm, n_params, val_loss, lambda, ...) and
`saved -> runs/gru_off_f100_s0`. The run dir holds `model.pt`, `stats.json`,
`info.json`, `arch.json`.

Physics-loss variants (gru/rssm/transformer only): `--physics A` or `--physics B`.
Select lambda on validation with the shared grid: add `--select-lambda`.

## 3. Calibration (validation/calibration data ONLY; cannot touch test)

```
python3 calibrate.py --arms gru rssm transformer hybrid linear ebm_only \
    --seeds 0 1 2 --out calibration_report.json
```
Expected: writes `calibration_report.json` with, per arm, `n_params`,
`val_rmse_K_mean/std`, per-seed values, `seed_to_seed_spread_K`, per-seed
`val_loss_curves`, and `still_decreasing_at_final_epoch` (arms listed in
`still_decreasing_arms` and printed as a WARNING); curves plotted to
`calibration_report_curves.png`. (EBM-only and
linear baseline errors are included.) It is structurally unable to load a test
split (see the test `test_calibration_cannot_load_test`).

## 4. Sweep (I run this myself — resumable)

Full grid = arms × physics{off,A,B where applicable} × fraction{0.1,0.3,1.0} ×
10 seeds (390 runs). **Do not launch the full sweep from the agent.**

```
# smoke first (tiny grid):
python3 sweep.py --tiny --epochs 1 --max-batches 2 --device cpu

# full sweep, 8 CPU workers (resumable; rerun to continue):
python3 sweep.py --workers 8

# GPU: workers is forced to 1 automatically
python3 sweep.py --device mps
```
Expected: `[sweep] N pending / M total (K done) ...` then
`[sweep] wrote N rows -> results.csv`. One row per run in `results.csv`; per-run
logs in `logs/<tag>.log`; per-run full metrics in `runs/<tag>/eval.json`. Safe to
Ctrl-C and rerun — completed tags are skipped.

## 5. Analyze

Fill `thresholds.yaml` first (H1/H2/H3 criteria). Then:

```
python3 analyze.py --results results.csv --thresholds thresholds.yaml
```
Expected: prints the H1/H2/H3 verdicts (supported / not supported / inconclusive),
writes `analysis_report.{json,md}` (incl. a descriptive table of each
configuration's Tier-2 `t2_polar_winter_S_pinned` error next to LINEAR and
HYBRID, mean + per seed, no thresholds) and figures into `figures_phase2/`:
`headline_ood_vs_fraction.png`, `rollout_vs_horizon.png`, `calibration.png`.

## 5b. Tier-2 response diagnostic (I run this myself)

```
python3 diag_tier2.py --workers 5      # -> diag_tier2.json + printed table
```

## 6. Evaluate a specific run / ensemble on test (eval-only module)

```
python3 eval.py --run runs/gru_off_f100_s0 --out runs/gru_off_f100_s0/eval.json
# GRU ensemble (repeat --run for members, add --ensemble):
python3 eval.py --run runs/ens_m0 --run runs/ens_m1 ... --ensemble
```
Expected: prints Tier-1 and Tier-2 metrics separately (intervention-response
RMSE `polar_winter_{N,S}` and `polar_winter_{N,S}_pinned` with
window/cell-step counts, + global, rollout-vs-horizon, "EBM-residual constraint
satisfaction (also the training objective for physics arms)", and
calibration for stochastic/ensemble models).

---

### Full smoke (what the agent ran, on the fake fixture)

```
python3 tests/fakedata.py data_fake
for a in gru rssm transformer hybrid linear ebm_only; do \
  python3 train.py --arm $a --physics off --fraction 0.1 --seed 0 \
    --epochs 1 --max-batches 2 --data-dir data_fake --device cpu --out runs/smoke_$a; done
python3 train.py --arm gru --physics A --fraction 0.1 --seed 0 --epochs 1 \
  --max-batches 2 --data-dir data_fake --device cpu --out runs/smoke_gruA
python3 train.py --arm gru --physics B --fraction 0.1 --seed 0 --epochs 1 \
  --max-batches 2 --data-dir data_fake --device cpu --out runs/smoke_gruB
python3 calibrate.py --arms gru linear ebm_only --seeds 0 1 --epochs 1 \
  --max-batches 2 --data-dir data_fake --device cpu
python3 sweep.py --tiny --epochs 1 --max-batches 1 --data-dir data_fake \
  --results results.csv --device cpu
python3 analyze.py --results results.csv
```
All steps complete; see docs/README_phase2.md for measured per-arm times and the
projected full-sweep cost.

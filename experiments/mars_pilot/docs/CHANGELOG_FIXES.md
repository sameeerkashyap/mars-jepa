# CHANGELOG — fix pass (2026-10-17)

State at start of this pass: no CHANGELOG existed; none of fixes 1–8 had been
applied (checked each). Test suite: 36 passed (fake fixtures only).
State at end: **50 passed** (`python3 -m pytest tests/ -q`, ~90 s CPU).

## 1. Tier 2 = constant only (`gen_data.py`)
* New module constants `TIER2_SHAPES = ["constant"]`, `TIER2_MAG = [1.5X, 2X]`;
  `SPLITS["main"]["test_tier2"]` uses them (was `["constant", "ramp"]`).
* Magnitude range overridable with `--tier2-mag LO HI`.
* Checked every Tier-2 definition: only `gen_data.py:SPLITS` defines Tier 2.
  `_smoke_splits()` (~line 217) has no Tier 2 split, so nothing to change there.
  `diagnostics_tier2.py` (old, level-based) still loops over constant+ramp; it
  is historical evidence for PREREG D1 and is left unchanged.
* Test: `tests/test_splits.py::test_tier2_constant_only_and_out_of_range`.

## 2. Normalization from the trained subset (`train.py`, `data_windows.py`)
* `train_one` now computes `stats = compute_norm_stats(train_sub)` (the 10% / 30%
  / 100% subsample), not the full train split. Docs updated (README_phase2 §4, §10).
* Test: `tests/test_phase2.py::test_normalization_uses_trained_subset[0.1|0.3]`.

## 3. Integration test on real code paths (`tests/test_integration.py`)
* 4 tiny trajectories (seeds 8500–8503, spin 0.25 yr, record 0.3 yr) generated
  through `gen_data.run_generation` → `_worker` → `run_truth` into a pytest tmp
  dir; then dataio loaders + guards, `calibrate.py` CLI (all 6 arms), `train.py`
  CLI (6 arms + gru A/B + a 2nd gru seed), `eval.py` CLI per run and as a GRU
  ensemble, Tier 1 + Tier 2. Subprocess output is captured, not printed.
* What had to change to make it run:
  * `gen_data.py`: generation loop factored into `run_generation(out_root,
    splits, rec_years, workers, *, spin_years)`; `workers=0` runs in-process;
    `spin_years` is a per-task field (default `SPIN_YEARS=3`, recorded in the
    manifest); log file handle closed at the end. CLI behaviour unchanged.
  * `eval.py`: added `--device` (it always auto-picked, i.e. MPS on this Mac)
    and `--quiet` (write `--out` without printing metric values).
* Nothing else broke on real generated data.

## 4. One data API (`dataio.py`; `dio.py` → `legacy/dio.py`)
* `dataio.py` is the only data module. Added `load_split_eval_labels(data_root,
  family, split, *, calibration_mode=False)` (per-split, aligned with
  `load_split`, raises in calibration_mode, logs test-split access) and
  `subsample_train(trajs, fraction, seed)` (now a thin wrapper over
  `stratified_fraction_subsets`, which removes the duplicated algorithm).
* `load_trajectory` now drops any un-prefixed `m/T_g/tau/control_T_s/
  response_T_s` keys instead of asserting. This was the only thing `dio`'s strip
  added.
* `train.py`, `calibrate.py`, `eval.py`, tests import from `dataio` only.
  `train_one(..., calibration_mode=False)` is new; `calibrate.py` now passes
  `calibration_mode=True` into training too, where previously only its own
  loads used it.
* Guards kept and tested: default loader has no hidden/eval keys; calibration_mode
  cannot load test splits or eval labels; every test-split load (observables or
  labels) is logged with a UTC timestamp.
* Tests: `test_single_data_api`, `test_split_eval_labels_logged_and_aligned`;
  `test_eval_labels_not_imported_by_train_or_models` now also covers
  `calibrate.py` and the new function name.

## 5. PREREG.md Deviations (§12, D1–D4)
* D1 design-stage diagnostics touched Tier-2 outcomes; D2 Tier 2 → constant-only;
  D3 legacy data moved to `data_legacy/`, to be regenerated with stratified
  magnitudes; D4 the single logged test-split load
  (`2026-10-17T20:51:28Z main/test_tier2`, from `data_legacy/test_access.log`).
* §4 Tier-2 bullet got a one-line pointer to D2. No frozen values changed.

## 6. EBM metric label + numberless thresholds
* Label "EBM-residual constraint satisfaction (also the training objective for
  physics arms)" is now in `eval.py` (docstring, `EBM_METRIC_LABEL`, and an
  `energy_budget_violation_label` field in every tier's metrics), in `analyze.py`
  (it previously did not report this metric at all; now a labeled per-config
  Tier-1/Tier-2 table in `analysis_report.{md,json}`), README_phase2 §7, RUNME.
  The metric key `energy_budget_violation_mean` and the results.csv columns
  `t{1,2}_ebm_viol` are unchanged, so sweep.py stays compatible.
* `thresholds.yaml`: all 9 fields blank (null), with no example numbers;
  `h2_min_seeds: 5` was also blanked. `analyze.py` falls back to `t2_resp_S` / `gru`
  / `A` for figures when fields are blank, and every verdict is "inconclusive
  (threshold unset)" until filled.
* Test: `test_thresholds_blank_and_ebm_metric_labeled`.

## 7. calibrate.py validation-loss curves
* `train_one(..., record_curve=True)` records val loss per epoch (`info["val_curve"]`).
* `calibrate.py` saves per-arm, per-seed `val_loss_curves`,
  `still_decreasing_per_seed`, `still_decreasing_at_final_epoch`, top-level
  `still_decreasing_arms`; plots `<out>_curves.png`; prints a WARNING listing
  flagged arms. "Still decreasing" = last epoch's val loss < previous × (1 −
  `--rel-tol`), default rel-tol 0. With fewer than 2 epochs the flag is `null`.
* Test: `test_calibrate_saves_curves_and_flags` (+ integration test).

## 8. `diag_tier2.py` (written, NOT run)
* Response-based: forced − same-seed control, record_years=3, constant forcing
  A ∈ {7.5, 8, 9, 10}, seeds 7000–7004, N/S polar bands |lat|>60. Polar winter =
  steps where band-mean insolation ≤ 10% of its record max (observable-only;
  `--winter-s-frac`). Reports per (A, hemisphere): signal (seed-mean winter-mean
  response), seed std, S/N, peak, fraction of control-pinned cells unpinned when
  forced. Writes `diag_tier2.json`.
* Test: `tests/test_diag_tier2.py`, a structure-only check on a tiny run (0.05 yr
  spin, 1 yr record). The real diagnostic was not run.

## Files moved
* `dio.py` → `legacy/dio.py`.

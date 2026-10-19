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

---

# CHANGELOG — metric-alignment pass (2026-10-17, later)

## 9. `metric_defs.py` (new) — single source of truth for polar-winter masks
* `BAND_DEG = 60.0`, `WINTER_S_FRAC = 0.10`; `polar_bands(lat_rad)`,
  `seasonal_ref(S_full, band)`, `winter_steps(S, band, frac, s_ref)`.
  Observable-only (insolation + latitude).
* Tests: `tests/test_metric_defs.py`.

## 10. `diag_tier2.py` uses `metric_defs`; seed counts for `frac_unpinned`
* Own band / winter-mask code replaced by `polar_bands` / `winter_steps`
  (identical definition: strict `>60` / `<-60` deg, `<= frac * record max`).
* `summarize` now reports, per (A, hemisphere): `frac_unpinned_n_seeds` (seeds
  with a control-pinned winter band cell), `n_seeds_no_ctrl_pinned`, and
  `frac_unpinned_min`. `frac_unpinned` is the mean over contributing seeds only
  (same value as the previous `nanmean`, but the dropped seeds are now counted).
  The table printout has a second block with these counts.
* Tests: `test_diag_tier2_counts_seeds_without_pinned_cells`,
  `test_diag_tier2_uses_metric_defs`.

## 11. `eval.py` — polar-winter metric restricted to winter steps (PREREG D6)
* Before: `polar_N/S` = RMSE over ALL H steps of each rollout window in the
  band, i.e. not winter-restricted.
* Now: bands from `metric_defs.polar_bands`; per trajectory and hemisphere,
  `s_ref = seasonal_ref(full trajectory S)`; per window the horizon's winter
  steps are `winter_steps(S_horizon, band, s_ref=s_ref)`. `polar_N/S` = RMSE over
  winter-step band cells only; a window with no winter steps → NaN, skipped by
  the mean.
* New `polar_N_pinned` / `polar_S_pinned`: same, restricted to cell-steps whose
  TRUE `control_T_s <= T_frost + PIN_TOL_K` (`PIN_TOL_K = 0.5`, T_frost from
  `common.frost_point` and config). `control_T_s` is used only in `eval.py`.
* New `intervention_response_counts`: per metric, `n_windows` (finite windows)
  and `n_cell_steps`; plus `n_windows_total`, `pin_tol_K`, `T_frost_K`.
* Removed `POLAR_DEG`, `_polar_masks`, `_band_rmse` (replaced by
  `_winter_masks`, `_masked_rmse`, `_nanmean`). `sweep.py` keys
  (`polar_N`, `polar_S`, `global_mean`) are unchanged, so results.csv is
  compatible; their meaning changed (winter-only).
* Tests: `tests/test_eval_metrics.py` (mask = metric_defs on a fixture, s_ref
  from full trajectory, winter-only differs from all-steps under large summer
  error, pinned metric uses only control-pinned cells, no eval-label references
  in train/calibrate/data_windows/physics_torch/metric_defs/models).

## 12. `diag_linearity.py` (new; simulator only)
* Same structure as `diag_tier2.py` (top-level `_seed_job`, spawn
  ProcessPoolExecutor, `--workers --seeds --out`, frozen-design asserts).
  Per seed: spin, control, constant A ∈ {1..5} (train) and {7.5, 8, 9, 10}
  (test), record 3 yr. Groups band / pinned / rest over polar-winter cell-steps;
  through-origin fit on train A, extrapolated to test A; true, predicted, error,
  relative error, seeds contributing, and `seeds_missing`. Writes
  `diag_linearity.json`.
* Test: `tests/test_diag_linearity.py` (structure only; no simulator run).

## 13. PREREG.md — appended D5, D6, D7 to §12 (no existing text edited;
  verified by diff against a pre-edit copy: additions only). All numbers and
  thresholds left blank (`____`).

## 14. Runs and files moved
* Pre-existing `diag_tier2_20seeds.{json,log}` (20:48, written by the old
  diag_tier2.py, not by this pass) moved to
  `legacy/diag_tier2_20seeds_prev_2048.{json,log}` before re-running.
* Ran in the background (simulator only): `diag_tier2.py --workers 5 --mags 7.5
  8.0 --seeds 7000..7019 --out diag_tier2_20seeds.json`, and
  `diag_linearity.py --workers 5`. Logs: `diag_tier2_20seeds.log`,
  `diag_linearity.log`.

## 15. Existing test changed: `tests/test_integration.py::_check_eval`
* It asserted `polar_N` and `polar_S` finite. With the winter-only metric
  (item 11, requirement "no winter steps → NaN"), the 0.3-yr scratch record
  (one window) has no winter step in at least one hemisphere, so the assertion
  failed (2 tests). Replaced with: `global_mean` finite; for each polar metric
  (incl. `_pinned`), finite ⇔ `n_windows > 0` ⇔ `n_cell_steps > 0`, and
  `n_windows <= n_windows_total`. No metric values were printed or inspected.
* Suite: **66 passed** (`python3 -m pytest tests/ -q`).

---

# CHANGELOG — polar-winter naming / counts / baselines pass (2026-10-17, later)

## 16. Renamed polar-winter result keys and columns
* `eval.py`: `polar_N/S` → `polar_winter_N/S`, `polar_N/S_pinned` →
  `polar_winter_N/S_pinned` (now module-level `eval.POLAR_KEYS`) in
  `intervention_response_rmse` and `intervention_response_counts`.
* `sweep.py` results.csv: `t{1,2}_resp_{N,S}` → `t{1,2}_polar_winter_{N,S}`.
  `t{1,2}_resp_global` unchanged (not a polar metric).
* `analyze.py` fallback headline metric `t2_resp_S` → `t2_polar_winter_S`.
* Tests (`test_integration.py`, `test_phase2.py`), README_phase2 §7, RUNME,
  thresholds.yaml comments updated. thresholds.yaml values still all blank.
* Not renamed: internal mask keys of `eval._winter_masks` ("N", "S_pinned", …),
  historical CHANGELOG entries, PREREG.md (D6/D7 still say `polar_S_pinned`;
  it is the same quantity as `polar_winter_S_pinned`).
* `sweep._append_row` now refuses to append to a results.csv whose header
  differs from the current `FIELDS` (e.g. an older all-step results.csv).

## 17. Window / cell counts for `*_pinned`; results.csv counts
* Checked: a window counts toward `polar_winter_{N,S}_pinned` only if its
  horizon has ≥ 1 winter-step band cell whose true control `T_s` ≤ T_frost +
  `PIN_TOL_K`; otherwise its value is NaN and it is excluded from the mean and
  from `n_windows`. This was already correct in eval.py; no change.
* Fixed: results.csv did not record the `_pinned` metrics nor any counts. It
  now has, per tier and polar metric, `t{1,2}_<key>`, `t{1,2}_<key>_n_windows`,
  `t{1,2}_<key>_n_cell_steps` (helper `sweep.polar_columns`). `n_cell_steps`
  counts (cell, time-step) pairs, as in eval.json.
* eval.py metrics now also record `protocol: {K, H, stride}`.
* Tests: `tests/test_polar_winter_results.py` — runs `eval.evaluate()` on a
  synthetic trajectory (loaders monkeypatched) and checks counts against an
  independent computation from `metric_defs`, including a winter window with no
  pinned cells (excluded) and the exact pinned RMSE; results.csv columns and
  header guard.

## 18. analyze.py: primary metric vs LINEAR and HYBRID (descriptive)
* `primary_vs_baselines(df)`: per (configuration, fraction), mean and per-seed
  `t2_polar_winter_S_pinned` next to `linear/off` and `hybrid/off` at the same
  fraction. No thresholds, no verdicts. Written to `analysis_report.json`
  (`primary_vs_baselines`) and as a table in `analysis_report.md`.
* Test: `test_primary_vs_baselines_table` (fake results).

## 19. LINEAR / EBM-only scored like the learned arms
* Checked: every arm goes through the same `eval.evaluate()`; K/H come from the
  run's `arch.json` (written from the shared `config_phase2.yaml` protocol);
  stride = H; windows, masks and metric code have no per-arm branch. Only
  differences are in how predictions are produced: stochastic arms (RSSM) use
  the mean of `n_samples` draws and the ensemble the member mean; LINEAR,
  EBM-only, GRU, hybrid, transformer are deterministic, single rollout. No
  protocol difference found, nothing changed.
* Test: `test_every_arm_scored_with_same_protocol[linear|ebm_only|gru|hybrid]`
  — same protocol, window count and per-metric counts as the independent
  computation.

Suite: **74 passed** (`python3 -m pytest tests/ -q`, ~80 s).

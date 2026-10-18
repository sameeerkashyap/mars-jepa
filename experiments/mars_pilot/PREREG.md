# Pre-registration — Mars 1-D climate pilot (DESIGN ONLY)

> **Status:** design pre-registration for the ML phase. **All hypothesis
> thresholds are left BLANK (`____`) for the reviewer to fill and commit.**
> No models have been trained; Checkpoint B (data generation) has not started.
> Frozen design parameters below must not change after this file is committed.

---

## 1. Objective

Test whether an **incomplete** physics prior (the simplified EBM, which omits dust
and the subsurface reservoir but keeps the CO2 frost clamp) helps a learned model
**extrapolate to unseen radiative forcings** on a Mars-like 1-D climate, relative
to a pure data-driven model.

### Hypotheses (criteria blank — reviewer to fill)

* **H1 (an OOD gap exists).** Pure data-driven models trained on in-distribution
  forcings degrade on out-of-distribution forcings.
  * Supported if: Tier-2 error / Tier-1 error ratio ≥ `____`, on metric `____`.
* **H2 (incomplete physics helps extrapolation).** Adding the EBM physics-residual
  loss improves OOD performance vs the same architecture with data loss only.
  * Supported if: relative OOD-error reduction ≥ `____` on metric `____`,
    significant across `____` dataset seeds.
* **H3 (physics helps data efficiency / hybrid helps).** The physics prior (as a
  loss or as a hybrid backbone) reduces the data needed to reach a target error,
  and/or the hybrid arm beats both pure-data and physics-loss arms OOD.
  * Supported if: `____`.

Primary and secondary **metrics** are defined in §6; the numeric thresholds in the
`____` fields are the reviewer's to set before any training.

---

## 2. Models (reference; already built & tested in Checkpoint A)

* **Truth (Model 1, `sim_truth.py`).** Surface `T_s` + subsurface `T_g`; hidden
  dust `τ(x,t)` (seasonal + storms); CO2 frost clamp; conservative FV diffusion;
  uniform-in-latitude forcing `F_a(t)`. `mismatch_scale` scales **both** dust and
  subsurface coupling `G`.
* **Simplified EBM (Model 2, `sim_ebm.py`).** Single `T`; no dust, no subsurface;
  **same CO2 frost clamp**; `C_eff, D, ε` fit by linear least squares to the
  unforced `scale=0` truth run. Supplies the physics residual `residual(...)`.

---

## 3. Frozen design parameters (do not change after commit)

| parameter | value | rationale |
|---|---|---|
| grid `n` | **64** | threshold location robust 64→128 (cap *mass* is not; see DIAGNOSTICS_C.md) |
| diffusion `D` | **0.10 W/m²/K** | places the frost threshold near the top of the forcing range; TUNED (see §9 limitation) |
| `mismatch_scale` | **0.10 primary** | physics residual ≈ forcing magnitude (8.6% of absorbed SW) |
| | **0.25 exploratory** | different truth world (no N cap; threshold inside training) — NOT a pre-registered hypothesis arm |
| timestep `dt` | **0.05 sol** (production) | energy-closure and stability verified |
| all other physics | `config.yaml` as committed | uncertain params flagged therein |

Everything in `config.yaml` is frozen at the committed revision for Checkpoint B.

---

## 4. Forcing design

* **Forcing magnitude knob X = 5 W/m².**
* **Train split:** magnitudes in **[0, X] = [0, 5]**.
* **Tier 1 test:** **unseen schedule shapes** (ramps, steps, pulses) with
  magnitudes **inside [0, 5]** — tests generalization to new *shapes*,
  within-regime.
* **Tier 2 test:** magnitudes **[1.5X, 2X] = [7.5, 10]** — past the frost
  threshold; tests extrapolation **across the regime change**. *(Shapes changed
  to constant-only after design-stage diagnostics — see §12 Deviations, D2.)*
* **Schedule types:** constant, linear ramp, step, and pulse. Train and test use
  disjoint shape/seed sets; each run has a **matched no-intervention control**
  (same seed and initial condition).
* **Seeds:** multiple dataset seeds (for error bars); fixed and non-overlapping
  between splits. Exact seed lists to be recorded in the manifest at generation.

---

## 5. Observable vs hidden (locked)

* **Observable (model input):** `T_s`, `S`, `F_a`.
* **Hidden (diagnostics; clearly marked not-for-input):** CO2 ice mass `m`,
  subsurface `T_g`, dust `τ`, truth flux terms.
* Consequence (intended): while a pole is frost-pinned, `T_s` is flat and the cap
  state is invisible, so the model must carry it in latent state from the forcing
  history. This is a deliberate test of latent memory and makes Tier 2 hard.

---

## 6. Metrics

**Primary:** intervention-response error on **polar-winter `T_s`**, band
**|lat| > 60°**, **reported per hemisphere** (the band is ~1 cell N, ~2–3 cells S).

**Secondary:** global-mean `T_s` error (near-linear in `F_a`; see §9).

Intervention response = `T_s(forced) − T_s(matched control)` at the same time,
seed, and initial condition.

### Measured context for the metric (facts, not thresholds)

Unpinning threshold = lowest `F_a` at which the winter pole leaves the frost point
(scale 0.10, D=0.10):

| hemisphere | storms off | storms on (5 seeds) | relation to training [0,5] |
|---|---|---|---|
| N | ≈ **1** | ≈1, std ≤0.08 K | unpins **inside** training (not an OOD-regime test) |
| S | ≈ **5.5** | ≈5.5–6, std up to 0.6 K | unpins **at the train/test boundary** |

**Implication:** the **S pole is the genuine across-regime extrapolation case**
(training never sees it unpinned); the N pole unpins within training, so its Tier-2
behaviour is magnitude-extrapolation within a regime the model has already seen.
Report both; treat the **S-pole polar-winter `T_s` in Tier 2 as the headline OOD
metric**, with N and global-mean as context.

Polar-winter signal-vs-storm-noise (scale 0.10, Tier 2): N S/N ≈ 60–120 (clean);
S S/N ≈ 3–8 (usable). See DIAGNOSTICS_C.md §D.

---

## 7. Phase-2 model arms

All arms observe only {`T_s`, `S`, `F_a`}; all predict `T_s` trajectories.

1. **Pure data-driven — GRU.** Recurrent sequence model, data loss only.
2. **Pure data-driven — RSSM.** Latent state-space model, data loss only.
3. **Physics-loss variants of 1 & 2.** Same architectures with the EBM
   physics-residual loss added (see §8).
4. **Hybrid (physics backbone + learned residual).** The fitted simplified EBM
   **including the frost clamp** is the forward backbone; a learned network
   `g_θ` predicts a correction to the surface tendency:
   `C_eff dT/dt = [EBM RHS: S(1-A) − εσT⁴ + D·diff + F_a] + g_θ(features)`,
   with the CO2 frost clamp applied to the combined update. `g_θ` sees only
   observable features (e.g. recent `T_s`, `S`, `F_a`, latitude). The EBM core
   is **fixed** (fitted `C_eff, D, ε`); only `g_θ` is trained. This arm tests
   whether handing the model the known-incomplete physics as scaffolding helps
   extrapolation. Run for the physics-on configuration; optionally with the §8
   physics loss on `g_θ`'s residual.
5. **Linear baseline — ridge / ARX linear state-space.** Added because the
   global-mean response is near-linear; if the linear model matches the nonlinear
   architectures on Tier 1, that is itself a finding and a necessary control.

Primary architecture comparisons for H2/H3: GRU and RSSM (and the hybrid). The
`mismatch_scale = 0.25` exploratory world is run only for the **best** architecture
at a single data fraction, physics on/off, and reported as a different truth world.

---

## 8. Physics-loss evaluation — two variants

* **Variant A (STRICT, main H2).** The EBM residual loss is evaluated **only on
  the labeled training-range trajectories** (magnitudes [0, 5]), with frost-pinned
  cells masked. This is the clean extrapolation test: physics is never shown
  anything about the test range. **A null OOD result under A is a likely and
  reportable outcome.**
* **Variant B (COLLOCATION, labeled variant).** In addition to A, the EBM residual
  is penalized on **model rollouts under unlabeled forcings sampled up to
  F_a ≈ 10** (no truth labels used — only the physics residual). This asks a
  different question ("does physics let you exploit unlabeled out-of-range
  forcings?") and uses knowledge of the test range; it is reported as a clearly
  labeled variant, not the main H2.

The residual is always masked on frost-pinned cells (`residual(...)` already does
this via the frost mask).

---

## 9. Data-efficiency design

For each arm, train at several training-set fractions (grid to be fixed at
generation, e.g. logarithmically spaced) and plot OOD error vs data fraction,
with error bars across dataset seeds. H3's data-efficiency criterion uses these
curves (threshold `____`).

---

## 10. Limitations (pre-registered now, not discoveries)

* **D was tuned** (0.30 → 0.20 → 0.10) to place the frost threshold inside the
  forcing range; the threshold swings wildly with D (appendix table: `>8 / ≈6 /
  ≈0 / none` for D = 0.07 / 0.10 / 0.15 / 0.20). **Tier 2 is a *designed* regime
  change, not a discovery.**
* **Caps under-extend vs Mars** (~67–80° here vs a literature value quoted from
  memory as ~50–55°) and seasonal CO2 exchange (~6–17 kg/m²) is below the
  tens-of-kg/m² scale cited. **These Mars reference numbers are unverified (from
  the agent, not a source) and must be checked against literature before the
  paper.**
* **No atmospheric heat/CO2 transport and no pressure feedback.** The frost point
  is a single global value at fixed pressure. The +5–6 W/m² threshold is a
  property of this toy and is not a real Mars number.
* **Cap mass is resolution-sensitive** (~2× between n=64 and n=128); only the
  threshold location is grid-robust.
* **`mismatch_scale` is not a pure "amount-of-wrongness" knob:** it changes the
  truth climate (via the G heat-buffer it suppresses the cap). 0.10 and 0.25 are
  different worlds, hence 0.25 is exploratory only.

---

## 11. Analysis plan (thresholds BLANK)

| hypothesis | metric | split | criterion (reviewer fills) |
|---|---|---|---|
| H1 | `____` | Tier 2 vs Tier 1 | `____` |
| H2 (Variant A) | `____` | Tier 2 | `____` |
| H2 (Variant B) | `____` | Tier 2 | `____` |
| H3 | `____` | data-efficiency curve | `____` |

Statistical treatment (test, correction, number of dataset seeds) to be fixed by
the reviewer in the same commit. Analysis is run once, after training, on the
held-out splits exactly as defined here.

---

## 12. Deviations (logged after the original commit)

Each entry records what changed, when, and whether any Tier-2 outcome was seen
before the change. No model had been trained on generated (non-fake) data at
the time of any entry below; only fake-fixture smoke runs existed.

* **D1 — Design-stage simulator diagnostics touched Tier-2 outcomes.** Before any
  model was trained, truth-simulator diagnostics computed polar-winter signal and
  storm noise at Tier-2 magnitudes (DIAGNOSTICS_C.md §D; `diagnostics_tier2.py` →
  `diagnostics_tier2.json`, 2026-10-17). These used fresh simulations with
  dedicated diagnostic seeds (7000–7004), not the Tier-2 test files, but they are
  outcome information about the Tier-2 regime and informed D2.
* **D2 — Tier 2 changed to constant-only.** Tier-2 shapes changed from
  {constant, ramp} to **{constant}** (`gen_data.py`: `TIER2_SHAPES`). Reason: the
  ramp `A·min(t/T,1)` reaches full forcing only at the end of the record, after
  the polar winters, so ramp trajectories carry no polar-winter signal for the
  headline metric. The magnitude range [1.5X, 2X] is unchanged and remains a
  parameter (`TIER2_MAG`, CLI `--tier2-mag`). A response-based confirmation
  diagnostic (`diag_tier2.py`) was written; its results are to be appended here.
* **D3 — Legacy data moved; regeneration with stratified magnitudes.** The
  earlier dataset (manifest created 2026-10-17T20:50Z; Tier 2 = constant+ramp)
  was moved to `data_legacy/` and is not used for any analysis. The dataset is to
  be regenerated with `gen_data.py` (stratified magnitudes per shape within each
  split, constant-only Tier 2). Regeneration is run by the reviewer.
* **D4 — One early test-split load.** `data_legacy/test_access.log` records one
  load, `2026-10-17T20:51:28Z  TEST-LOAD  main/test_tier2`, made during a loader
  sanity check on the legacy data, before any model was trained. No eval output
  or run directory exists from that time, i.e. no test metric is on record from
  it; the legacy data are not used for analysis (D3).


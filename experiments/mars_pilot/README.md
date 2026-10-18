# Mars 1-D Climate Pilot — Checkpoint A

A small, self-contained latitude-only (1-D) climate model of Mars surface
temperature, built at **two levels of fidelity**, plus the machinery to measure
how *incomplete* the simplified physics is. This is **Phase 1** of a research
pilot on whether incomplete physics helps learned models extrapolate to unseen
forcings. **No ML models are included here.**

> Checkpoint A only. Checkpoint B (data generation) has **not** been started.

## Contents

| File | Purpose |
|------|---------|
| `common.py` | Shared physics: grid, insolation, diffusion operator, CO2 frost, config loader |
| `sim_truth.py` | **Model 1** — the richer "truth" simulator (generates all data later) |
| `sim_ebm.py` | **Model 2** — the simplified EBM; fit + residual function |
| `config.yaml` | All parameters; uncertain ones flagged `# UNCERTAIN` with ranges |
| `tests/test_checkpoint_a.py` | Conservation / closure / periodicity / determinism tests |
| `mismatch_report.py` | Model-2 residual magnitude vs `mismatch_scale` (prints + JSON) |
| `make_plots.py` | The three required figures (into `figures/`) |

Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest` only.

```
python3 -m pytest tests/     # run the test suite
python3 mismatch_report.py   # prints the mismatch table, writes mismatch_report.json
python3 make_plots.py        # writes figures/*.png
```

---

## Grid and shared inputs

* **State grid:** zonal-mean state on a uniform grid in `x = sin(latitude)`,
  **64 cell centres** on `[-1, 1]`. The area weight of a cell is proportional to
  its width in `x` (because the surface-area fraction of the sphere between
  `sin(lat)` and `sin(lat)+dx` is exactly `dx/2`). Weights are uniform here and
  normalised to sum to 1.

* **Insolation `S(x, t)`:** daily-mean insolation computed from Mars obliquity
  (25.19°), eccentricity (0.0934) and a 668.6-sol year. Time `t=0` is perihelion;
  Kepler's equation is solved (Newton) for the true anomaly, from which the solar
  longitude `Ls = Ls_perihelion + true_anomaly` and the orbital distance factor
  `(a/r)^2` follow. The standard daily-mean formula

  `S = (S0/π)(a/r)^2 [H0 sinφ sinδ + cosφ cosδ sinH0]`

  is used, with the sunset hour angle `cos H0 = -tanφ tanδ` clamped to `[-1, 1]`.
  **Polar night is handled exactly** (`H0 = 0 ⇒ S = 0`); polar day gives `H0 = π`.

* **Forcing `F_a(t)`:** one scalar added radiative forcing (W/m²).
  **ASSUMPTION:** it is applied **uniformly in latitude** (added to every cell's
  surface energy budget equally).

* **Diffusion:** uses the conservative finite-volume form
  `D d/dx[(1 - x²) dT/dx]` with **zero flux at `x = ±1`**. The interface factor
  `(1 - x²)` is exactly zero at the poles, so no explicit boundary term is needed
  and the operator **integrates to zero globally** (checked to machine precision,
  test *a*). Diffusion is advanced with an unconditionally-stable backward-Euler
  sub-step whose matrix rows sum to zero, so heat is conserved exactly by the
  implicit solve as well.

* **CO2 frost clamp (both models):**
  * fixed surface pressure (`610 Pa`, flagged; **pressure feedback is EXCLUDED**);
  * frost point from an inverted CO2 vapor-pressure fit
    `p = A·exp(-B/T) ⇒ T_frost = B / (ln A - ln p)`, giving **≈ 148 K at 610 Pa**,
    consistent with Mars polar CO2 frost temperatures (coefficients flagged);
  * if a cell would fall below `T_frost`, it is **held at `T_frost`** and the
    energy deficit is converted into CO2 ice mass `m ≥ 0` via latent heat `L`;
    surpluses **sublimate** ice back. Exact per-cell bookkeeping:
    `C·(T_new − T_tentative) = L·Δm`.

### Energy sign convention (important for the closure test)

The surface budget is `C_s dT_s/dt = NetFlux + L dm/dt` (latent heat is *released*
when frost deposits, `dm/dt > 0`). Rearranged, `d/dt(C_s T_s) − L dm/dt = NetFlux`.
We therefore define the **latent energy stored in ice as `E_ice = -L·m`**
(depositing gas→ice releases heat, so ice carries negative energy relative to gas).
The global closure that the test checks is:

`Δ(area-weighted heat) + Δ(E_ice) = ∫ NetFlux dt`
i.e. `Δ[Σ wᵢ(C_s T_s + C_g T_g)] − L·Σ wᵢ Δmᵢ = ∫ NetFlux dt`,

where `NetFlux = S(1-A)exp(-τ) − εσT⁴ + F_a` (diffusion and the surface↔subsurface
coupling both integrate to zero globally and drop out).

---

## Model 1 — Truth simulator (`sim_truth.py`)

Surface temperature `T_s` plus a subsurface reservoir `T_g`:

```
C_s dT_s/dt = S(1-A) exp(-τ(x,t)) - ε σ T_s⁴ + D d/dx[(1-x²) dT_s/dx]
              - G (T_s - T_g) + F_a(t)          (+ latent heat during frost)
C_g dT_g/dt = G (T_s - T_g)
```

* **`τ(x, t)` — hidden dust optical depth.** A seasonal baseline (peaks near
  perihelion, `Ls ≈ 251°`) plus optional random **dust-storm pulses**
  (Poisson-distributed in number, Gaussian in time, with a latitudinal shape
  biased to the southern hemisphere). Storms are drawn **deterministically from
  the trajectory seed**. `τ` is saved for diagnostics but **never given to any
  model**.
* **Subsurface coupling.** `C_g ≫ C_s` and conductance `G` (all flagged).
* **`mismatch_scale ∈ [0, 1]`** scales **both** the entire dust optical depth
  (baseline + seasonal + storms) **and** the coupling `G`. At
  `mismatch_scale = 0`: `τ = 0 ⇒ exp(-τ) = 1` and `G = 0` (subsurface decoupled),
  so the truth **reduces exactly to Model 2** with `C_eff = C_s`.
* **Albedo** `A = A_ice` where ice is present (`m > 0`), else `A_bare`.

**Numerics:** operator splitting per step `dt` — (1) explicit forward-Euler source
update, (2) implicit conservative diffusion sub-step, (3) CO2 frost clamp.
Default `dt = 0.05` sol (well within explicit stability for the ≈3-sol radiative
and ≈3-sol coupling timescales; diffusion is implicit and unconditionally stable).

---

## Model 2 — Simplified EBM (`sim_ebm.py`)

```
C_eff dT/dt = S(1-A) - ε σ T⁴ + D d/dx[(1-x²) dT/dx] + F_a(t)   (+ frost clamp)
```

No subsurface, no dust. `C_eff`, `D`, `ε` are **fit by linear least squares** to
the **unforced `mismatch_scale = 0` truth run** (where the truth *is* Model 2).
The fit is linear in the three unknowns:

`[dT/dt, -diff, σT⁴]·[C_eff, D, ε]ᵀ = S(1-A)`,

solved over all frost-free cells/times. With the defaults it recovers
`D ≈ 0.0998`, `ε ≈ 0.950` essentially exactly and `C_eff ≈ 4.9e5` (vs `C_s = 5.0e5`;
the small gap is operator-splitting + finite-difference discretization), at an
RMS fit error of **≈ 0.5–0.7 W/m²**. The fit error is *not* zero: with the current
`D = 0.10` regime the shared CO2 frost clamp forms seasonal caps even at
`mismatch_scale = 0`, and that nonlinearity cannot be captured by a linear EBM —
so `scale = 0` is not a perfect oracle (see `DIAGNOSTICS.md`).

`residual(T, S, F_a, dt, params, cfg)` returns the Model-2 residual for **any**
temperature trajectory, with **frost-pinned cells masked out** (set to NaN; a cell
is "frost-pinned" when `T ≤ T_frost + 0.5 K`).

---

## Checkpoint A results

### Tests (`python3 -m pytest tests/`) — all pass

| Test | What it checks | Tolerance |
|------|----------------|-----------|
| a | Diffusion alone conserves area-weighted global heat | `< 1e-9` relative (machine precision) |
| b | Truth energy closure: `Δheat + Δ(E_ice) = ∫NetFlux` | `< 1e-8` relative |
| c | Ice mass never negative; ice only where `T ≤ T_frost` | exact |
| d | Unforced run reaches a periodic seasonal cycle | `< 0.5 K` area-wtd RMS year-over-year |
| e | `mismatch_scale = 0` ⇒ truth ≈ Model 2 | fit RMS `< 1.0 W/m²`; forward RMS `< 2 K`/2 yr |
| f | Determinism for a fixed seed; different seed differs | exact |

### Mismatch report (`mismatch_report.py`)

Model-2 residual RMS (W/m²) evaluated on truth trajectories, EBM fit once on the
`scale = 0` unforced run, forced column uses `F_a = 10 W/m²`:

| `mismatch_scale` | unforced | forced |
|---|---|---|
| 0.00 | 0.46 | 0.05 |
| 0.25 | 19.7 | 19.5 |
| 0.50 | 33.5 | 33.2 |
| 1.00 | 54.6 | 53.8 |

The residual is small at `scale = 0` (truth ≈ Model 2, bounded below by the shared
frost nonlinearity) and grows monotonically — this is the quantitative "how
incomplete is the physics" signal. `diagnostics.py` / `DIAGNOSTICS.md` extend this
to `scale ∈ {0, 0.05, 0.1, 0.25, 0.5, 1.0}` with the absorbed-SW fraction and the
ratio to the forcing, which is what sets the operating `mismatch_scale`.

### Figures (`figures/`)

* `truth_T_vs_lat_season.png` — truth `T_s` vs latitude and `Ls` (unforced).
* `polar_ice.png` — seasonal polar CO2 ice caps. Caps grow in each hemisphere's
  winter and sublimate in summer; eccentricity makes them asymmetric, and at the
  default `mismatch_scale = 1.0` the heavily-dusted N pole retains a perennial cap.
  (At the proposed operating `mismatch_scale = 0.1` both caps are fully seasonal.)
* `truth_vs_ebm_forced.png` — truth (solid) vs fitted Model 2 (dashed) global-mean
  `T_s` under constant `F_a ∈ {2, 5, 10, 20} W/m²`. Model 2 runs warm because it
  lacks dust; storm-driven cold dips appear only in the truth.

---

## Assumptions (complete list)

1. Zonal-mean, latitude-only state; uniform grid in `x = sin(lat)`.
2. Area weights ∝ cell width in `x` (exact for a sphere).
3. `F_a(t)` is **uniform in latitude** and added to the surface budget.
4. Daily-mean insolation (no diurnal cycle); the surface `C_s` represents a
   seasonal effective heat capacity, not a diurnal skin.
5. `t = 0` is perihelion; `Ls` from a Newton solve of Kepler's equation.
6. CO2 frost at **fixed surface pressure**; **atmospheric-pressure feedback is
   excluded** (the condensing CO2 mass does not change the pressure or `T_frost`).
7. Frost point from an inverted `p = A exp(-B/T)` fit; a single global `T_frost`
   (pressure is spatially uniform).
8. Albedo is a two-valued function of ice presence (`A_bare` / `A_ice`); no
   continuous snow-aging or thickness dependence.
9. `mismatch_scale` scales the **whole** dust field and `G` together, so
   `scale = 0` collapses the truth onto Model 2 exactly.
10. Subsurface is a single well-mixed reservoir (one extra temperature), coupled
    only by a linear conductance `G`.
11. Model 2 is fit to the **unforced** `scale = 0` run; the same fit is reused for
    all residual/mismatch evaluations.
12. Diffusion coefficient `D` is a single constant (no latitude or season
    dependence).

## Parameters I am least sure about

Listed in `config.yaml` with `# UNCERTAIN` and ranges. The ones that most affect
the results:

* **`physics.D` (diffusion, 0.10 W/m²/K).** Controls how cold the winter pole gets
  and therefore whether a seasonal CO2 cap forms. Chosen at the low end of the
  0.10–0.60 range so that **even the dust-free base climate (`mismatch_scale = 0`)**
  forms a cap — this makes the frost/ice nonlinearity *shared* physics present at
  every scale, decoupling it from the mismatch knob. This is the single most
  consequential tuning choice; see `DIAGNOSTICS.md` for the D-sensitivity table.
* **`truth.C_s`, `truth.C_g`, `truth.G`.** Set the seasonal amplitude and the
  surface↔subsurface lag; only loosely constrained. `C_g ≫ C_s` is enforced.
* **`truth.dust_*` and `truth.storm_*`.** Dust baseline/amplitude and the storm
  statistics directly set the size of the physics mismatch; genuinely uncertain
  and the dominant driver of the residual magnitude.
* **`co2.vapor_A`, `co2.vapor_B`, `co2.latent_heat`, `co2.surface_pressure_pa`.**
  Fit constants / fixed pressure; set the frost point (≈148 K) and cap mass.
* **`planet.solar_constant`, `physics.emissivity`, albedos.** Standard but
  loosely known; set the absolute temperature level.

## Tests I loosened (and why)

* **Periodicity (test *d*)** is evaluated with **random storms disabled**
  (`storm_rate_per_year = 0`) for that test only. Storms are, by construction,
  non-periodic from year to year, so including them would make a strict
  year-over-year periodicity test meaningless. The test isolates the
  *deterministic* seasonal cycle; tolerance is `0.5 K` area-weighted RMS.
* **`scale = 0` agreement (test *e*)** uses two thresholds: the least-squares fit
  RMS must be `< 1.0 W/m²` (it is ≈0.5; bounded below by the shared frost clamp,
  not zero), and a 2-year forward run of Model 2 from
  a shared spun-up state must track the truth to `< 2 K` area-weighted RMS. The
  `2 K` forward tolerance (rather than something tighter) absorbs the small
  `C_eff` vs `C_s` discretization gap and the independent frost-clamp paths; the
  physics is otherwise identical.
* No test tolerance was loosened to hide a real disagreement.

---

## Pre-registration diagnostics

`python3 diagnostics.py` (→ `DIAGNOSTICS.md`, `diagnostics.json`) produces the
simulator-only tables used to lock **X** and **`mismatch_scale`**: D sensitivity,
mismatch residual vs scale (with absorbed-SW fraction and ratio to forcing), the
forced-response table, storm-noise across seeds, and the linearity /
extrapolation check. Headline findings: global-/polar-mean temperature responds
**nearly linearly** to `F_a` (so a global-mean OOD test is trivial), while the
**CO2 cap mass and winter-pole temperature have a threshold nonlinearity at
F_a ≈ 5–6 W/m²** — that is where a non-trivial OOD test must live.

## Next (Checkpoint B — NOT started)

Awaiting the reviewer's locked forcing range **X** and **`mismatch_scale`** (and
the X=5 vs X=4 split) before writing `PREREG.md` and building `gen_data.py`, the
`.npz` datasets (observable vs clearly-marked hidden group), the JSON manifest,
and `requirements.txt`.

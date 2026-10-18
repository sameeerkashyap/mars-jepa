# Pre-registration diagnostics, round 3 (simulator only)

`python3 diagnostics_c.py` → `diagnostics_c.json`. Truth simulator only,
`dt = 0.1 sol`, `D = 0.10` (frozen). Answers the reviewer's four checks.

---

## A. τ-vs-G ablation — the mechanism, TESTED not argued

Dust (`τ`) and subsurface coupling (`G`) scaled **independently** at fixed D=0.10,
base scale 0.10, F_a=0, storms off. Cap mass (kg/m²) by hemisphere:

| world | cap N | cap S | wmin N | wmin S |
|---|---|---|---|---|
| none (τ=0, G=0) = base | 268 | 822 | 147.74 | 147.74 |
| τ only (G=0) | 344 | **860** | 147.74 | 147.74 |
| G only (τ=0) | 0 | **291** | 147.91 | 147.74 |
| both on | 19 | 356 | 147.74 | 147.74 |

**Verdict (settles the flip-flop in my earlier summaries):**

* **G is the cap suppressor.** Turning on G alone drops the S cap from 822 → 291
  and kills the N cap (268 → 0). The subsurface buffer warms the winter pole — my
  *second* explanation is the correct one.
* **Dust alone slightly *strengthens* the cap** (822 → 860), consistent with less
  absorbed SW → cooler poles via reduced transport. My *first* summary ("no dust
  cooling, so no cap") was **wrong**, and the ablation says so.
* Both-on (356) sits between: G suppresses, dust adds a little back.

---

## B. Explicit threshold at mismatch_scale = 0.25

D=0.10, storms off:

| F_a | cap N | cap S | wmin N | wmin S |
|---|---|---|---|---|
| 0 | 0 | 270 | 148.24 | 147.74 |
| 1 | 0 | 187 | 149.24 | 147.74 |
| 2 | 0 | 113 | 150.24 | 147.74 |
| 3 | 0 | 53 | 151.22 | 147.74 |
| 4 | 0 | 10 | 152.19 | 147.74 |
| 5 | 0 | 0 | 153.15 | 148.30 |

**At scale 0.25 the regime change sits INSIDE the training range:**

* The **N pole never caps** (stronger G-buffer at higher scale) → no N threshold.
* The **S cap collapses over [0, 5]** (270 → 0), threshold ≈ **5**, i.e. at the top
  of training. So Tier 2 [7.5, 10] is **entirely cap-free** and the S winter pole
  has already unpinned within training.
* This is a **different experiment** from scale 0.10 (threshold ≈ 6 at the
  train/test boundary). Confirms the reviewer: **keep 0.25 exploratory**, describe
  it as a different truth world, not a "more mismatch" version of 0.10.

---

## C. Resolution check (n = 64 vs 128), scale 0.10, storms off

`cap_mass_max / polar-winter-min` across F_a:

| n | F4 | F5 | F6 | F7 | F8 | band cells (\|lat\|>60) |
|---|---|---|---|---|---|---|
| 64 | 45/147.7 | 5/147.7 | 0/148.5 | 0/149.5 | 0/150.6 | 8 |
| 128 | 109/147.7 | 33/147.7 | 0/147.9 | 0/149.0 | 0/150.1 | 18 |

* **Threshold location is robust:** cap vanishes between F_a=5 and 6 at **both**
  resolutions. The +5–6 W/m² threshold is **not** a grid artifact.
* **Cap mass is resolution-sensitive** (~2× larger at n=128) because the finer grid
  resolves colder poleward cells. Report cap *thickness/mass* only qualitatively.
* Recommendation: **keep n = 64** for data generation (the invariant we test — the
  threshold — is stable; n=128 is ~4× costlier). Note the mass sensitivity.

---

## D. Polar-winter signal PER HEMISPHERE, storms ON (scale 0.10)

signal = `wmin − 147.74`; noise = std over 5 seeds:

| n | F_a | N signal | N std | N S/N | S signal | S std | S S/N |
|---|---|---|---|---|---|---|---|
| 64 | 6 | 4.97 | 0.08 | **62** | 0.31 | 0.33 | 0.9 |
| 64 | 8 | 6.89 | 0.08 | **91** | 2.23 | 0.57 | 3.9 |
| 64 | 10 | 8.76 | 0.07 | **121** | 4.30 | 0.55 | 7.9 |
| 128 | 8 | 6.27 | 0.07 | 87 | 1.72 | 0.56 | 3.1 |
| 128 | 10 | 8.16 | 0.07 | 119 | 3.81 | 0.53 | 7.2 |

* **Strong hemispheric asymmetry** (as the reviewer predicted — the band is only
  ~1 cell N, ~2–3 cells S). The **N pole** has a *marginal* cap that unpins early
  and rises steeply with very low storm noise → **S/N ≈ 60–120**: a clean Tier-2
  signal. The **S pole** has a robust cap, unpins later, buried in storm noise near
  threshold (**S/N ≈ 0.9 at F_a=6**) but usable in Tier 2 (**S/N ≈ 4–8**).
* **Report hemispheres separately.** The primary Tier-2 metric is well-posed, but
  its difficulty is very different N vs S; a single pooled number would be
  misleading. Values are stable 64 → 128.

---

## Decisions for PREREG (carried forward)

**Locked-in from these checks (pending your final yes):**
* D = 0.10, n = 64, both **frozen** at Checkpoint B.
* mismatch_scale = **0.10 primary**; **0.25 exploratory only** (different truth
  world — no N cap, threshold ≈5 inside training).
* X = 5; Tier 1 = unseen schedule shapes in [0, 5]; Tier 2 = magnitudes [7.5, 10].
* Primary metric = polar-winter T_s error, band |lat| > 60°, **reported per
  hemisphere** (N is the clean signal, S the hard one). Secondary = global-mean T_s.
* Observable = {T_s, S, F_a}; hidden = {ice mass m, T_g, τ, flux terms}.

**Design choices I will bake into PREREG (your guidance adopted):**
* **Physics-loss evaluation: A (strict, training-range only) as the main H2**,
  **B (collocation on unlabeled rollouts up to F_a≈10) as a clearly-labeled
  variant.** A null result on A/Tier 2 is a likely and reportable outcome.
* **Add a linear baseline** (ridge / ARX linear state-space) as an extra phase-2
  arm, since the global-mean response is near-linear and Tier 1 may not separate
  nonlinear architectures. If the linear model matches everyone on Tier 1, that is
  itself a finding.

**Limitations to pre-register (not discoveries):**
* D was tuned (0.30 → 0.20 → 0.10) to place the threshold in range; the threshold
  swings wildly with D (appendix table: >8 / ≈6 / ≈0 / none for D=0.07/0.10/0.15/
  0.20). Tier 2 is a **designed** regime change.
* Caps under-extend vs Mars (~67–80° here vs a literature value I quoted from
  memory as ~50–55°) and the seasonal exchange (~6–17 kg/m²) is below the
  ~tens-of-kg/m² scale I cited. **These Mars reference numbers are from me, not a
  source — verify against literature before the paper.** No atmospheric transport,
  no pressure feedback; the threshold is a toy artifact.

**Housekeeping note:** the n=128 ablation runs showed heavy CPU oversubscription
(dense LU across BLAS threads). For Checkpoint-B bulk generation I'll pin BLAS
threads and/or switch the diffusion solve to a banded solver — a performance-only
change, no physics impact.

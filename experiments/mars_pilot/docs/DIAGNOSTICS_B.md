# Pre-registration diagnostics, round 2 (simulator only)

Produced by `python3 diagnostics_b.py` (→ `diagnostics_b.json`). Truth simulator
only; `dt = 0.1 sol`. Answers the three items the reviewer asked for before
locking X and `mismatch_scale`.

---

## 1. Threshold location vs D — and an important physical finding

### The cap is controlled by TWO competing parts of the mismatch

Investigating why D=0.15 caps at `scale=0` but not at `scale=0.10` (which has
*more* dust, so should be colder) revealed a real mechanism, not a bug:

* **Dust** (`∝ mismatch_scale`) cools the **sunlit** hemisphere. In polar winter
  there is *no sunlight*, so dust does **not** cool the winter pole directly.
* **Subsurface coupling G** (`∝ mismatch_scale`) is a heat **buffer**: the deep
  reservoir `T_g` holds summer heat (`C_g ≫ C_s`) and bleeds it back to the
  surface in winter, **warming the winter pole and suppressing the cap**.

Net effect at the winter pole: **larger `mismatch_scale` → warmer winter pole →
smaller cap** (the G-buffer beats the dust there). Confirmed (storms off, F_a=0,
spin-independent):

| D | cap @ scale 0 | cap @ scale 0.10 | winter-min N @ scale 0.10 |
|---|---|---|---|
| 0.10 | 822 | 356 | 147.74 (pinned) |
| 0.13 | 513 | 66 | 151.3 (N uncapped) |
| 0.15 | 286 | **0** | 153.9 (uncapped) |

So the cap, and therefore the whole threshold experiment, depends on **both** D
and `mismatch_scale`.

### Threshold vs D at the operating scale (mismatch_scale = 0.10, storms off)

Threshold = lowest `F_a` at which the winter cap no longer forms. Entries are
`cap_mass_max (kg/m²) / polar-winter-min T_s (K)`:

| D | F=0 | F=2 | F=4 | F=5 | F=6 | F=7 | F=8 | threshold |
|---|---|---|---|---|---|---|---|---|
| 0.07 | 703/147.7 | 528/147.7 | 342/147.7 | 255/147.7 | 176/147.7 | 107/147.7 | 49/147.7 | **> 8** |
| **0.10** | 356/147.7 | 178/147.7 | 45/147.7 | 5/147.7 | 0/148.5 | 0/149.5 | 0/150.6 | **≈ 6** |
| 0.15 | 0/148.4 | 0/150.4 | 0/152.4 | 0/153.4 | 0/154.3 | 0/155.3 | 0/156.2 | **≈ 0** |
| 0.20 | 0/153.2 | 0/155.1 | 0/156.9 | 0/157.8 | 0/158.7 | 0/159.5 | 0/160.4 | **none** |

**The threshold swings wildly across plausible D** (the reviewer's worry,
confirmed): `> 8` at D=0.07, `≈ 6` at D=0.10, `≈ 0` at D=0.15, gone at D=0.20.

* **D=0.07** is *too robust*: the cap survives past F_a=8, so the polar-winter
  metric would be **flat (no signal) across the whole forcing range** — useless.
* **D≥0.15** has **no cap** at the operating scale — no frost nonlinearity at all.
* **D=0.10** is the only value that places the threshold **near the top of the
  forcing range (≈6)**, i.e. just below Tier 2 — which is what makes the OOD test
  meaningful.

**This is tuning, and the paper must say so.** D=0.10 is chosen to put the frost
threshold at a useful location; the threshold is not a robust prediction. I will
**freeze D=0.10 at Checkpoint B and not touch it again.**

### Cap extent & seasonal cycle at D=0.10 vs Mars

| scale | N cap reaches | S cap reaches | peak cap | global-mean ice cycle |
|---|---|---|---|---|
| 0.0 | 79.9° (top cell only) | 67.2° | 822 kg/m² | 0 → 16.6 kg/m² |
| 0.10 | 79.9° (top cell only) | 72.4° | 359 kg/m² | 0 → 5.6 kg/m² |

**Honest comparison to Mars** (to be stated as a limitation):

* **Extent:** real Mars seasonal CO2 caps reach ~**50–55°** latitude at maximum.
  Our caps are **confined too close to the poles** (~67–80°) — the toy has a
  single global frost point and no atmospheric CO2/heat transport, so cold does
  not propagate equatorward. The N cap is especially thin (one grid cell).
* **Seasonal cycle:** on Mars ~25–30% of the ~610 Pa CO2 atmosphere condenses
  seasonally (~50 kg/m² column-equivalent globally). Our global-mean exchange is
  **~6–17 kg/m²** — right order of magnitude but low, again because the caps are
  under-extended.
* **Conclusion:** qualitatively Mars-like (seasonal polar CO2 caps, N/S asymmetry
  from eccentricity), **not quantitatively calibrated**. The `+5–6 W/m²` threshold
  is a property of this toy (no transport, no pressure feedback) and must not be
  read as a real Mars number.

---

## 2. Polar-band storm noise (primary metric)

Primary metric = **polar-winter minimum T_s** in the band **|lat| > 60°**
(uses **T_s only**). `mismatch_scale = 0.10`, storms **ON**, 5 seeds. Signal =
rise above the frost point (147.74 K):

| F_a | mean winter-min | std (noise) | range | signal above frost | S/N |
|---|---|---|---|---|---|
| 5.0 | 147.74 | 0.000 | 0.000 | 0.00 | — (pinned) |
| 6.0 | 148.05 | 0.331 | 0.717 | 0.31 | **0.9** |
| 8.0 | 149.97 | 0.572 | 1.503 | 2.23 | **3.9** |
| 10.0 | 152.04 | 0.545 | 1.432 | 4.30 | **7.9** |

Reading:

* For `F_a ≤ 5` the winter pole is **exactly frost-pinned** → zero signal **and**
  zero storm noise. So within **Tier 1 [0, 5] the polar-winter metric is flat**;
  Tier 1 discrimination must come from global-mean / non-polar-winter T_s.
* Right at the threshold (`F_a = 6`) the signal (0.31 K) is **below** storm noise
  (S/N ≈ 0.9) — the cap flickers on/off with storms.
* In **Tier 2 [7.5, 10]** the signal clears the noise comfortably:
  **S/N ≈ 4 (F_a=8) to 8 (F_a=10)**; at F_a=7.5 it is ~2–3.

So the polar-winter-T_s metric is specifically a **Tier-2 discriminator**: it
measures whether a model extrapolates **across the frost regime change** it never
saw in training. That is exactly the H1/H2 question.

---

## 3. Observable / hidden split and the metric — confirmation + one conflict

* **Primary metric uses T_s only.** Confirmed: `polar_winter_min` reads
  `r["T_s"]` exclusively; no ice mass, no `T_g`, no `τ`.
* **Ice mass hidden — CONFLICT with the original brief, flagging for your call.**
  The Checkpoint-B brief says: *"Save .npz with T_s (observable), S, F_a, ice
  mass, and diagnostics (T_g, τ, flux terms) in a separate hidden group"* — i.e.
  it lists **ice mass as observable**. You now want **ice mass hidden**. I agree
  hiding it is the better design (the model must infer frost behavior from T_s
  alone, otherwise the cap is half-observed), and I will implement it that way —
  but please **confirm you want to override the brief**, since this changes the
  observable set.

Proposed Checkpoint-B grouping (pending your confirmation):

* **observable (model input):** `T_s`, `S`, `F_a`
* **hidden (diagnostics, clearly marked not-for-input):** `m` (ice mass), `T_g`,
  `τ`, truth flux terms

---

## Updated recommendation

* **mismatch_scale = 0.10 primary**, **0.25 secondary** (physics on/off for GRU +
  RSSM at one data fraction) — unchanged.
* **X = 5; Tier 1 = unseen schedule shapes in [0, 5]; Tier 2 = magnitudes
  [7.5, 10]** — agreed. Report tiers separately.
* **Primary metric:** intervention-response error on **polar-winter T_s**,
  band **|lat| > 60°** (a Tier-2 discriminator). **Secondary:** global-mean T_s
  (the within-regime, ~linear metric that carries Tier 1).
* **Freeze D = 0.10** at Checkpoint B.
* **Paper limitation (pre-registered):** caps under-extend vs Mars (~67–80° vs
  ~50–55°) and the +5–6 W/m² threshold is a toy artifact (no transport, no
  pressure feedback).

Confirm (a) override the brief to **hide ice mass**, (b) polar band at **|lat| >
60°** (the pinned region is poleward of ~67°, so 60° cleanly contains it), and I'll
write `PREREG.md` and start Checkpoint B.

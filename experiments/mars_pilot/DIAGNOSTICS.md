# Pre-registration diagnostics (simulator only)

Produced by `python3 diagnostics.py` (writes `diagnostics.json`). All numbers are
from the **truth simulator only** — no models trained. These are to lock **X**
and **mismatch_scale** with the reviewer before Checkpoint B.

> **Timestep note:** these steady-state diagnostics use `dt = 0.1 sol` for speed;
> annual means reproduce the `dt = 0.05 sol` production value to `< ~0.1 K`.
> Seeds fixed; `seed = 0` unless a seed sweep is stated.

---

## 0. The design change that these diagnostics forced: D 0.20 → 0.10

The reviewer flagged that the frost/cap nonlinearity and the physics mismatch were
coupled through `mismatch_scale` (at low scale there was no dust-driven cooling, so
**no cap formed at all**). Fix: lower the diffusion coefficient so the **dust-free
base climate itself** (`mismatch_scale = 0`, i.e. the Model-2 climate) is cold
enough at the winter pole to form a seasonal CO2 cap. Then the frost/ice
nonlinearity is **shared physics present at every scale**, independent of the
mismatch knob.

### 1. D sensitivity (base = dust-free `scale=0` climate; storms off)

| D | base global-mean | base winter-pole min | base cap mass | ΔT_glob (F_a=10, s=0.25) | ΔT_polar |
|---|---|---|---|---|---|
| 0.10 | 209.4 K | **147.7 K (pinned)** | **822 kg/m²** | 5.43 K | 6.97 K |
| 0.15 | 209.9 K | **147.7 K (pinned)** | 286 kg/m² | 5.37 K | 6.50 K |
| 0.20 | 210.3 K | 148.6 K (no cap) | 0 kg/m² | 5.33 K | 6.19 K |

D only weakly affects the forced response, but it decides whether the base climate
caps at all. **D = 0.10** (global mean ≈209 K, winter pole pinned, robust cap) is
now the config default. D = 0.15 is a viable alternative with a smaller cap.

---

## 2. Mismatch residual vs `mismatch_scale`

Model 2 fit **once** on the `scale=0` unforced run (`C_eff≈4.88e5, D≈0.0998,
ε≈0.950`, fit RMS **0.69 W/m²**). Residual = area-weighted RMS over frost-free
cells of the Model-2 residual evaluated on the truth trajectory. "forced" uses a
constant `F_a = 10 W/m²`. "% abs" = residual as a fraction of mean absorbed
shortwave. Last column = residual ÷ 10 W/m².

| scale | unforced RMS | % absorbed | forced RMS | % absorbed | forced RMS / F_a(10) |
|---|---|---|---|---|---|
| 0.00 | 0.69 | 0.6% | 0.05 | 0.0% | 0.01 |
| 0.05 | 4.77 | 4.4% | 4.76 | 4.4% | 0.48 |
| **0.10** | **9.03** | **8.6%** | **8.97** | **8.5%** | **0.90** |
| 0.25 | 19.69 | 20.4% | 19.53 | 20.2% | 1.95 |
| 0.50 | 33.48 | 39.6% | 33.16 | 39.2% | 3.32 |
| 1.00 | 54.56 | 83.9% | 53.80 | 82.5% | 5.38 |

Interpretation vs the reviewer's criterion ("residual in the same ballpark as the
forcing, not far above"):

* **scale = 0.00** is *not* a perfect oracle any more — the shared frost clamp
  leaves ~0.69 W/m² the linear EBM can't fit. (Good; addresses the oracle worry.)
* **scale = 0.10** → 9 W/m², ≈ **0.9× a 10 W/m² forcing** and **~1.8× a 5 W/m²
  forcing**; 8.6% of absorbed SW. This is "same ballpark, not far above."
* **scale = 0.25** → 20 W/m² ≈ **2–4× the forcing**, 20% of absorbed SW — at or
  above the upper edge of "same ballpark."
* scale 0.50 / 1.00: dust dominates (40–84% of absorbed SW). Avoid.

**Recommendation:** `mismatch_scale = 0.10` as the **primary** level;
`0.25` as a **secondary** stronger-mismatch run. Avoid 0 and 1.0 (per reviewer).

---

## 3. Forced-response table (storms OFF, dT relative to F_a=0)

**mismatch_scale = 0.10**

| F_a | global-mean | ΔT_glob | polar | ΔT_polar | cap mass |
|---|---|---|---|---|---|
| 0.0 | 207.4 | 0.00 | 184.0 | 0.00 | 356 |
| 2.0 | 208.5 | 1.08 | 185.4 | 1.41 | 178 |
| 5.0 | 210.1 | 2.66 | 187.5 | 3.49 | 5 |
| 7.5 | 211.4 | 3.96 | 189.2 | 5.18 | 0 |
| 10.0 | 212.7 | 5.23 | 190.8 | 6.84 | 0 |
| 15.0 | 215.1 | 7.70 | 194.0 | 10.03 | 0 |
| 20.0 | 217.5 | 10.08 | 197.1 | 13.08 | 0 |

**mismatch_scale = 0.25**

| F_a | global-mean | ΔT_glob | polar | ΔT_polar | cap mass |
|---|---|---|---|---|---|
| 0.0 | 204.0 | 0.00 | 181.9 | 0.00 | 270 |
| 2.0 | 205.1 | 1.12 | 183.3 | 1.45 | 113 |
| 5.0 | 206.8 | 2.77 | 185.4 | 3.57 | 0 |
| 7.5 | 208.1 | 4.12 | 187.2 | 5.29 | 0 |
| 10.0 | 209.4 | 5.43 | 188.8 | 6.97 | 0 |
| 15.0 | 212.0 | 7.99 | 192.1 | 10.22 | 0 |
| 20.0 | 214.4 | 10.46 | 195.2 | 13.32 | 0 |

Global sensitivity ≈ **0.53 K per W/m²**, polar ≈ **0.70 K per W/m²**.

### Fine scan locating the frost/cap threshold (scale = 0.10)

| F_a | cap mass | winter-pole min | ΔT_glob | ΔT_polar |
|---|---|---|---|---|
| 0 | 356 | 147.74 | 0.00 | 0.00 |
| 1 | 262 | 147.74 | 0.54 | 0.71 |
| 2 | 178 | 147.74 | 1.08 | 1.41 |
| 3 | 104 | 147.74 | 1.61 | 2.11 |
| 4 | 45 | 147.74 | 2.14 | 2.80 |
| **5** | **5** | **147.74** | 2.66 | 3.49 |
| **6** | **0** | **148.49** | 3.18 | 4.17 |
| 8 | 0 | 150.59 | 4.21 | 5.52 |
| 10 | 0 | 152.64 | 5.23 | 6.84 |

**The nonlinearity has a clear threshold at F_a ≈ 5–6 W/m²:** below it the winter
pole is frost-pinned at 147.74 K and a cap exists; above it the cap is gone and the
winter-pole minimum rises. The cap mass itself is a strongly nonlinear observable
(hits zero and stays there).

---

## 4. Storm noise (storms ON, unforced, 5 seeds)

| scale | global-mean | g-mean std | g-mean range | polar std | polar range |
|---|---|---|---|---|---|
| 0.10 | 206.6 K | 0.32 K | 0.86 K | 0.10 K | 0.28 K |
| 0.25 | 202.0 K | 0.78 K | 2.12 K | 0.27 K | 0.75 K |

Signal-vs-noise (global mean, scale 0.10): forcing signal is 1.08 K at F_a=2 and
2.66 K at F_a=5, vs 0.32 K storm std → **S/N ≈ 3.4 (F_a=2) to 8.3 (F_a=5)**,
comfortably above the reviewer's "≈2×" bar across the whole range.

---

## 5. Linearity check (THE important one)

Fit a straight line to the forced response on `[0, X]` and predict `[1.5X, 2X]`.
`X = 5` → test range `[7.5, 10]`.

| scale | target | slope (K/(W/m²)) | quad coef | F_a=7.5 actual/linpred/err | F_a=10 actual/linpred/err |
|---|---|---|---|---|---|
| 0.10 | global | 0.532 | −0.0018 | 3.96 / 3.99 / **0.04 K** | 5.23 / 5.32 / **0.10 K** |
| 0.10 | polar | 0.697 | −0.0028 | 5.18 / 5.24 / **0.05 K** | 6.84 / 6.98 / **0.14 K** |
| 0.25 | global | 0.554 | −0.0021 | 4.12 / 4.16 / 0.04 K | 5.43 / 5.54 / 0.11 K |
| 0.25 | polar | 0.713 | −0.0038 | 5.29 / 5.36 / 0.06 K | 6.97 / 7.14 / 0.17 K |

**Finding — and a caveat the reviewer should weigh in on.** The **global-mean and
polar-mean temperature responses are essentially linear**: a line fit on `[0,5]`
predicts `[7.5,10]` to within 0.04–0.17 K. By those targets alone the OOD test is
**trivial** — any architecture would extrapolate well.

The reason is exactly the regime change the reviewer warned about: with `X = 5` the
CO2 cap **collapses inside the training range** (gone by F_a≈5–6), so both the high
end of training and the whole test range sit in the cap-free, linear regime.

**Where the nonlinearity actually lives:** in the **cap-mass** and
**winter-pole / polar-temperature** observables, whose threshold sits at F_a ≈ 5–6
(section 3 fine scan). Those are *observable* quantities (ice mass and T_s are both
in the model-input group), so an OOD test on them is **not** trivial.

---

## Recommendations to lock X and mismatch_scale

1. **mismatch_scale = 0.10 (primary), 0.25 (secondary).** At 0.10 the physics
   residual (9 W/m², 8.6% of absorbed SW, ≈0.9× a 10 W/m² forcing) is the same
   size as the forcing — incomplete physics, not dominant. 0.25 (20 W/m²) is the
   upper edge; useful as a stronger-mismatch comparison for the best architecture.

2. **X: the global-mean OOD target is trivially linear, so the split and the
   evaluation must be chosen around the frost/cap threshold (F_a ≈ 5–6).** Two
   defensible options:
   * **X = 5, test [7.5, 10]** (reviewer's start): the threshold sits at the
     train/test boundary. Training sees caps shrinking + winter pinned; the test
     region is cap-free with a rising winter pole it has **not** seen. Non-trivial
     for cap-mass / polar-temperature targets, trivial for global mean.
   * **X ≈ 3–4, test [1.5X, 2X] = [4.5–8]**: puts the threshold *between* train and
     test as pure extrapolation across the kink — a sharper OOD test. Signal is
     still ≥3× storm noise.

   My suggestion: **X = 5** as agreed, **but pre-register that the primary OOD
   metric is evaluated on T_s including the polar cells and on CO2 cap mass**, not
   only global-mean T_s — otherwise the test can't separate architectures. If you
   prefer a harder split, drop to **X = 4**.

3. **Flag (regime change):** the cap collapse at F_a ≈ 5–6 is intrinsic to the
   physics and is the nonlinearity we want. The choice of X decides whether the
   train/test boundary straddles it. This must be fixed in `PREREG.md` **before**
   any training, and `mismatch_scale` must be fixed from these numbers (not chosen
   after seeing which level makes physics "win").

Send back your locked **X** and **mismatch_scale** (and whether you want the X=5 or
X=4 split) and I'll write `PREREG.md` and start Checkpoint B.

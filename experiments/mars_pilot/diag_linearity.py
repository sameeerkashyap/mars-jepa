"""
diag_linearity.py -- how well does a LINEAR-in-A response, fit on the training
magnitudes, extrapolate to the Tier-2 magnitudes?  Simulator only (no dataset
or test-split file is read; seeds 7000-7004 are outside every dataset range).

For each storm seed s and constant forcing A:

    spin   : run_truth(F_a=0, seed=s, SPIN_YEARS)            -> initial state
    control: run_truth(F_a=0, seed=s, REC_YEARS, init)       (shared across A)
    forced : run_truth(F_a=A, seed=s, REC_YEARS, init)
             A in TRAIN_MAGS = {1,2,3,4,5}  and  TEST_MAGS = {7.5,8,9,10}
    response R(t, x) = T_s(forced) - T_s(control)

Per hemisphere, over polar-winter steps (metric_defs: band |lat|>60, band-mean
S <= WINTER_S_FRAC * its record max), the mean response over the cell-steps of
three groups:
    band   : all band cells
    pinned : band cell-steps whose CONTROL T_s <= T_frost + PIN_TOL_K
    rest   : the other band cell-steps
Per seed and group: slope k = sum(A r) / sum(A^2) over the train A (line
through the origin), predicted r_hat(A) = k A at each test A.  Reported per
test A: mean over seeds of true r, predicted r_hat, error r_hat - r, relative
error (r_hat - r) / r, and how many seeds contributed.  A seed with no cell-step
in a group is listed under "seeds_missing", never dropped silently.

Frozen design: scale 0.10, D=0.10, n=64, dt=0.05; record_years=3 (= dataset).

    python3 diag_linearity.py --workers 5          # writes diag_linearity.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from metric_defs import BAND_DEG, WINTER_S_FRAC, polar_bands, winter_steps

TRAIN_MAGS = [1.0, 2.0, 3.0, 4.0, 5.0]
TEST_MAGS = [7.5, 8.0, 9.0, 10.0]
DIAG_SEEDS = [7000, 7001, 7002, 7003, 7004]
REC_YEARS = 3
PIN_TOL_K = 0.5
GROUPS = ("band", "pinned", "rest")


def _seed_job(args):
    """All runs for one storm seed. Top-level for the spawn start method."""
    seed, train_mags, test_mags, rec_years, winter_frac, pin_tol = args
    from common import load_config, frost_point
    from sim_truth import run_truth
    from gen_data import SCALE, SPIN_YEARS

    cfg = load_config()
    assert cfg["time"]["dt_sol"] == 0.05 and cfg["physics"]["D"] == 0.10 \
        and cfg["grid"]["n"] == 64, "frozen design parameters changed!"
    T_frost = frost_point(cfg["co2"]["surface_pressure_pa"], cfg["co2"])

    spin = run_truth(cfg, F_a=0.0, seed=seed, years=SPIN_YEARS,
                     mismatch_scale=SCALE)
    init = dict(T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                m_init=spin["m_final"])
    ctrl = run_truth(cfg, F_a=0.0, seed=seed, years=rec_years,
                     mismatch_scale=SCALE, **init)
    bands = polar_bands(ctrl["lat"])
    S_ctrl = np.asarray(ctrl["S"])
    Tc = np.asarray(ctrl["T_s"])
    winter = {h: winter_steps(S_ctrl, sel, frac=winter_frac)
              for h, sel in bands.items()}
    # (time, ncell) group masks, fixed by the control (same for every A)
    masks = {}
    for h, sel in bands.items():
        b = winter[h][:, None] & sel[None, :]
        pin = b & (Tc <= T_frost + pin_tol)
        masks[h] = {"band": b, "pinned": pin, "rest": b & ~pin}

    out = {"seed": seed, "T_frost": float(T_frost),
           "n_band_cells": {h: int(s.sum()) for h, s in bands.items()},
           "n_winter_steps": {h: int(w.sum()) for h, w in winter.items()},
           "n_cell_steps": {h: {g: int(m.sum()) for g, m in masks[h].items()}
                            for h in masks},
           "response_K": {}}
    for A in list(train_mags) + list(test_mags):
        forced = run_truth(cfg, F_a=float(A), seed=seed, years=rec_years,
                           mismatch_scale=SCALE, **init)
        Tf = np.asarray(forced["T_s"])
        n = min(len(Tf), len(Tc))
        R = Tf[:n] - Tc[:n]
        out["response_K"][str(A)] = {
            h: {g: (float(R[m[:n]].mean()) if m[:n].any() else None)
                for g, m in masks[h].items()}
            for h in masks}
    return out


def fit_seed(per, h, g, train_mags, test_mags):
    """Through-origin fit on train A for one seed; None if no usable data."""
    tr = [(A, per["response_K"][str(A)][h][g]) for A in train_mags]
    tr = [(A, r) for A, r in tr if r is not None]
    te = {A: per["response_K"][str(A)][h][g] for A in test_mags}
    if not tr or any(v is None for v in te.values()):
        return None
    a = np.array([A for A, _ in tr])
    r = np.array([v for _, v in tr])
    k = float((a * r).sum() / (a * a).sum())
    return {"slope_K_per_Wm2": k, "n_train_points": len(tr),
            "train_resid_rms_K": float(np.sqrt(np.mean((r - k * a) ** 2))),
            "true_K": {str(A): te[A] for A in test_mags},
            "pred_K": {str(A): k * A for A in test_mags}}


def summarize(per_seed, train_mags, test_mags):
    table = {}
    for h in ("N", "S"):
        table[h] = {}
        for g in GROUPS:
            fits = {p["seed"]: fit_seed(p, h, g, train_mags, test_mags)
                    for p in per_seed}
            ok = {s: f for s, f in fits.items() if f is not None}
            row = {"n_seeds": len(ok), "n_seeds_total": len(per_seed),
                   "seeds_missing": sorted(s for s, f in fits.items() if f is None),
                   "slope_mean": (float(np.mean([f["slope_K_per_Wm2"]
                                                 for f in ok.values()]))
                                  if ok else None),
                   "by_A": {}}
            for A in test_mags:
                if not ok:
                    row["by_A"][str(A)] = None
                    continue
                t = np.array([f["true_K"][str(A)] for f in ok.values()])
                p = np.array([f["pred_K"][str(A)] for f in ok.values()])
                e = p - t
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = e / t
                row["by_A"][str(A)] = {
                    "true_K": float(t.mean()), "pred_K": float(p.mean()),
                    "err_K": float(e.mean()),
                    "err_seed_std_K": float(e.std(ddof=1)) if len(e) > 1 else float("nan"),
                    "rel_err": float(np.mean(rel)),
                    "rel_err_min": float(np.min(rel)),
                    "rel_err_max": float(np.max(rel))}
            table[h][g] = row
    return table


def print_table(table, test_mags):
    print("Linear (through-origin, fit on train A) extrapolation of the "
          "polar-winter response, |lat|>60, K")
    hdr = (f"{'hem':>3} {'group':>6} {'seeds':>5} {'A':>5} | {'true':>6} "
           f"{'pred':>6} {'err':>6} {'rel':>6}")
    print(hdr); print("-" * len(hdr))
    for h, groups in table.items():
        for g, row in groups.items():
            seeds = f"{row['n_seeds']}/{row['n_seeds_total']}"
            for A in test_mags:
                v = row["by_A"][str(A)]
                vals = ("   n/a" * 4 if v is None else
                        f"{v['true_K']:>6.2f} {v['pred_K']:>6.2f} "
                        f"{v['err_K']:>6.2f} {v['rel_err']:>6.3f}")
                print(f"{h:>3} {g:>6} {seeds:>5} {float(A):>5.1f} | {vals}")
            if row["seeds_missing"]:
                print(f"    {h}/{g}: no cell-steps for seeds {row['seeds_missing']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    ap.add_argument("--train-mags", type=float, nargs="+", default=TRAIN_MAGS)
    ap.add_argument("--test-mags", type=float, nargs="+", default=TEST_MAGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DIAG_SEEDS)
    ap.add_argument("--rec-years", type=float, default=REC_YEARS)
    ap.add_argument("--winter-s-frac", type=float, default=WINTER_S_FRAC)
    ap.add_argument("--pin-tol", type=float, default=PIN_TOL_K)
    ap.add_argument("--out", default="diag_linearity.json")
    args = ap.parse_args(argv)

    jobs = [(s, args.train_mags, args.test_mags, args.rec_years,
             args.winter_s_frac, args.pin_tol) for s in args.seeds]
    if args.workers <= 1:
        per_seed = [_seed_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            per_seed = list(ex.map(_seed_job, jobs))
    table = summarize(per_seed, args.train_mags, args.test_mags)
    out = {"definition": __doc__, "train_mags": args.train_mags,
           "test_mags": args.test_mags, "seeds": args.seeds,
           "rec_years": args.rec_years, "band_deg": BAND_DEG,
           "winter_s_frac": args.winter_s_frac, "pin_tol_K": args.pin_tol,
           "table": table, "per_seed": per_seed}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print_table(table, args.test_mags)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

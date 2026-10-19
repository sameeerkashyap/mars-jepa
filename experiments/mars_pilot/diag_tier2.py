"""
diag_tier2.py -- RESPONSE-based Tier-2 diagnostic (supersedes the level-based
diagnostics_tier2.py, which measured min T_s above frost, not the response).

For each storm seed s in DIAG_SEEDS (7000-7004; outside every dataset family
range, so no dataset or test-split file is read) and each constant forcing A in
MAGS = {7.5, 8, 9, 10} W/m^2:

    spin   : run_truth(F_a=0, seed=s, SPIN_YEARS)            -> initial state
    control: run_truth(F_a=0, seed=s, REC_YEARS, init)       (shared across A)
    forced : run_truth(F_a=A, seed=s, REC_YEARS, init)       constant forcing
    response R(t, x) = T_s(forced) - T_s(control)            same seed + init

Per hemisphere (polar band |lat| > 60 deg; N = lat > 60, S = lat < -60):
  * polar-winter times = steps where the band-mean insolation S is
    <= WINTER_S_FRAC * (its max over the record)   [observable-only definition]
  * per seed: winter-mean band response r_s (K), and the fraction of winter
    band cells that are frost-pinned in the control but unpinned when forced
    (T_s > T_frost + PIN_TOL_K)
  * table: signal = mean_s r_s ; noise = std_s r_s ; S/N = signal / noise ;
    peak = mean_s max_t(band-mean R over winter) ; unpinned fraction (mean_s).

Frozen design: scale 0.10, D=0.10, n=64, dt=0.05; record_years=3 (= dataset).

Run it yourself (slow: 5 seeds x (1 spin + 1 control + 4 forced) runs of
3 years each):

    python3 diag_tier2.py --workers 5          # writes diag_tier2.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from metric_defs import BAND_DEG, WINTER_S_FRAC, polar_bands, winter_steps

MAGS = [7.5, 8.0, 9.0, 10.0]
DIAG_SEEDS = [7000, 7001, 7002, 7003, 7004]
REC_YEARS = 3
PIN_TOL_K = 0.5


def _seed_job(args):
    """All runs for one storm seed. Top-level for the spawn start method."""
    seed, mags, rec_years, winter_frac, pin_tol = args
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
    winter = {h: winter_steps(S_ctrl, sel, frac=winter_frac)
              for h, sel in bands.items()}

    out = {"seed": seed, "T_frost": float(T_frost),
           "n_band_cells": {h: int(sel.sum()) for h, sel in bands.items()},
           "n_winter_steps": {h: int(w.sum()) for h, w in winter.items()},
           "by_A": {}}
    Tc = np.asarray(ctrl["T_s"])
    for A in mags:
        forced = run_truth(cfg, F_a=float(A), seed=seed, years=rec_years,
                           mismatch_scale=SCALE, **init)
        Tf = np.asarray(forced["T_s"])
        n = min(len(Tf), len(Tc))
        R = Tf[:n] - Tc[:n]
        res = {}
        for h, sel in bands.items():
            w = winter[h][:n]
            if not w.any():
                res[h] = None
                continue
            band_R = R[w][:, sel].mean(axis=1)                 # (n_winter,)
            pinned_c = Tc[:n][w][:, sel] <= T_frost + pin_tol
            unpinned_f = Tf[:n][w][:, sel] > T_frost + pin_tol
            frac = (float((pinned_c & unpinned_f).sum() / pinned_c.sum())
                    if pinned_c.any() else float("nan"))
            res[h] = {"winter_mean_response_K": float(band_R.mean()),
                      "winter_peak_response_K": float(band_R.max()),
                      "frac_ctrl_pinned_unpinned_forced": frac}
        out["by_A"][str(A)] = res
    return out


def summarize(per_seed, mags):
    table = {}
    for A in mags:
        row = {}
        for h in ("N", "S"):
            vals = [p["by_A"][str(A)][h] for p in per_seed
                    if p["by_A"][str(A)][h] is not None]
            if not vals:
                row[h] = None
                continue
            r = np.array([v["winter_mean_response_K"] for v in vals])
            fr = np.array([v["frac_ctrl_pinned_unpinned_forced"] for v in vals])
            fin = np.isfinite(fr)          # NaN = seed had no control-pinned cell
            sig, sd = float(r.mean()), float(r.std(ddof=1)) if len(r) > 1 else float("nan")
            row[h] = {
                "signal_K": sig, "seed_std_K": sd,
                "SN": (sig / sd) if (np.isfinite(sd) and sd > 1e-9) else float("inf"),
                "peak_K": float(np.mean([v["winter_peak_response_K"] for v in vals])),
                "frac_unpinned": float(fr[fin].mean()) if fin.any() else float("nan"),
                "frac_unpinned_min": float(fr[fin].min()) if fin.any() else float("nan"),
                "frac_unpinned_n_seeds": int(fin.sum()),
                "n_seeds_no_ctrl_pinned": int((~fin).sum()),
                "n_seeds": len(vals)}
        table[str(A)] = row
    return table


def print_table(table):
    hdr = (f"{'A':>5} | {'N sig':>6} {'N std':>6} {'N S/N':>6} {'N pk':>6} {'N unp':>5}"
           f" | {'S sig':>6} {'S std':>6} {'S S/N':>6} {'S pk':>6} {'S unp':>5}")
    print("Tier-2 polar-winter RESPONSE (forced - same-seed control), constant "
          "forcing, |lat|>60, K")
    print(hdr); print("-" * len(hdr))
    for A, row in table.items():
        cells = []
        for h in ("N", "S"):
            v = row[h]
            cells.append("   n/a" * 4 + "   n/a" if v is None else
                         f"{v['signal_K']:>6.2f} {v['seed_std_K']:>6.3f} "
                         f"{v['SN']:>6.1f} {v['peak_K']:>6.2f} {v['frac_unpinned']:>5.2f}")
        print(f"{float(A):>5.1f} | {cells[0]} | {cells[1]}")
    print("unp = mean over seeds WITH a control-pinned winter band cell; seeds "
          "contributing (of total) / min:")
    for A, row in table.items():
        parts = []
        for h in ("N", "S"):
            v = row[h]
            parts.append(f"{h} n/a" if v is None else
                         f"{h} {v['frac_unpinned_n_seeds']}/{v['n_seeds']} "
                         f"min {v['frac_unpinned_min']:.2f}")
        print(f"{float(A):>5.1f} | " + " | ".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    ap.add_argument("--mags", type=float, nargs="+", default=MAGS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DIAG_SEEDS)
    ap.add_argument("--rec-years", type=float, default=REC_YEARS)
    ap.add_argument("--winter-s-frac", type=float, default=WINTER_S_FRAC)
    ap.add_argument("--pin-tol", type=float, default=PIN_TOL_K)
    ap.add_argument("--out", default="diag_tier2.json")
    args = ap.parse_args(argv)

    jobs = [(s, args.mags, args.rec_years, args.winter_s_frac, args.pin_tol)
            for s in args.seeds]
    if args.workers <= 1:
        per_seed = [_seed_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            per_seed = list(ex.map(_seed_job, jobs))
    table = summarize(per_seed, args.mags)
    out = {"definition": __doc__, "mags": args.mags, "seeds": args.seeds,
           "rec_years": args.rec_years, "band_deg": BAND_DEG,
           "winter_s_frac": args.winter_s_frac, "pin_tol_K": args.pin_tol,
           "table": table, "per_seed": per_seed}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print_table(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

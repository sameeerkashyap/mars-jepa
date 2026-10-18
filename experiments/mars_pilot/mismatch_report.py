"""
Mismatch report: quantify how incomplete the simplified physics (Model 2) is.

Procedure:
  1. Fit Model 2 (C_eff, D, eps) ONCE to the mismatch_scale = 0 unforced truth
     run (where the truth reduces to Model 2).
  2. Evaluate the Model-2 residual on truth trajectories at
     mismatch_scale in {0, 0.25, 0.5, 1.0}, both unforced (F_a = 0) and
     forced (F_a = FORCED).  Report the area-weighted RMS residual (W/m^2)
     over frost-free cells.

The residual grows with mismatch_scale -- this is the "how incomplete is the
physics" signal that the ML phase will try to learn.
"""

import json

import numpy as np

from common import Grid, load_config
from sim_truth import run_truth
from sim_ebm import fit_ebm, residual

FORCED = 10.0          # W/m^2 constant forcing used for the "forced" column
SCALES = [0.0, 0.25, 0.5, 1.0]
YEARS = 3


def weighted_rms(res, w):
    """Area-weighted RMS of a (time, cell) residual, ignoring NaN (masked)."""
    sq = res**2
    vals = []
    for row in sq:
        good = ~np.isnan(row)
        if good.any():
            vals.append(np.sum(w[good] * row[good]) / np.sum(w[good]))
    return float(np.sqrt(np.mean(vals)))


def main():
    cfg = load_config()
    grid = Grid(cfg["grid"]["n"])
    sol_s = cfg["planet"]["sol_seconds"]
    dt_out = cfg["time"]["output_every_sol"] * sol_s

    # --- fit once on the scale-0 unforced run ---
    base = run_truth(cfg, F_a=0.0, seed=0, years=YEARS, mismatch_scale=0.0)
    params, fit_rms = fit_ebm(cfg, base)

    report = {
        "fit": {"C_eff": params["C_eff"], "D": params["D"],
                "eps": params["eps"], "fit_rms_Wm2": fit_rms},
        "forced_Wm2": FORCED,
        "residual_rms_Wm2": {},
    }

    print(f"Fitted Model 2: C_eff={params['C_eff']:.4g}  "
          f"D={params['D']:.4g}  eps={params['eps']:.4g}  "
          f"(fit RMS={fit_rms:.4g} W/m^2)\n")
    print(f"{'mismatch_scale':>15} {'unforced RMS':>14} "
          f"{'forced RMS':>14}  (W/m^2)")

    for s in SCALES:
        row = {}
        for label, Fa in [("unforced", 0.0), ("forced", FORCED)]:
            tr = run_truth(cfg, F_a=Fa, seed=0, years=YEARS, mismatch_scale=s)
            res = residual(tr["T_s"], tr["S"], Fa, dt_out, params, cfg)
            row[label] = weighted_rms(res, grid.w)
        report["residual_rms_Wm2"][str(s)] = row
        print(f"{s:>15.2f} {row['unforced']:>14.4g} {row['forced']:>14.4g}")

    with open("mismatch_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote mismatch_report.json")


if __name__ == "__main__":
    main()

"""
Tier-2 signal-and-noise AS A FUNCTION OF MAGNITUDE (reviewer correction).

For the two Tier-2 shapes (constant, ramp), at magnitudes {7.5, 8, 9, 10},
with 5 storm seeds each, report the per-hemisphere polar-winter signal above the
frost point and the storm noise. Does NOT change the Tier-2 range and does NOT
read the dataset -- it runs fresh truth sims with dedicated DIAGNOSTIC seeds
(7000-7004) so no dataset seed ranges or test-split files are touched.

Scale 0.10, D=0.10, n=64, dt=0.05 (frozen). Band |lat| > 60, per hemisphere.
"""

import json

import numpy as np

from common import load_config, Grid
from sim_truth import run_truth
from gen_data import make_schedule, SCALE, SPIN_YEARS

FROST = 147.74
MAGS = [7.5, 8.0, 9.0, 10.0]
SHAPES = ["constant", "ramp"]
DIAG_SEEDS = [7000, 7001, 7002, 7003, 7004]
REC_YEARS = 2


def polar_winter_min(r, sel):
    return float(r["T_s"][:, sel].min())


def main():
    cfg = load_config()
    assert cfg["time"]["dt_sol"] == 0.05 and cfg["physics"]["D"] == 0.10
    grid = Grid(cfg["grid"]["n"])
    latd = np.degrees(grid.lat)
    selN, selS = latd > 60, latd < -60
    year = cfg["planet"]["year_sols"]
    total = REC_YEARS * year

    out = {"band_deg": 60, "frost": FROST, "seeds": DIAG_SEEDS,
           "rec_years": REC_YEARS, "table": {}}
    print("Tier-2 polar-winter signal (wmin - 147.74) and storm noise (std),")
    print("scale 0.10, per hemisphere, 5 storm seeds.\n")
    print(f"{'shape':>8} {'A':>5} | {'N sig':>6} {'N std':>6} {'N S/N':>6} "
          f"| {'S sig':>6} {'S std':>6} {'S S/N':>6}")
    print("-" * 66)
    for shape in SHAPES:
        for A in MAGS:
            wN, wS = [], []
            for seed in DIAG_SEEDS:
                spin = run_truth(cfg, F_a=0.0, seed=seed, years=SPIN_YEARS,
                                 mismatch_scale=SCALE)
                fa, _ = make_schedule(shape, A, total, np.random.default_rng(seed))
                r = run_truth(cfg, F_a=fa, seed=seed, years=REC_YEARS,
                              mismatch_scale=SCALE,
                              T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                              m_init=spin["m_final"])
                wN.append(polar_winter_min(r, selN))
                wS.append(polar_winter_min(r, selS))
            wN, wS = np.array(wN), np.array(wS)

            def stat(a):
                sig = float(a.mean() - FROST)
                sd = float(a.std())
                return sig, sd, (sig / sd if sd > 1e-6 else float("inf"))
            sN, dN, rN = stat(wN)
            sS, dS, rS = stat(wS)
            out["table"][f"{shape}_A{A}"] = {
                "N": {"signal": sN, "std": dN, "SN": rN},
                "S": {"signal": sS, "std": dS, "SN": rS}}
            print(f"{shape:>8} {A:>5.1f} | {sN:>6.2f} {dN:>6.3f} {rN:>6.1f} "
                  f"| {sS:>6.2f} {dS:>6.3f} {rS:>6.1f}")

    with open("diagnostics_tier2.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote diagnostics_tier2.json")


if __name__ == "__main__":
    main()

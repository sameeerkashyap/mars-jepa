"""
Round-3 pre-registration diagnostics (simulator only), per reviewer request:

  A. tau-vs-G ablation: scale dust and subsurface coupling SEPARATELY at fixed D
     to TEST (not argue) which suppresses the winter cap.
  B. Explicit threshold at mismatch_scale = 0.25.
  C. Resolution check at n = 128 (threshold + polar signal) vs n = 64.
  D. Polar signal reported per hemisphere (the band is only a few cells).

Truth simulator only; dt = 0.1 sol.
"""

import json
import copy

import numpy as np

from common import Grid, load_config
from sim_truth import run_truth


def base_cfg(n=64):
    c = load_config()
    c["time"]["dt_sol"] = 0.1
    c["grid"]["n"] = n
    c["physics"]["D"] = 0.10          # frozen
    return c


def _with(cfg, **over):
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        sect, key = k.split(".")
        c[sect][key] = v
    return c


def run_meas(cfg, scale, F_a, seed, spin=3, meas=1):
    s = run_truth(cfg, F_a=F_a, seed=seed, years=spin, mismatch_scale=scale)
    return run_truth(cfg, F_a=F_a, seed=seed, years=meas, mismatch_scale=scale,
                     T_init=s["T_s_final"], Tg_init=s["T_g_final"],
                     m_init=s["m_final"])


def hemi_winter_min(r, grid, hemi, band=60.0):
    latd = np.degrees(grid.lat)
    sel = (latd > band) if hemi == "N" else (latd < -band)
    return float(r["T_s"][:, sel].min()), int(sel.sum())


def cap_by_hemi(r, grid):
    latd = np.degrees(grid.lat)
    n = float(r["m"][:, latd > 0].max())
    s = float(r["m"][:, latd < 0].max())
    return n, s


def main():
    out = {}

    # ----------------------------------------------- A. tau vs G ablation
    print("=== A. tau-vs-G ablation (D=0.10, base scale=0.10, F_a=0, storms off) ===")
    print("   mechanism test: does dust (tau) or the subsurface buffer (G) kill the cap?")
    cfg = base_cfg(64)
    grid = Grid(64)
    s = 0.10
    # four worlds built purely from config knobs:
    #   both:  dust on (scale s), G on (scale s)   -> normal
    #   tauon: dust on (scale s), G off (G=0)
    #   Gon:   dust off (tau=0),  G on (scale s)
    #   none:  dust off, G off (= base climate)
    cfg_dust_off = _with(cfg, **{"truth.dust_base": 0.0,
                                 "truth.dust_seasonal_amp": 0.0,
                                 "truth.storm_rate_per_year": 0.0})
    cfg_storms_off = _with(cfg, **{"truth.storm_rate_per_year": 0.0})
    cfg_Goff = _with(cfg_storms_off, **{"truth.G": 0.0})
    cfg_dust_off_storms = _with(cfg_dust_off, **{"truth.storm_rate_per_year": 0.0})

    worlds = {
        "both_on   (dust+G)": (cfg_storms_off, s),
        "tau_only  (G=0)   ": (cfg_Goff, s),
        "G_only    (tau=0) ": (cfg_dust_off_storms, s),
        "none      (base)  ": (cfg_dust_off_storms, 0.0),
    }
    out["ablation"] = {}
    for name, (c, sc) in worlds.items():
        r = run_meas(c, sc, 0.0, 0)
        capN, capS = cap_by_hemi(r, grid)
        wminN, _ = hemi_winter_min(r, grid, "N")
        wminS, _ = hemi_winter_min(r, grid, "S")
        out["ablation"][name.strip()] = {
            "cap_N": capN, "cap_S": capS, "wmin_N": wminN, "wmin_S": wminS}
        print(f"  {name}: capN={capN:4.0f} capS={capS:4.0f} "
              f"wminN={wminN:6.2f} wminS={wminS:6.2f}")
    print("  -> compare 'none' vs 'G_only' (does G warm the pole?) and "
          "'none' vs 'tau_only' (dust sign).")

    # ----------------------------------------------- B. threshold at scale 0.25
    print("\n=== B. Threshold at mismatch_scale=0.25 (D=0.10, storms off) ===")
    cfg = _with(base_cfg(64), **{"truth.storm_rate_per_year": 0.0})
    grid = Grid(64)
    out["threshold_scale0.25"] = {}
    print(f"{'F_a':>5} {'capN':>6} {'capS':>6} {'wminN':>7} {'wminS':>7}")
    for Fa in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
        r = run_meas(cfg, 0.25, Fa, 0)
        capN, capS = cap_by_hemi(r, grid)
        wminN, _ = hemi_winter_min(r, grid, "N")
        wminS, _ = hemi_winter_min(r, grid, "S")
        out["threshold_scale0.25"][str(Fa)] = {
            "cap_N": capN, "cap_S": capS, "wmin_N": wminN, "wmin_S": wminS}
        print(f"{Fa:>5.1f} {capN:>6.0f} {capS:>6.0f} {wminN:>7.2f} {wminS:>7.2f}")

    # ----------------------------------------------- C. resolution check n=128
    print("\n=== C. Resolution check: threshold at n=128 vs n=64 "
          "(scale=0.10, storms off) ===")
    out["resolution_threshold"] = {}
    for n in [64, 128]:
        cfg = _with(base_cfg(n), **{"truth.storm_rate_per_year": 0.0})
        grid = Grid(n)
        line = f"  n={n:>3}: "
        row = {}
        for Fa in [4.0, 5.0, 6.0, 7.0, 8.0]:
            r = run_meas(cfg, 0.10, Fa, 0)
            cap = float(r["m"].max())
            wmin = min(hemi_winter_min(r, grid, "N")[0],
                       hemi_winter_min(r, grid, "S")[0])
            row[str(Fa)] = {"cap": cap, "wmin": wmin}
            line += f"F{Fa:g}:{cap:4.0f}/{wmin:5.1f}  "
        out["resolution_threshold"][str(n)] = row
        nb = (np.abs(np.degrees(grid.lat)) > 60).sum()
        print(line + f"(band cells |lat|>60: {nb})")

    # ----------------------------------------------- D. polar signal per hemisphere
    print("\n=== D. Polar-winter signal per hemisphere, storms ON (scale=0.10) ===")
    print("   (signal = wmin - 147.74; noise = std over seeds)")
    seeds = list(range(5))
    out["polar_signal_hemi"] = {}
    for n in [64, 128]:
        cfg = base_cfg(n)
        grid = Grid(n)
        print(f" -- n={n} --")
        out["polar_signal_hemi"][str(n)] = {}
        for Fa in ([6.0, 8.0, 10.0] if n == 64 else [8.0, 10.0]):
            resN = np.array([hemi_winter_min(run_meas(cfg, 0.10, Fa, sd), grid, "N")[0]
                             for sd in seeds])
            resS = np.array([hemi_winter_min(run_meas(cfg, 0.10, Fa, sd), grid, "S")[0]
                             for sd in seeds])
            def stat(a):
                m, sd_ = float(a.mean()), float(a.std())
                sig = m - 147.74
                return m, sd_, sig, (sig / sd_ if sd_ > 1e-6 else float("inf"))
            mN, sN, sigN, snN = stat(resN)
            mS, sS, sigS, snS = stat(resS)
            out["polar_signal_hemi"][str(n)][str(Fa)] = {
                "N": {"mean": mN, "std": sN, "signal": sigN, "SN": snN},
                "S": {"mean": mS, "std": sS, "signal": sigS, "SN": snS}}
            print(f"  F_a={Fa:>4.1f}  N: sig={sigN:5.2f} std={sN:5.3f} S/N={snN:5.1f}"
                  f"   |  S: sig={sigS:5.2f} std={sS:5.3f} S/N={snS:5.1f}")

    with open("diagnostics_c.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote diagnostics_c.json")
    return out


if __name__ == "__main__":
    main()

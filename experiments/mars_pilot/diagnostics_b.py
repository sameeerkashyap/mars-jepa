"""
Round-2 pre-registration diagnostics requested by the reviewer:

  1. Threshold location vs D in {0.07, 0.10, 0.15, 0.20}, plus winter cap extent
     (equatorward latitude) and the seasonal CO2 cycle at D=0.10, to sanity-check
     against real Mars.
  2. Polar-band storm noise at F_a = {5, 6, 8, 10} (std across seeds) for the
     primary metric (polar-winter T_s), to confirm the few-K signal clears noise.
  3. (Text) confirmation of the observable/hidden split and the polar metric.

Simulator only. Uses dt=0.1 sol for speed (annual means within ~0.1 K of dt=0.05).
"""

import json
import copy

import numpy as np

from common import Grid, load_config
from sim_truth import run_truth

POLAR_LAT = 60.0        # polar band: |latitude| > 60 deg (primary-metric band)


def cfg_fast():
    c = load_config()
    c["time"]["dt_sol"] = 0.1
    return c


def _with(cfg, **over):
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        sect, key = k.split(".")
        c[sect][key] = v
    return c


def run_meas(cfg, scale, F_a, seed, D=None, spin=3, meas=1):
    """Spin up then return the measurement-year run dict."""
    c = cfg if D is None else _with(cfg, **{"physics.D": D})
    s = run_truth(c, F_a=F_a, seed=seed, years=spin, mismatch_scale=scale)
    r = run_truth(c, F_a=F_a, seed=seed, years=meas, mismatch_scale=scale,
                  T_init=s["T_s_final"], Tg_init=s["T_g_final"],
                  m_init=s["m_final"])
    return r


def polar_winter_min(r, grid):
    """Minimum T_s over the year within the polar band (either hemisphere)."""
    band = np.abs(np.degrees(grid.lat)) > POLAR_LAT
    return float(r["T_s"][:, band].min())


def cap_extent_deg(r, grid):
    """Most equatorward latitude (deg) at which CO2 ice ever forms, per hemisphere.
    Returns (north_edge, south_edge) as absolute latitudes; None if no ice."""
    latd = np.degrees(grid.lat)
    ever_iced = (r["m"] > 0.0).any(axis=0)       # per cell
    north = latd[(latd > 0) & ever_iced]
    south = latd[(latd < 0) & ever_iced]
    n_edge = float(north.min()) if north.size else None   # smallest +lat with ice
    s_edge = float(south.max()) if south.size else None   # largest -lat (closest to eq)
    return n_edge, s_edge


def main():
    cfg = cfg_fast()
    grid = Grid(cfg["grid"]["n"])
    w = grid.w
    out = {}

    # ---------------------------------------------------------------- 1. threshold vs D
    print("=== 1. Threshold location vs D (scale=0.10, storms off) ===")
    print("   threshold = lowest F_a at which the winter cap no longer forms")
    cfg_ns = _with(cfg, **{"truth.storm_rate_per_year": 0.0})
    Fa_scan = [0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    out["threshold_vs_D"] = {}
    print(f"{'D':>6} " + " ".join(f"F{f:g}:cap/wmin" for f in Fa_scan))
    for D in [0.07, 0.10, 0.15, 0.20]:
        row = {}
        cap_gone_at = None
        line = f"{D:>6.2f} "
        for Fa in Fa_scan:
            r = run_meas(cfg_ns, 0.10, Fa, 0, D=D)
            cap = r["m"].max()
            wmin = polar_winter_min(r, grid)
            row[str(Fa)] = {"cap_mass": float(cap), "winter_min": wmin}
            if cap <= 1.0 and cap_gone_at is None:
                cap_gone_at = Fa
            line += f" {cap:4.0f}/{wmin:5.1f}"
        row["threshold_Fa"] = cap_gone_at
        out["threshold_vs_D"][str(D)] = row
        print(line + f"   -> threshold ~F_a={cap_gone_at}")

    # ---------------------------------------------------------------- cap extent & cycle
    print("\n=== cap extent & seasonal cycle at D=0.10 ===")
    out["cap_extent"] = {}
    for scale in [0.0, 0.10]:
        r = run_meas(_with(cfg, **{"physics.D": 0.10}), scale, 0.0, 0)
        n_edge, s_edge = cap_extent_deg(r, grid)
        glob_ice = (r["m"] * w).sum(axis=1)       # area-weighted mean ice (kg/m2)
        out["cap_extent"][str(scale)] = {
            "north_cap_edge_deg": n_edge, "south_cap_edge_deg": s_edge,
            "cap_mass_peak_kg_m2": float(r["m"].max()),
            "global_mean_ice_peak_kg_m2": float(glob_ice.max()),
            "global_mean_ice_min_kg_m2": float(glob_ice.min()),
        }
        print(f" scale={scale}: cap reaches N {n_edge}  S {s_edge} (abs lat deg); "
              f"peak cap {r['m'].max():.0f} kg/m2; "
              f"global-mean ice {glob_ice.min():.1f}->{glob_ice.max():.1f} kg/m2")
    print(" Mars reference (verify in paper): seasonal CO2 caps reach ~50-55 deg")
    print("   latitude at maximum; NOT a quantitative match (toy: no transport,")
    print("   no pressure feedback).")

    # ---------------------------------------------------------------- 2. polar storm noise
    print("\n=== 2. Polar-band storm noise (|lat|>60), scale=0.10, storms ON ===")
    print("   metric = polar-winter min T_s; signal = rise above frost (147.74 K)")
    seeds = list(range(5))
    out["polar_storm_noise"] = {"band_deg": POLAR_LAT, "seeds": seeds}
    print(f"{'F_a':>5} {'mean_wmin':>10} {'std':>7} {'range':>7} "
          f"{'signal(>147.74)':>16} {'S/N':>7}")
    for Fa in [5.0, 6.0, 8.0, 10.0]:
        vals = [polar_winter_min(run_meas(cfg, 0.10, Fa, sd), grid) for sd in seeds]
        vals = np.array(vals)
        mean, std, rng = float(vals.mean()), float(vals.std()), float(np.ptp(vals))
        signal = mean - 147.74
        sn = signal / std if std > 1e-6 else float("inf")
        out["polar_storm_noise"][str(Fa)] = {
            "mean_winter_min": mean, "std": std, "range": rng,
            "signal_above_frost": signal, "S_over_N": sn}
        print(f"{Fa:>5.1f} {mean:>10.2f} {std:>7.3f} {rng:>7.3f} "
              f"{signal:>16.2f} {sn:>7.1f}")

    with open("diagnostics_b.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote diagnostics_b.json")
    return out


if __name__ == "__main__":
    main()

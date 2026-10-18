"""
Simulator-only diagnostics to inform the choice of X (forcing range) and
mismatch_scale, BEFORE any model is trained (Checkpoint B not started).

Produces (prints + writes DIAGNOSTICS.md / diagnostics.json):

  1. D sensitivity: does the DUST-FREE base climate (scale=0) form caps?
  2. Mismatch residual vs mismatch_scale, with absorbed-SW fraction and
     ratio to the forcing.
  3. Forced-response table: global-mean & polar dT vs F_a (storms off).
  4. Storm noise: year-to-year / seed spread of annual-mean T (storms on).
  5. Linearity check: fit a line on [0, X], predict [1.5X, 2X].

All runs are the TRUTH simulator only.
"""

import json

import numpy as np

from common import Grid, load_config, SIGMA
from sim_truth import run_truth
from sim_ebm import fit_ebm, residual


def _cfg_with(cfg, **over):
    import copy
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        sect, key = k.split(".")
        c[sect][key] = v
    return c


def annual_metrics(cfg, scale, F_a, seed, storms, spin_years=3, meas_years=1):
    """Run truth and return annual-mean diagnostics over the last meas_years."""
    c = cfg if storms else _cfg_with(cfg, **{"truth.storm_rate_per_year": 0.0})
    grid = Grid(c["grid"]["n"])
    w = grid.w
    s = run_truth(c, F_a=F_a, seed=seed, years=spin_years, mismatch_scale=scale)
    r = run_truth(c, F_a=F_a, seed=seed, years=meas_years, mismatch_scale=scale,
                  T_init=s["T_s_final"], Tg_init=s["T_g_final"],
                  m_init=s["m_final"])
    Ts = r["T_s"]
    gmean = (Ts * w).sum(axis=1)                     # global-mean per time
    polar = 0.5 * (Ts[:, 0] + Ts[:, -1])             # mean of the two end cells
    absorbed = (r["absorbed"] * w).sum(axis=1)
    return {
        "gmean_annual": float(gmean.mean()),
        "polar_annual": float(polar.mean()),
        "polar_winter_min": float(min(Ts[:, 0].min(), Ts[:, -1].min())),
        "cap_mass_max": float(r["m"].max()),
        "absorbed_mean": float(absorbed.mean()),
        "T_frost": float(r["T_frost"]),
    }


# -------------------------------------------------------------- 1. D sensitivity
def d_sensitivity(cfg):
    rows = []
    for D in [0.10, 0.15, 0.20]:
        c = _cfg_with(cfg, **{"physics.D": D})
        base = annual_metrics(c, scale=0.0, F_a=0.0, seed=0, storms=False)
        forced = annual_metrics(c, scale=0.25, F_a=10.0, seed=0, storms=False)
        base0 = annual_metrics(c, scale=0.25, F_a=0.0, seed=0, storms=False)
        rows.append({
            "D": D,
            "base_gmean": base["gmean_annual"],
            "base_winter_min": base["polar_winter_min"],
            "base_cap_mass": base["cap_mass_max"],
            "dT_global_Fa10_scale0.25": forced["gmean_annual"] - base0["gmean_annual"],
            "dT_polar_Fa10_scale0.25": forced["polar_annual"] - base0["polar_annual"],
        })
    return rows


# -------------------------------------------- 2. mismatch residual vs scale
def mismatch_table(cfg, scales, forced_Fa=10.0):
    grid = Grid(cfg["grid"]["n"])
    w = grid.w
    sol_s = cfg["planet"]["sol_seconds"]
    dt_out = cfg["time"]["output_every_sol"] * sol_s

    # fit once on scale-0 unforced run
    base = run_truth(cfg, F_a=0.0, seed=0, years=3, mismatch_scale=0.0)
    params, fit_rms = fit_ebm(cfg, base)

    def wrms(res):
        vals = []
        for row in res**2:
            g = ~np.isnan(row)
            if g.any():
                vals.append(np.sum(w[g] * row[g]) / np.sum(w[g]))
        return float(np.sqrt(np.mean(vals)))

    rows = []
    for s in scales:
        row = {"scale": s}
        for label, Fa in [("unforced", 0.0), ("forced", forced_Fa)]:
            tr = run_truth(cfg, F_a=Fa, seed=0, years=3, mismatch_scale=s)
            res = residual(tr["T_s"], tr["S"], Fa, dt_out, params, cfg)
            absorbed = (tr["absorbed"] * w).sum(axis=1).mean()
            r = wrms(res)
            row[f"{label}_rms"] = r
            row[f"{label}_absorbed"] = float(absorbed)
            row[f"{label}_pct_absorbed"] = 100.0 * r / absorbed
        rows.append(row)
    return params, fit_rms, forced_Fa, rows


# ------------------------------------------- 3. forced response (storms off)
def forced_response(cfg, scale, Fa_list):
    # base is the F_a = 0 entry (must be present in Fa_list)
    metrics = {Fa: annual_metrics(cfg, scale=scale, F_a=Fa, seed=0, storms=False)
               for Fa in Fa_list}
    base = metrics[0.0]
    rows = []
    for Fa in Fa_list:
        m = metrics[Fa]
        rows.append({
            "F_a": Fa,
            "gmean": m["gmean_annual"],
            "dT_global": m["gmean_annual"] - base["gmean_annual"],
            "polar": m["polar_annual"],
            "dT_polar": m["polar_annual"] - base["polar_annual"],
            "cap_mass": m["cap_mass_max"],
        })
    return rows


# ------------------------------------------- 4. storm noise across seeds
def storm_noise(cfg, scale, seeds):
    g, p = [], []
    for sd in seeds:
        m = annual_metrics(cfg, scale=scale, F_a=0.0, seed=sd, storms=True)
        g.append(m["gmean_annual"])
        p.append(m["polar_annual"])
    return {
        "scale": scale,
        "n_seeds": len(seeds),
        "gmean_mean": float(np.mean(g)),
        "gmean_std": float(np.std(g)),
        "gmean_range": float(np.ptp(g)),
        "polar_mean": float(np.mean(p)),
        "polar_std": float(np.std(p)),
        "polar_range": float(np.ptp(p)),
    }


# ------------------------------------------- 5. linearity check
def linearity(rows, scale, X):
    Fa = np.array([r["F_a"] for r in rows])
    dT = np.array([r["dT_global"] for r in rows])
    dTp = np.array([r["dT_polar"] for r in rows])

    train = Fa <= X
    test_lo, test_hi = 1.5 * X, 2.0 * X

    def fit_and_extrapolate(y):
        a, b = np.polyfit(Fa[train], y[train], 1)       # linear fit on [0, X]
        qa, qb, qc = np.polyfit(Fa[train], y[train], 2)  # quadratic for curvature
        out = {"slope": float(a), "intercept": float(b),
               "quad_coef": float(qa), "extrap": {}}
        for f in Fa:
            if test_lo - 1e-6 <= f <= test_hi + 1e-6:
                pred = a * f + b
                out["extrap"][str(f)] = {
                    "actual": float(y[Fa == f][0]),
                    "linear_pred": float(pred),
                    "abs_err_K": float(abs(y[Fa == f][0] - pred)),
                }
        return out

    return {
        "scale": scale, "X": X, "test_range": [test_lo, test_hi],
        "global": fit_and_extrapolate(dT),
        "polar": fit_and_extrapolate(dTp),
    }


def main():
    cfg = load_config()
    # faster timestep for these steady-state diagnostics (radiative timescale
    # ~3 sol; dt=0.1 sol is well within explicit stability and reproduces the
    # dt=0.05 annual means to <~0.1 K).
    cfg = _cfg_with(cfg, **{"time.dt_sol": 0.1})
    Fa_list = [0.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    scales = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    seeds = list(range(5))
    out = {}

    print("=== 1. D sensitivity (base = dust-free scale=0 climate) ===")
    out["d_sensitivity"] = d_sensitivity(cfg)
    for r in out["d_sensitivity"]:
        print(f"D={r['D']:.2f}  base_gmean={r['base_gmean']:.1f}K  "
              f"winter_min={r['base_winter_min']:.1f}K  "
              f"cap={r['base_cap_mass']:.0f}kg/m2  "
              f"dT_glob(Fa10,s0.25)={r['dT_global_Fa10_scale0.25']:.2f}K  "
              f"dT_polar={r['dT_polar_Fa10_scale0.25']:.2f}K")

    print("\n=== 2. Mismatch residual vs scale (fit on scale=0 unforced) ===")
    params, fit_rms, fFa, rows = mismatch_table(cfg, scales)
    out["mismatch"] = {"fit_params": params, "fit_rms": fit_rms,
                       "forced_Fa": fFa, "rows": rows}
    print(f"EBM fit: C_eff={params['C_eff']:.4g} D={params['D']:.4g} "
          f"eps={params['eps']:.4g} fit_rms={fit_rms:.3g} W/m2")
    print(f"{'scale':>6} {'unf_rms':>8} {'unf_%abs':>9} "
          f"{'for_rms':>8} {'for_%abs':>9} {'for_rms/Fa':>11}")
    for r in rows:
        print(f"{r['scale']:>6.2f} {r['unforced_rms']:>8.2f} "
              f"{r['unforced_pct_absorbed']:>8.1f}% {r['forced_rms']:>8.2f} "
              f"{r['forced_pct_absorbed']:>8.1f}% "
              f"{r['forced_rms']/fFa:>11.2f}")

    print("\n=== 3. Forced-response table (storms OFF) ===")
    out["forced_response"] = {}
    fr_cache = {}
    for sc in [0.1, 0.25]:
        rows = forced_response(cfg, sc, Fa_list)
        fr_cache[sc] = rows
        out["forced_response"][str(sc)] = rows
        print(f"-- mismatch_scale={sc} --")
        print(f"{'F_a':>6} {'gmean':>8} {'dT_glob':>8} "
              f"{'polar':>8} {'dT_pol':>8} {'cap':>7}")
        for r in rows:
            print(f"{r['F_a']:>6.1f} {r['gmean']:>8.1f} {r['dT_global']:>8.2f} "
                  f"{r['polar']:>8.1f} {r['dT_polar']:>8.2f} "
                  f"{r['cap_mass']:>7.0f}")

    print("\n=== 4. Storm noise (storms ON, unforced, 6 seeds) ===")
    out["storm_noise"] = {}
    for sc in [0.1, 0.25]:
        n = storm_noise(cfg, sc, seeds)
        out["storm_noise"][str(sc)] = n
        print(f"scale={sc}: gmean {n['gmean_mean']:.2f}K "
              f"(std {n['gmean_std']:.3f}, range {n['gmean_range']:.3f}) | "
              f"polar {n['polar_mean']:.2f}K "
              f"(std {n['polar_std']:.3f}, range {n['polar_range']:.3f})")

    print("\n=== 5. Linearity check (fit line on [0,X], predict [1.5X,2X]) ===")
    out["linearity"] = {}
    for sc in [0.1, 0.25]:
        for X in [5.0]:
            lin = linearity(fr_cache[sc], sc, X)
            out["linearity"][f"scale{sc}_X{X}"] = lin
            print(f"-- scale={sc}, X={X}, test={lin['test_range']} --")
            for which in ["global", "polar"]:
                b = lin[which]
                print(f"  {which}: slope={b['slope']:.3f} K/(W/m2), "
                      f"quad_coef={b['quad_coef']:.4f}")
                for f, e in b["extrap"].items():
                    print(f"    F_a={f}: actual={e['actual']:.2f} "
                          f"linpred={e['linear_pred']:.2f} "
                          f"err={e['abs_err_K']:.3f}K")

    with open("diagnostics.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote diagnostics.json")
    return out


if __name__ == "__main__":
    main()

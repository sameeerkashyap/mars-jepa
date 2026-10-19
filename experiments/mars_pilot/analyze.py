"""
analyze.py -- figures + PREREG hypothesis table from results.csv.

Outputs (into figures_phase2/ and analysis_report.{json,md}):
  * headline: OOD intervention-response error (Tier-2 S-pole) vs data fraction,
    one curve per configuration (arm+physics), mean with bootstrap CI over seeds;
  * rollout error vs horizon (from per-run runs/<tag>/eval.json when present);
  * calibration plot (coverage_90 and CRPS per configuration);
  * a table evaluating H1/H2/H3 against thresholds.yaml, each labeled
    supported / not supported / inconclusive;
  * a DESCRIPTIVE table (no thresholds, no verdicts) of each configuration's
    primary-metric error (PREREG D7: Tier-2 S control-pinned polar winter,
    column t2_polar_winter_S_pinned) next to the LINEAR and HYBRID baselines,
    mean and per-seed, per data fraction.

Designed to run on FAKE results (see tests) as well as the real sweep output.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = "figures_phase2"
EBM_METRIC_LABEL = ("EBM-residual constraint satisfaction (also the training objective for physics arms)")
PURE_ARMS = ["gru", "rssm", "transformer"]
PRIMARY_METRIC = "t2_polar_winter_S_pinned"          # PREREG D7
BASELINE_CONFIGS = ("linear/off", "hybrid/off")


def _config_label(row):
    return f"{row['arm']}/{row['physics']}"


def load_results(path):
    df = pd.read_csv(path)
    df = df[df.get("status", "ok") == "ok"].copy()
    for c in df.columns:
        if c.startswith(("t1_", "t2_")) or c in ("fraction", "seed"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["config"] = df.apply(_config_label, axis=1)
    return df


def bootstrap_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    if vals.size == 1:
        return vals[0], vals[0], vals[0]
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, vals.size, replace=True).mean()
             for _ in range(n_boot)]
    return (float(vals.mean()), float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def headline_figure(df, metric="t2_polar_winter_S", out=None):
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg, g in df.groupby("config"):
        fracs = sorted(g["fraction"].unique())
        mean, lo, hi = [], [], []
        for f in fracs:
            vals = g[g["fraction"] == f][metric].values
            m, l, h = bootstrap_ci(vals)
            mean.append(m); lo.append(l); hi.append(h)
        mean = np.array(mean)
        ax.plot(fracs, mean, marker="o", label=cfg)
        ax.fill_between(fracs, lo, hi, alpha=0.18)
    ax.set_xlabel("training data fraction")
    ax.set_ylabel(f"OOD Tier-2 response RMSE [{metric}] (K)")
    ax.set_title("Headline: OOD intervention-response error vs data fraction")
    ax.legend(fontsize=8)
    out = out or os.path.join(FIGDIR, "headline_ood_vs_fraction.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def rollout_figure(df, run_root="runs", out=None):
    os.makedirs(FIGDIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = 0
    for cfg, g in df.groupby("config"):
        curves = []
        for tag in g["tag"].unique():
            fp = os.path.join(run_root, str(tag), "eval.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as fh:
                ev = json.load(fh)
            arr = ev.get("tiers", {}).get("tier2", {}).get(
                "rollout_rmse_vs_horizon")
            if arr:
                curves.append(arr)
        if curves:
            L = min(len(c) for c in curves)
            arr = np.array([c[:L] for c in curves])
            ax.plot(range(L), arr.mean(0), label=cfg)
            plotted += 1
    ax.set_xlabel("rollout horizon step")
    ax.set_ylabel("Tier-2 forced global-mean RMSE (K)")
    ax.set_title("Rollout error vs horizon")
    if plotted:
        ax.legend(fontsize=8)
    out = out or os.path.join(FIGDIR, "rollout_vs_horizon.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def calibration_figure(df, out=None):
    os.makedirs(FIGDIR, exist_ok=True)
    cal = df.dropna(subset=["t2_cov90"]) if "t2_cov90" in df else df.iloc[0:0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(cal):
        grp = cal.groupby("config")
        labels = list(grp.groups.keys())
        cov = [grp.get_group(k)["t2_cov90"].mean() for k in labels]
        crps = [grp.get_group(k)["t2_crps"].mean() for k in labels]
        axes[0].bar(range(len(labels)), cov)
        axes[0].axhline(0.9, color="k", ls="--", lw=1, label="target 0.90")
        axes[0].set_xticks(range(len(labels)))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("90% interval coverage"); axes[0].legend()
        axes[1].bar(range(len(labels)), crps)
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("CRPS (K)")
    else:
        axes[0].text(0.5, 0.5, "no calibration rows", ha="center")
    fig.suptitle("Calibration (stochastic models + ensemble)")
    out = out or os.path.join(FIGDIR, "calibration.png")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


# --------------------------------------------------------------------------
def _mean_metric(df, arm, physics, fraction, metric):
    g = df[(df["arm"] == arm) & (df["physics"] == physics)
           & (np.isclose(df["fraction"], fraction))]
    v = pd.to_numeric(g[metric], errors="coerce").dropna().values
    return (float(np.mean(v)), len(v)) if len(v) else (np.nan, 0)


def evaluate_hypotheses(df, thr):
    out = {}
    maxf = df["fraction"].max()
    metric = thr.get("h1_metric") or "t2_polar_winter_S"
    t1_metric = metric.replace("t2_", "t1_")

    # H1
    r = {"name": "H1 (OOD gap exists)"}
    t2 = [_mean_metric(df, a, "off", maxf, metric)[0] for a in PURE_ARMS]
    t1 = [_mean_metric(df, a, "off", maxf, t1_metric)[0] for a in PURE_ARMS]
    t2m = np.nanmean(t2); t1m = np.nanmean(t1)
    ratio = t2m / t1m if (t1m and np.isfinite(t1m) and t1m != 0) else np.nan
    r["observed_ratio"] = float(ratio) if np.isfinite(ratio) else None
    r["threshold"] = thr.get("h1_ratio")
    r["verdict"] = _verdict(ratio, thr.get("h1_ratio"), ">=")
    out["H1"] = r

    # H2 (relative OOD reduction of physics vs off)
    r = {"name": "H2 (physics helps extrapolation)"}
    arch = thr.get("h2_arch") or "gru"
    variant = thr.get("h2_variant") or "A"
    off_m, n_off = _mean_metric(df, arch, "off", maxf, metric)
    phys_m, n_phys = _mean_metric(df, arch, variant, maxf, metric)
    rel = (off_m - phys_m) / off_m if (off_m and np.isfinite(off_m)) else np.nan
    r["arch"] = arch; r["variant"] = variant
    r["observed_rel_reduction"] = float(rel) if np.isfinite(rel) else None
    r["threshold"] = thr.get("h2_rel_reduction")
    r["n_seeds"] = int(min(n_off, n_phys))
    min_seeds = thr.get("h2_min_seeds")
    if min_seeds is None:
        r["verdict"] = "inconclusive (threshold unset)"
    else:
        r["verdict"] = _verdict(rel, thr.get("h2_rel_reduction"), ">=",
                                extra_ok=r["n_seeds"] >= min_seeds)
    out["H2"] = r

    # H3 (data efficiency + hybrid)
    r = {"name": "H3 (data efficiency / hybrid helps)"}
    arch = thr.get("h3_arch") or "gru"
    off_full, _ = _mean_metric(df, arch, "off", maxf, metric)
    # smallest fraction at which physics=A reaches off_full
    reach = None
    for f in sorted(df["fraction"].unique()):
        pm, _ = _mean_metric(df, arch, "A", f, metric)
        if np.isfinite(pm) and np.isfinite(off_full) and pm <= off_full:
            reach = f; break
    r["frac_physics_reaches_off_full"] = reach
    r["frac_ratio_threshold"] = thr.get("h3_frac_ratio")
    de_ok = (reach is not None and thr.get("h3_frac_ratio") is not None
             and reach <= thr["h3_frac_ratio"] * maxf)
    # hybrid vs best pure
    hyb, _ = _mean_metric(df, "hybrid", "off", maxf, metric)
    pure = [_mean_metric(df, a, "off", maxf, metric)[0] for a in PURE_ARMS]
    best_pure = np.nanmin(pure) if len(pure) else np.nan
    margin = thr.get("h3_hybrid_margin")
    hy_ok = (np.isfinite(hyb) and np.isfinite(best_pure) and margin is not None
             and hyb <= margin * best_pure)
    r["hybrid"] = float(hyb) if np.isfinite(hyb) else None
    r["best_pure"] = float(best_pure) if np.isfinite(best_pure) else None
    if thr.get("h3_frac_ratio") is None and margin is None:
        r["verdict"] = "inconclusive (threshold unset)"
    else:
        r["verdict"] = "supported" if (de_ok or hy_ok) else "not supported"
    out["H3"] = r
    return out


def _verdict(obs, threshold, op, extra_ok=True):
    if threshold is None:
        return "inconclusive (threshold unset)"
    if obs is None or not np.isfinite(obs):
        return "inconclusive (no data)"
    cond = (obs >= threshold) if op == ">=" else (obs <= threshold)
    return "supported" if (cond and extra_ok) else "not supported"


def ebm_metric_table(df):
    """Per-config mean of the EBM-residual metric (Tier 1 / Tier 2), labeled."""
    rows = {}
    for cfg, g in df.groupby("config"):
        rows[cfg] = {c: (float(g[c].mean()) if c in g and g[c].notna().any()
                         else None) for c in ("t1_ebm_viol", "t2_ebm_viol")}
    return {"label": EBM_METRIC_LABEL, "by_config": rows}


def _seed_values(df, config, fraction, metric):
    """{seed: value} for one configuration at one data fraction (NaN kept)."""
    g = df[(df["config"] == config) & np.isclose(df["fraction"], fraction)]
    return {int(sd): float(v) for sd, v in zip(g["seed"], g[metric])}


def _summ(per_seed):
    v = np.array([x for x in per_seed.values() if np.isfinite(x)], float)
    return {"mean": float(v.mean()) if v.size else None,
            "n_seeds_finite": int(v.size), "per_seed": per_seed}


def primary_vs_baselines(df, metric=PRIMARY_METRIC,
                         baselines=BASELINE_CONFIGS):
    """Descriptive only: per (configuration, fraction), the metric's mean and
    per-seed values next to each baseline's at the same fraction.  No
    thresholds, no verdicts."""
    rows = []
    if metric not in df:
        return {"metric": metric, "baselines": list(baselines), "rows": rows}
    for frac in sorted(df["fraction"].unique()):
        base = {b: _summ(_seed_values(df, b, frac, metric)) for b in baselines}
        for cfg in sorted(df[np.isclose(df["fraction"], frac)]["config"].unique()):
            rows.append({"config": cfg, "fraction": float(frac),
                         "arm": _summ(_seed_values(df, cfg, frac, metric)),
                         "baselines": base})
    return {"metric": metric, "baselines": list(baselines), "rows": rows}


def _fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.4g}"


def _fmt_seeds(per_seed):
    return ", ".join(f"s{k}={_fmt(v)}" for k, v in sorted(per_seed.items())) or "-"


def write_report(hyps, figs, out_json="analysis_report.json",
                 out_md="analysis_report.md", ebm=None, primary=None):
    with open(out_json, "w") as fh:
        json.dump({"hypotheses": hyps, "figures": figs, "ebm_metric": ebm,
                   "primary_vs_baselines": primary}, fh, indent=2)
    lines = ["# Phase-2 analysis report", "",
             "## PREREG hypotheses", "",
             "| hypothesis | observed | threshold | verdict |",
             "|---|---|---|---|"]
    for key, h in hyps.items():
        if key == "H1":
            obs = h.get("observed_ratio")
        elif key == "H2":
            obs = h.get("observed_rel_reduction")
        else:
            obs = h.get("frac_physics_reaches_off_full")
        thr = h.get("threshold", h.get("frac_ratio_threshold"))
        lines.append(f"| {h['name']} | {obs} | {thr} | {h['verdict']} |")
    if ebm and ebm["by_config"]:
        lines += ["", f"## {ebm['label']}", "",
                  "Mean normalized squared EBM residual of the model's forced "
                  "rollouts. Physics-loss arms were trained to minimize this "
                  "quantity, so lower values for them are expected by "
                  "construction and are not independent evidence.", "",
                  "| config | Tier 1 | Tier 2 |", "|---|---|---|"]
        for cfg, v in ebm["by_config"].items():
            lines.append(f"| {cfg} | {v['t1_ebm_viol']} | {v['t2_ebm_viol']} |")
    if primary and primary["rows"]:
        bl = primary["baselines"]
        lines += ["", f"## Primary metric vs baselines (`{primary['metric']}`)",
                  "", "Descriptive only: no thresholds or verdicts. RMSE (K); "
                  "mean over seeds with a finite value, then per-seed values. "
                  "Baselines are taken at the same data fraction.", "",
                  "| config | fraction | mean | per seed | "
                  + " | ".join(f"{b} mean | {b} per seed" for b in bl) + " |",
                  "|---|---|---|---|" + "---|---|" * len(bl)]
        for r in primary["rows"]:
            cells = [r["config"], f"{r['fraction']:g}", _fmt(r["arm"]["mean"]),
                     _fmt_seeds(r["arm"]["per_seed"])]
            for b in bl:
                cells += [_fmt(r["baselines"][b]["mean"]),
                          _fmt_seeds(r["baselines"][b]["per_seed"])]
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Figures", ""]
    for k, v in figs.items():
        lines.append(f"- {k}: `{v}`")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.csv")
    ap.add_argument("--thresholds", default="thresholds.yaml")
    ap.add_argument("--run-root", default="runs")
    args = ap.parse_args()

    df = load_results(args.results)
    thr = yaml.safe_load(open(args.thresholds)) or {}
    figs = {
        "headline": headline_figure(df, metric=thr.get("h1_metric")
                                    or "t2_polar_winter_S"),
        "rollout": rollout_figure(df, run_root=args.run_root),
        "calibration": calibration_figure(df),
    }
    hyps = evaluate_hypotheses(df, thr)
    write_report(hyps, figs, ebm=ebm_metric_table(df),
                 primary=primary_vs_baselines(df))
    print(json.dumps(hyps, indent=2))
    print("figures:", figs)


if __name__ == "__main__":
    main()

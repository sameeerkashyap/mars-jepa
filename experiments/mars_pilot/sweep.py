"""
sweep.py -- standalone, resumable sweep driver.  I (the reviewer) run this.

Grid (config_phase2.yaml -> sweep):
  arms x physics{off,A,B where applicable} x fraction{0.1,0.3,1.0} x 10 seeds.

Each run: train one configuration, evaluate on BOTH test tiers, append ONE row
to results.csv.  Resumable: a run whose (arm,physics,fraction,seed) already has
a row in results.csv is skipped.  Per-run logs go to logs/<tag>.log.

Parallelism: multiprocessing Pool on CPU (default).  GPU-aware: if the resolved
device is cuda/mps, workers is forced to 1 (sequential) to avoid device
contention.  Each worker runs on CPU unless --device overrides.

DO NOT expect the agent to launch the full sweep; use --limit / --tiny to smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import traceback
from multiprocessing import Pool

import torch

from common import load_config
from train import (load_phase2_config, pick_device, train_one, save_run,
                   PHYSICS_ARMS)
import eval as evalmod

RESULTS = "results.csv"
LOGDIR = "logs"
RUNDIR = "runs"

# Polar-winter metrics (eval.POLAR_KEYS), per tier: value, n_windows and
# n_cell_steps contributing, e.g. t2_polar_winter_S_pinned{,_n_windows,_n_cell_steps}.
POLAR_FIELDS = [f"{t}_{k}{suf}" for t in ("t1", "t2") for k in evalmod.POLAR_KEYS
                for suf in ("", "_n_windows", "_n_cell_steps")]

FIELDS = ["tag", "arm", "physics", "fraction", "seed", "lambda", "n_params",
          "val_loss",
          *POLAR_FIELDS,
          "t1_resp_global", "t2_resp_global",
          "t1_ebm_viol", "t2_ebm_viol",
          "t2_crps", "t2_cov90", "t2_spread_ood_rho",
          "wall_sec", "status"]


def polar_columns(prefix, tier_metrics):
    """results.csv polar-winter columns for one tier's eval.evaluate() output."""
    ir = tier_metrics["intervention_response_rmse"]
    cnt = tier_metrics["intervention_response_counts"]
    out = {}
    for k in evalmod.POLAR_KEYS:
        out[f"{prefix}_{k}"] = ir[k]
        out[f"{prefix}_{k}_n_windows"] = cnt[k]["n_windows"]
        out[f"{prefix}_{k}_n_cell_steps"] = cnt[k]["n_cell_steps"]
    return out


def enumerate_grid(p2):
    sw = p2["sweep"]
    runs = []
    for arm in sw["arms"]:
        physics_opts = sw["physics"] if arm in PHYSICS_ARMS else ["off"]
        for phys in physics_opts:
            for frac in sw["fractions"]:
                for seed in sw["seeds"]:
                    runs.append((arm, phys, frac, seed))
    return runs


def tag_for(arm, phys, frac, seed):
    return f"{arm}_{phys}_f{int(frac*100)}_s{seed}"


def _load_done(results_path):
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as fh:
            for row in csv.DictReader(fh):
                done.add(row["tag"])
    return done


def _append_row(results_path, row):
    exists = os.path.exists(results_path)
    if exists:
        with open(results_path) as fh:
            header = next(csv.reader(fh), None)
        if header is not None and header != FIELDS:
            raise RuntimeError(f"{results_path} has a different column set "
                               "(older sweep version?); write to a new file")
    with open(results_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_one(spec):
    (arm, phys, frac, seed, device, epochs, max_batches, data_dir,
     tiers) = spec
    tag = tag_for(arm, phys, frac, seed)
    os.makedirs(LOGDIR, exist_ok=True)
    logpath = os.path.join(LOGDIR, f"{tag}.log")
    t0 = time.time()
    row = {k: "" for k in FIELDS}
    row.update(tag=tag, arm=arm, physics=phys, fraction=frac, seed=seed)
    try:
        p2 = load_phase2_config()
        full = load_config()
        if data_dir:
            p2["data"]["data_dir"] = data_dir
        with open(logpath, "w") as lg:
            lg.write(f"START {tag} device={device}\n")

        if arm == "gru_ensemble":
            member_dirs = []
            for mi in range(p2["arch"]["ensemble"]["n_members"]):
                msd = seed * 100 + mi
                m, stats, info = train_one("gru", "off", frac, msd,
                                           p2["physics"]["lambda_default"], p2,
                                           full, device, epochs=epochs,
                                           max_batches=max_batches)
                rd = os.path.join(RUNDIR, f"{tag}_m{mi}")
                save_run(rd, m, stats, info, p2)
                member_dirs.append(rd)
            val_loss = info["val_loss"]
            n_params = info["n_params"]
            run_dirs, ensemble = member_dirs, True
        else:
            lam = p2["physics"]["lambda_default"]
            model, stats, info = train_one(arm, phys, frac, seed, lam, p2, full,
                                           device, epochs=epochs,
                                           max_batches=max_batches)
            rd = os.path.join(RUNDIR, tag)
            save_run(rd, model, stats, info, p2)
            val_loss, n_params = info["val_loss"], info["n_params"]
            row["lambda"] = info["lambda"]
            run_dirs, ensemble = [rd], False

        row["val_loss"], row["n_params"] = val_loss, n_params

        split_map = {"tier1": "test_tier1", "tier2": "test_tier2"}
        mets = {}
        for tier in tiers:
            mets[tier] = evalmod.evaluate(run_dirs, data_dir or
                                          p2["data"]["data_dir"], "main",
                                          split_map[tier], p2, full, device,
                                          ensemble=ensemble)
        if "tier1" in mets:
            ir = mets["tier1"]["intervention_response_rmse"]
            row.update(polar_columns("t1", mets["tier1"]))
            row.update(t1_resp_global=ir["global_mean"],
                       t1_ebm_viol=mets["tier1"]["energy_budget_violation_mean"])
        if "tier2" in mets:
            ir = mets["tier2"]["intervention_response_rmse"]
            row.update(polar_columns("t2", mets["tier2"]))
            row.update(t2_resp_global=ir["global_mean"],
                       t2_ebm_viol=mets["tier2"]["energy_budget_violation_mean"])
            cal = mets["tier2"].get("calibration")
            if cal:
                row.update(t2_crps=cal["crps"], t2_cov90=cal["coverage_90"],
                           t2_spread_ood_rho=cal["spread_ood_spearman"])
        # persist full metrics for analyze.py (rollout-vs-horizon curves etc.)
        edir = os.path.join(RUNDIR, tag)
        os.makedirs(edir, exist_ok=True)
        with open(os.path.join(edir, "eval.json"), "w") as fh:
            json.dump({"tiers": mets}, fh)
        row["status"] = "ok"
    except Exception:
        row["status"] = "error"
        with open(logpath, "a") as lg:
            lg.write(traceback.format_exc())
    row["wall_sec"] = round(time.time() - t0, 2)
    with open(logpath, "a") as lg:
        lg.write(f"DONE {tag} status={row['status']} {row['wall_sec']}s\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default=None, help="override device")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N pending configs (smoke)")
    ap.add_argument("--tiny", action="store_true",
                    help="tiny grid: first arm of each kind, 1 fraction, 2 seeds")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--tiers", nargs="+", default=["tier1", "tier2"])
    args = ap.parse_args()

    p2 = load_phase2_config()
    device = args.device or pick_device(p2["train"]["device"])
    workers = args.workers
    if device in ("cuda", "mps") and workers > 1:
        print(f"[sweep] device={device}: forcing workers=1 (GPU-aware)")
        workers = 1

    grid = enumerate_grid(p2)
    if args.tiny:
        grid = [g for g in grid if g[2] == p2["sweep"]["fractions"][0]
                and g[3] in (0, 1)]
        seen = set()
        tiny = []
        for g in grid:                       # one physics combo per arm
            key = (g[0], g[3])
            if key in seen:
                continue
            seen.add(key)
            tiny.append(g)
        grid = tiny

    done = _load_done(args.results)
    pending = [g for g in grid if tag_for(*g) not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"[sweep] {len(pending)} pending / {len(grid)} total "
          f"({len(done)} done), workers={workers}, device={device}")

    specs = [(a, p, f, s, device, args.epochs, args.max_batches, args.data_dir,
              args.tiers) for (a, p, f, s) in pending]

    if workers == 1:
        rows = [run_one(sp) for sp in specs]
    else:
        with Pool(workers) as pool:
            rows = pool.map(run_one, specs)
    for row in rows:
        _append_row(args.results, row)
    print(f"[sweep] wrote {len(rows)} rows -> {args.results}")


if __name__ == "__main__":
    main()

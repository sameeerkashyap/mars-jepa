"""
Checkpoint B -- data generation from the TRUTH simulator (expanded).

Three DISJOINT dataset families (seed ranges recorded in the manifest):
  * MAIN        : train / val (forcing [0, X]) + Tier-1 / Tier-2 test splits
  * CALIBRATION : train / val only (forcing [0, X]); for seed-to-seed spread
  * HELD-OUT    : reserved seed range, NOT generated

Design frozen per PREREG.md: D=0.10, n=64, dt=0.05 sol, mismatch_scale=0.10, X=5.

Forcing magnitudes are STRATIFIED across each split's range (per shape).
Each trajectory has a matched no-forcing CONTROL (same seed + initial state);
control T_s and response T_s are stored under the 'eval__' prefix (eval labels),
hidden diagnostics under 'hidden__'.  Neither is returned by the default loader.

USAGE (run the FULL generation yourself, in your own terminal -- NOT via the
agent), e.g. on macOS (spawn):

    python3 gen_data.py --out data --workers 6

It is resumable (safe to re-run; finished files are skipped) and logs to
<out>/gen.log.  A tiny smoke test is allowed via --smoke.

    python3 gen_data.py --smoke --out data_scratch --workers 2
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np

from common import load_config
from sim_truth import run_truth

HIDDEN = "hidden__"
EVAL = "eval__"
X = 5.0
SCALE = 0.10
SPIN_YEARS = 3
RECORD_YEARS = 3

SEED_RANGES = {
    "main":              [1000, 1999],
    "calibration":       [3000, 3999],
    "held_out_reserved": [9000, 9999],   # NOT generated
}

# Tier-2 config: constant forcing ONLY (ramp reaches full forcing only after the
# polar winters and carries no polar-winter signal; see PREREG Deviations).
# The magnitude range is a parameter; default [1.5X, 2X].
TIER2_SHAPES = ["constant"]
TIER2_MAG = [1.5 * X, 2 * X]

# Training shapes = {constant, ramp}; Tier-1 = UNSEEN shapes {step, pulse} at
# in-range magnitude; Tier-2 = constant at out-of-range magnitude TIER2_MAG.
SPLITS = {
    "main": {
        "train":      dict(shapes=["constant", "ramp"], mag=[0.0, X],         seeds=list(range(1000, 1100))),  # 100
        "val":        dict(shapes=["constant", "ramp"], mag=[0.0, X],         seeds=list(range(1100, 1120))),  # 20
        "test_tier1": dict(shapes=["step", "pulse"],    mag=[0.0, X],         seeds=list(range(1200, 1220))),  # 20
        "test_tier2": dict(shapes=TIER2_SHAPES,       mag=TIER2_MAG,          seeds=list(range(1300, 1320))),  # 20
    },
    "calibration": {
        "train":      dict(shapes=["constant", "ramp"], mag=[0.0, X],         seeds=list(range(3000, 3040))),  # 40
        "val":        dict(shapes=["constant", "ramp"], mag=[0.0, X],         seeds=list(range(3100, 3110))),  # 10
    },
}

# ------------------------------------------------------------------ definitions
FORCING_DEFINITIONS = """\
Forcing schedules F_a(t), t in sols over the recorded window [0, T] with
T = record_years * year_sols.  A = stratified magnitude (W/m^2); timing
parameters are drawn deterministically from the trajectory seed.

  constant : F_a(t) = A                                   (for all t)
  ramp     : F_a(t) = A * min(t / T, 1)                   (linear 0 -> A over T)
  step     : F_a(t) = A if t >= t_step else 0 ;
             t_step = U(0.3, 0.6) * T                      (seed-drawn)
  pulse    : F_a(t) = A if t0 <= t <= t1 else 0 ;
             t0 = U(0.20, 0.40) * T, t1 = U(0.60, 0.80) * T (seed-drawn)
"""


def make_schedule(shape, A, total_sol, rng):
    """Return (F_a callable of t_sol, description dict). See FORCING_DEFINITIONS."""
    if shape == "constant":
        return (lambda t: A), {"shape": shape, "magnitude": A}
    if shape == "ramp":
        return (lambda t: A * min(t / total_sol, 1.0)), \
               {"shape": shape, "magnitude": A, "form": "A*min(t/T,1)"}
    if shape == "step":
        ts = float(rng.uniform(0.3, 0.6) * total_sol)
        return (lambda t, ts=ts: (A if t >= ts else 0.0)), \
               {"shape": shape, "magnitude": A, "t_step_sol": ts}
    if shape == "pulse":
        t0 = float(rng.uniform(0.20, 0.40) * total_sol)
        t1 = float(rng.uniform(0.60, 0.80) * total_sol)
        return (lambda t, t0=t0, t1=t1: (A if (t0 <= t <= t1) else 0.0)), \
               {"shape": shape, "magnitude": A, "pulse_window_sol": [t0, t1]}
    raise ValueError(f"unknown shape {shape}")


def stratified_magnitude(seed, rank, total, lo, hi):
    """Deterministic magnitude placing trajectory `rank` of `total` (same shape)
    into its own stratum of [lo, hi], with seed-drawn jitter inside the stratum."""
    if total <= 0:
        return lo
    u = float(np.random.default_rng((int(seed), 0xA1)).uniform(0.0, 1.0))
    return lo + (rank + u) / total * (hi - lo)


def expected_n_time(cfg, rec_years):
    dt = cfg["time"]["dt_sol"]
    total = rec_years * cfg["planet"]["year_sols"]
    nsteps = int(round(total / dt))
    out_every = max(1, int(round(cfg["time"]["output_every_sol"] / dt)))
    return nsteps // out_every + 1


def build_tasks(cfg, out_root, rec_years, splits=SPLITS, spin_years=SPIN_YEARS):
    """Deterministic task list with stratified magnitudes (no simulation)."""
    tasks = []
    for family, fam_splits in splits.items():
        for split, spec in fam_splits.items():
            seeds, shapes = spec["seeds"], spec["shapes"]
            lo, hi = spec["mag"]
            shape_of = [shapes[i % len(shapes)] for i in range(len(seeds))]
            total = Counter(shape_of)
            rank = defaultdict(int)
            for i, seed in enumerate(seeds):
                sh = shape_of[i]
                r = rank[sh]; rank[sh] += 1
                A = stratified_magnitude(seed, r, total[sh], lo, hi)
                fpath = os.path.join(out_root, family, split, f"traj_{seed:04d}.npz")
                tasks.append({
                    "family": family, "split": split, "seed": int(seed),
                    "shape": sh, "magnitude": A, "mag_range": [lo, hi],
                    "file": fpath, "rel": os.path.relpath(fpath, out_root),
                    "rec_years": rec_years, "spin_years": spin_years,
                })
    return tasks


def _generate_arrays(cfg, task):
    seed, shape, A, rec_years = (task["seed"], task["shape"], task["magnitude"],
                                 task["rec_years"])
    total_sol = rec_years * cfg["planet"]["year_sols"]
    rng = np.random.default_rng(seed)
    fa_func, _ = make_schedule(shape, A, total_sol, rng)

    spin = run_truth(cfg, F_a=0.0, seed=seed, years=task.get("spin_years", SPIN_YEARS),
                     mismatch_scale=SCALE)
    init = dict(T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                m_init=spin["m_final"])
    forced = run_truth(cfg, F_a=fa_func, seed=seed, years=rec_years,
                       mismatch_scale=SCALE, **init)
    control = run_truth(cfg, F_a=0.0, seed=seed, years=rec_years,
                        mismatch_scale=SCALE, **init)
    n = min(len(forced["T_s"]), len(control["T_s"]))
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    arrays = {
        # observable (model input)
        "t_sol": f32(forced["t_sol"][:n]), "Ls": f32(forced["Ls"][:n]),
        "lat": f32(forced["lat"]), "x": f32(forced["x"]), "w": f32(forced["w"]),
        "T_s": f32(forced["T_s"][:n]), "S": f32(forced["S"][:n]),
        "F_a": f32(forced["F_a"][:n]),
        # eval labels (NOT model input; separate loader)
        EVAL + "control_T_s": f32(control["T_s"][:n]),
        EVAL + "response_T_s": f32(forced["T_s"][:n] - control["T_s"][:n]),
        # hidden diagnostics (NOT model input)
        HIDDEN + "m": f32(forced["m"][:n]), HIDDEN + "T_g": f32(forced["T_g"][:n]),
        HIDDEN + "tau": f32(forced["tau"][:n]),
        HIDDEN + "absorbed": f32(forced["absorbed"][:n]),
        HIDDEN + "emitted": f32(forced["emitted"][:n]),
        HIDDEN + "diffusion": f32(forced["diffusion"][:n]),
        HIDDEN + "coupling": f32(forced["coupling"][:n]),
        HIDDEN + "control_m": f32(control["m"][:n]),
    }
    return arrays, n


def _worker(task):
    """Generate one trajectory (resumable, atomic). Returns status dict.
    Top-level so it is picklable under the macOS 'spawn' start method."""
    cfg = load_config()
    fpath = task["file"]
    if os.path.exists(fpath):
        return {"id": f"{task['family']}/{task['split']}/{task['seed']:04d}",
                "status": "skip", "n_time": expected_n_time(cfg, task["rec_years"]),
                "task": task}
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    arrays, n = _generate_arrays(cfg, task)
    tmp = fpath[:-4] + ".tmp.npz"            # atomic: write temp, then rename
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, fpath)
    return {"id": f"{task['family']}/{task['split']}/{task['seed']:04d}",
            "status": "done", "n_time": int(n), "task": task}


def _setup_logging(out_root):
    os.makedirs(out_root, exist_ok=True)
    logger = logging.getLogger("gen_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(out_root, "gen.log"))
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


def _smoke_splits():
    return {"main": {"train": dict(shapes=["constant", "ramp"], mag=[0.0, X],
                                   seeds=[1000, 1001])}}


def write_manifest(out_root, cfg, tasks, results, rec_years, wall,
                   spin_years=SPIN_YEARS):
    by_id = {r["id"]: r for r in results}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_design": {"D": 0.10, "n": 64, "dt_sol": 0.05,
                          "mismatch_scale": SCALE, "X": X,
                          "spin_years": spin_years, "record_years": rec_years},
        "seed_ranges": SEED_RANGES,
        "observable_keys": ["t_sol", "Ls", "lat", "x", "w", "T_s", "S", "F_a"],
        "eval_key_prefix": EVAL, "hidden_key_prefix": HIDDEN,
        "forcing_definitions": FORCING_DEFINITIONS,
        "note_held_out": "seeds 9000-9999 reserved and intentionally NOT generated",
        "wall_clock_sec": round(wall, 1),
        "splits": {}, "trajectories": [],
    }
    for t in tasks:
        tid = f"{t['family']}/{t['split']}/{t['seed']:04d}"
        n_time = by_id.get(tid, {}).get("n_time")
        manifest["trajectories"].append({
            "id": tid, "family": t["family"], "split": t["split"],
            "seed": t["seed"], "shape": t["shape"], "magnitude": t["magnitude"],
            "forcing_range": t["mag_range"], "file": t["rel"], "n_time": n_time})
        key = f"{t['family']}/{t['split']}"
        sp = manifest["splits"].setdefault(
            key, {"mag_range": t["mag_range"], "shapes": [], "seeds": [],
                  "files": [], "count": 0})
        if t["shape"] not in sp["shapes"]:
            sp["shapes"].append(t["shape"])
        sp["seeds"].append(t["seed"]); sp["files"].append(t["rel"])
        sp["count"] += 1
    with open(os.path.join(out_root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny scratch run: 2 short trajectories")
    ap.add_argument("--rec-years", type=int, default=None)
    ap.add_argument("--tier2-mag", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help=f"Tier-2 magnitude range (default {TIER2_MAG})")
    ap.add_argument("--print-defs", action="store_true")
    args = ap.parse_args(argv)

    if args.print_defs:
        print(FORCING_DEFINITIONS)
        return

    cfg = load_config()
    assert cfg["time"]["dt_sol"] == 0.05 and cfg["physics"]["D"] == 0.10 \
        and cfg["grid"]["n"] == 64, "frozen design parameters changed!"

    splits = _smoke_splits() if args.smoke else SPLITS
    if args.tier2_mag is not None and not args.smoke:
        import copy
        splits = copy.deepcopy(SPLITS)
        splits["main"]["test_tier2"]["mag"] = [float(v) for v in args.tier2_mag]
    rec_years = args.rec_years if args.rec_years is not None \
        else (1 if args.smoke else RECORD_YEARS)
    out_root = args.out

    run_generation(out_root, splits, rec_years, args.workers, smoke=args.smoke)


def run_generation(out_root, splits, rec_years, workers, *, spin_years=SPIN_YEARS,
                   smoke=False):
    """Generate `splits` into `out_root` and write the manifest.

    workers=0 runs in-process (used by the integration test on tiny scratch
    splits); otherwise a spawn ProcessPoolExecutor.  Returns the manifest."""
    cfg = load_config()
    logger = _setup_logging(out_root)
    # clean up any stale temp files left by a previous hard interrupt
    import glob
    for stale in glob.glob(os.path.join(out_root, "**", "*.tmp.npz"), recursive=True):
        try:
            os.remove(stale)
            logger.info("removed stale temp %s", stale)
        except OSError:
            pass
    tasks = build_tasks(cfg, out_root, rec_years, splits=splits,
                        spin_years=spin_years)
    logger.info("start: %d tasks, %d workers, out=%s, rec_years=%s, smoke=%s",
                len(tasks), workers, out_root, rec_years, smoke)

    t0 = time.time()
    results = []
    try:
        if workers == 0:
            for t in tasks:
                r = _worker(t)
                results.append(r)
                logger.info("%-6s %s (n_time=%s)", r["status"], r["id"], r["n_time"])
        else:
            ctx = mp.get_context("spawn")       # explicit; macOS default anyway
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                futs = {ex.submit(_worker, t): t for t in tasks}
                for fut in as_completed(futs):
                    r = fut.result()
                    results.append(r)
                    logger.info("%-6s %s (n_time=%s)", r["status"], r["id"], r["n_time"])
    except KeyboardInterrupt:
        logger.warning("interrupted: %d/%d done; re-run to resume (atomic writes "
                       "mean no partial files).", len(results), len(tasks))
        return None

    wall = time.time() - t0
    # manifest reflects the full intended task list; n_time filled where known
    man = write_manifest(out_root, cfg, tasks, results, rec_years, wall,
                         spin_years=spin_years)
    counts = {k: v["count"] for k, v in man["splits"].items()}
    logger.info("counts per split: %s", counts)
    logger.info("total trajectories: %d; wall-clock: %.1fs; manifest written.",
                sum(counts.values()), wall)
    for h in list(logger.handlers):           # release gen.log file handle
        h.close(); logger.removeHandler(h)
    return man

if __name__ == "__main__":
    main()

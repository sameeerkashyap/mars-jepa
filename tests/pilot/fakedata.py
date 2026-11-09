"""Tiny FAKE dataset with the SAME shapes/keys as the real Checkpoint-B data.

Used until the owning session finishes regenerating data/.  Mirrors the current
dataio.py format: observable keys plus 'hidden__' and 'eval__' prefixed groups,
and a manifest carrying a 'trajectories' list (what dataio.split_entries reads).
"""

from __future__ import annotations

import json
import os

import numpy as np

HIDDEN = "hidden__"
EVAL = "eval__"
NCELL = 64

# (family, split, seeds, magnitude range)
LAYOUT = [
    ("main", "train", list(range(1000, 1012)), (0.0, 5.0)),
    ("main", "val", [1100, 1101, 1102], (0.0, 5.0)),
    ("main", "test_tier1", [1200, 1201], (0.0, 5.0)),
    ("main", "test_tier2", [1300, 1301], (7.5, 10.0)),
    ("calibration", "train", [3000, 3001, 3002, 3003], (0.0, 5.0)),
    ("calibration", "val", [3100, 3101], (0.0, 5.0)),
]


def _traj(seed, mag_range, n_time):
    rng = np.random.default_rng(seed)
    x = np.linspace(-0.98, 0.98, NCELL).astype(np.float32)
    lat = np.arcsin(x).astype(np.float32)
    w = np.full(NCELL, 1.0 / NCELL, dtype=np.float32)
    t_sol = np.arange(n_time, dtype=np.float32)
    Ls = ((t_sol / 668.6) * 360.0 % 360.0).astype(np.float32)
    # temperature: cold poles (near frost ~148K), warm equator, seasonal wobble
    base = 150.0 + 70.0 * np.cos(lat)[None, :]
    season = 15.0 * np.sin(2 * np.pi * t_sol / 668.6)[:, None] * np.sign(x)[None, :]
    T_s = (base + season + rng.standard_normal((n_time, NCELL))).astype(np.float32)
    T_s = np.maximum(T_s, 147.7).astype(np.float32)
    S = np.maximum(0.0, 400.0 * np.cos(lat)[None, :]
                   + 50.0 * np.sin(2 * np.pi * t_sol / 668.6)[:, None]
                   ).astype(np.float32) * np.ones((n_time, 1), np.float32)
    A = float(rng.uniform(*mag_range))
    F_a = np.full(n_time, A, dtype=np.float32)
    control_T_s = (T_s - 0.3 * A).astype(np.float32)
    response_T_s = (T_s - control_T_s).astype(np.float32)
    arrays = {
        "t_sol": t_sol, "Ls": Ls, "lat": lat, "x": x, "w": w,
        "T_s": T_s, "S": S, "F_a": F_a,
        EVAL + "control_T_s": control_T_s,
        EVAL + "response_T_s": response_T_s,
        HIDDEN + "m": rng.standard_normal((n_time, NCELL)).astype(np.float32),
        HIDDEN + "T_g": rng.standard_normal((n_time, NCELL)).astype(np.float32),
        HIDDEN + "tau": rng.standard_normal((n_time, NCELL)).astype(np.float32),
    }
    return arrays, A


def build_fake_dataset(root, n_time=240):
    os.makedirs(root, exist_ok=True)
    manifest = {"created_utc": "fake", "splits": {}, "trajectories": [],
                "observable_keys": ["t_sol", "Ls", "lat", "x", "w", "T_s", "S",
                                    "F_a"],
                "hidden_key_prefix": HIDDEN, "eval_key_prefix": EVAL}
    for family, split, seeds, mag in LAYOUT:
        outdir = os.path.join(root, family, split)
        os.makedirs(outdir, exist_ok=True)
        files = []
        for i, s in enumerate(seeds):
            arrays, A = _traj(s, mag, n_time)
            rel = os.path.join(family, split, f"traj_{s:04d}.npz")
            np.savez_compressed(os.path.join(root, rel), **arrays)
            files.append(rel)
            manifest["trajectories"].append({
                "id": f"{family}/{split}/{s:04d}", "family": family,
                "split": split, "seed": int(s), "shape": "constant",
                "magnitude": A, "file": rel, "n_time": n_time})
        manifest["splits"][f"{family}/{split}"] = {
            "seeds": [int(s) for s in seeds], "count": len(files),
            "files": files, "mag_range": list(mag)}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return root


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "data_fake"
    build_fake_dataset(out)
    print(f"wrote fake dataset -> {out}")

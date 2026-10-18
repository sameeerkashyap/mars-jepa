"""
Tests for the Checkpoint-B data loader guards (dataio.py).

A tiny fake dataset is fabricated (no simulator needed) so the test is fast.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio
from dataio import (load_trajectory, load_eval_labels, load_hidden, load_split,
                    HIDDEN, EVAL)


def _write_traj(path, seed, magnitude):
    rng = np.random.default_rng(seed)
    arrays = {
        "t_sol": np.arange(5, dtype=np.float32),
        "T_s": rng.standard_normal((5, 4)).astype(np.float32),
        "S": rng.standard_normal((5, 4)).astype(np.float32),
        "F_a": rng.standard_normal(5).astype(np.float32),
        EVAL + "control_T_s": rng.standard_normal((5, 4)).astype(np.float32),
        EVAL + "response_T_s": rng.standard_normal((5, 4)).astype(np.float32),
        HIDDEN + "m": rng.standard_normal((5, 4)).astype(np.float32),
        HIDDEN + "T_g": rng.standard_normal((5, 4)).astype(np.float32),
        HIDDEN + "tau": rng.standard_normal((5, 4)).astype(np.float32),
    }
    np.savez_compressed(path, **arrays)


@pytest.fixture()
def fake_data(tmp_path):
    root = str(tmp_path / "data")
    splits = {
        "main/train": ("main", "train", [1000, 1001]),
        "main/test_tier1": ("main", "test_tier1", [1200]),
        "main/test_tier2": ("main", "test_tier2", [1300]),
        "calibration/train": ("calibration", "train", [3000]),
    }
    man = {"splits": {}, "trajectories": []}
    for key, (family, split, seeds) in splits.items():
        outdir = os.path.join(root, family, split)
        os.makedirs(outdir, exist_ok=True)
        files = []
        for i, s in enumerate(seeds):
            fp = os.path.join(outdir, f"traj_{s:04d}.npz")
            _write_traj(fp, s, magnitude=float(i))
            rel = os.path.relpath(fp, root)
            files.append(rel)
            man["trajectories"].append({"id": f"{key}/{s:04d}", "family": family,
                                        "split": split, "seed": s,
                                        "magnitude": float(i), "file": rel})
        man["splits"][key] = {"seeds": seeds, "count": len(files), "files": files}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump(man, fh)
    return root


def test_default_loader_hides_hidden_and_eval(fake_data):
    obs = load_trajectory(dataio.list_split_files(fake_data, "main", "train")[0])
    assert not any(k.startswith((HIDDEN, EVAL)) for k in obs)
    for forbidden in ("m", "T_g", "tau", "control_T_s", "response_T_s"):
        assert forbidden not in obs
    assert {"T_s", "S", "F_a"} <= set(obs)


def test_eval_labels_separate_loader(fake_data):
    f = dataio.list_split_files(fake_data, "main", "test_tier2")[0]
    labels = load_eval_labels(f)
    assert set(labels) == {"control_T_s", "response_T_s"}
    with pytest.raises(RuntimeError):
        load_eval_labels(f, calibration_mode=True)


def test_hidden_only_via_explicit_loader(fake_data):
    f = dataio.list_split_files(fake_data, "main", "train")[0]
    assert set(load_hidden(f)) >= {"m", "T_g", "tau"}


def test_calibration_cannot_load_test(fake_data):
    with pytest.raises(RuntimeError):
        load_split(fake_data, "main", "test_tier1", calibration_mode=True)
    with pytest.raises(RuntimeError):
        load_split(fake_data, "main", "test_tier2", calibration_mode=True)
    with pytest.raises(KeyError):
        load_split(fake_data, "calibration", "test_tier1")


def test_test_load_is_logged(fake_data):
    logpath = os.path.join(fake_data, "test_access.log")
    assert not os.path.exists(logpath)
    load_split(fake_data, "main", "test_tier1")
    load_split(fake_data, "main", "test_tier2")
    lines = open(logpath).read().strip().splitlines()
    assert len(lines) == 2
    from datetime import datetime
    for ln in lines:
        ts, tag, which = ln.split("\t")
        assert tag == "TEST-LOAD" and which.startswith("main/test_tier")
        datetime.fromisoformat(ts)


def test_non_test_split_not_logged(fake_data):
    load_split(fake_data, "main", "train")
    load_split(fake_data, "calibration", "train")
    assert not os.path.exists(os.path.join(fake_data, "test_access.log"))

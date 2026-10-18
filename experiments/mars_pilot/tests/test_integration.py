"""
End-to-end integration test on REAL code paths (no fake fixtures):

  gen_data.run_generation (truth simulator, 4 tiny trajectories, scratch dir)
  -> dataio loaders + guards
  -> calibrate.py (CLI)
  -> train.py (CLI), one run per arm (+ physics A/B for gru)
  -> eval.py (CLI) on Tier 1 + Tier 2, single runs and a GRU ensemble.

Trajectories are short (spin 0.25 yr, record 0.3 yr ~ 201 stored steps, enough
for one K+H=160 window) and use dedicated seeds 8500-8503, outside every dataset
family range.  Subprocess output is captured, never printed (it contains
scratch test-split metrics).
"""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dataio
import gen_data

SPIN_YEARS = 0.25
REC_YEARS = 0.3
SPLITS = {"main": {
    "train":      dict(shapes=["constant"], mag=[0.0, gen_data.X], seeds=[8500]),
    "val":        dict(shapes=["ramp"],     mag=[0.0, gen_data.X], seeds=[8501]),
    "test_tier1": dict(shapes=["step"],     mag=[0.0, gen_data.X], seeds=[8502]),
    "test_tier2": dict(shapes=gen_data.TIER2_SHAPES, mag=gen_data.TIER2_MAG,
                       seeds=[8503]),
}}
ARMS = ["gru", "rssm", "transformer", "hybrid", "linear", "ebm_only"]
SMALL = ["--epochs", "2", "--max-batches", "1", "--device", "cpu"]


def _run(args, timeout=600):
    p = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True,
                       text=True, timeout=timeout)
    assert p.returncode == 0, (f"{args[0]} failed (rc={p.returncode}):\n"
                               + p.stderr[-2000:])
    return p


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("integration_data"))
    man = gen_data.run_generation(root, SPLITS, REC_YEARS, workers=0,
                                  spin_years=SPIN_YEARS)
    assert man is not None
    return root


def test_generation_manifest(scratch):
    man = json.load(open(os.path.join(scratch, "manifest.json")))
    assert len(man["trajectories"]) == 4
    assert man["splits"]["main/test_tier2"]["shapes"] == ["constant"]
    assert all(t["n_time"] and t["n_time"] >= 160 for t in man["trajectories"])


def test_loaders_on_generated_data(scratch):
    log = os.path.join(scratch, "test_access.log")
    for split in SPLITS["main"]:
        trajs = dataio.load_split(scratch, "main", split)
        assert len(trajs) == 1
        keys = set(trajs[0])
        assert {"T_s", "S", "F_a", "lat", "w"} <= keys
        assert not any(k.startswith(("hidden__", "eval__")) for k in keys)
        assert not keys & {"m", "T_g", "tau", "control_T_s", "response_T_s"}
    labs = dataio.load_split_eval_labels(scratch, "main", "test_tier2")
    assert set(labs[0]) == {"control_T_s", "response_T_s"}
    with pytest.raises(RuntimeError):
        dataio.load_split(scratch, "main", "test_tier1", calibration_mode=True)
    with pytest.raises(RuntimeError):
        dataio.load_split_eval_labels(scratch, "main", "val", calibration_mode=True)
    lines = open(log).read().splitlines()
    assert len(lines) == 3 and all("\tTEST-LOAD\tmain/test_tier" in l for l in lines)


def test_calibrate_cli(scratch):
    out = os.path.join(scratch, "cal.json")
    _run(["calibrate.py", "--family", "main", "--arms", *ARMS, "--seeds", "0",
          "--data-dir", scratch, "--out", out, *SMALL])
    rep = json.load(open(out))
    for arm in ARMS:
        r = rep["arms"][arm]
        assert math.isfinite(r["val_rmse_K_mean"])
        if arm != "ebm_only":
            assert len(r["val_loss_curves"][0]) == 2
            assert isinstance(r["still_decreasing_at_final_epoch"], bool)
    assert os.path.exists(rep["curves_png"])


@pytest.fixture(scope="module")
def runs(scratch):
    configs = [(a, "off", 0) for a in ARMS] + [("gru", "A", 0), ("gru", "B", 0),
                                               ("gru", "off", 1)]
    out = {}
    for arm, phys, seed in configs:
        rd = os.path.join(scratch, "runs", f"{arm}_{phys}_s{seed}")
        _run(["train.py", "--arm", arm, "--physics", phys, "--fraction", "0.1",
              "--seed", str(seed), "--data-dir", scratch, "--out", rd, *SMALL])
        for f in ("model.pt", "stats.json", "info.json", "arch.json"):
            assert os.path.exists(os.path.join(rd, f))
        out[(arm, phys, seed)] = rd
    return out


def _check_eval(path):
    ev = json.load(open(path))
    assert set(ev["tiers"]) == {"tier1", "tier2"}
    for tier in ev["tiers"].values():
        assert tier["n_windows"] >= 1
        ir = tier["intervention_response_rmse"]
        assert all(math.isfinite(ir[k]) for k in ("polar_N", "polar_S",
                                                   "global_mean"))
        assert np.all(np.isfinite(tier["rollout_rmse_vs_horizon"]))
        assert math.isfinite(tier["energy_budget_violation_mean"])
        assert tier["energy_budget_violation_label"].startswith(
            "EBM-residual constraint satisfaction")
    return ev


def test_eval_cli_each_arm(scratch, runs):
    for key, rd in runs.items():
        out = os.path.join(rd, "eval.json")
        _run(["eval.py", "--run", rd, "--data-dir", scratch, "--device", "cpu",
              "--quiet", "--out", out])
        ev = _check_eval(out)
        if key[0] == "rssm":
            assert "calibration" in ev["tiers"]["tier2"]


def test_eval_cli_ensemble(scratch, runs):
    out = os.path.join(scratch, "ens_eval.json")
    _run(["eval.py", "--run", runs[("gru", "off", 0)], "--run",
          runs[("gru", "off", 1)], "--ensemble", "--data-dir", scratch,
          "--device", "cpu", "--quiet", "--out", out])
    ev = _check_eval(out)
    assert "calibration" in ev["tiers"]["tier2"]

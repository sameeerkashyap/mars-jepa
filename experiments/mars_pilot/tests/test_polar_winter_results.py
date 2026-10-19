"""Polar-winter result columns end to end: eval.evaluate() window/cell counts,
the *_pinned window rule, identical scoring protocol for every arm (LINEAR and
EBM-only included), the results.csv columns written by sweep.py, and the
descriptive primary-vs-baselines table in analyze.py.

Synthetic fixtures only: the split loaders are monkeypatched, no data/ access."""

import copy
import csv
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import analyze
import eval as evalmod
import metric_defs as md
import sweep
import train
from common import load_config, frost_point
from models.base import build_model

NCELL, NT, PERIOD = 64, 700, 668.6
PIN_UNTIL = 160                    # control pinned only before this step
RESP_PINNED, RESP_OTHER = 2.0, 9.0
STATS = {"T_s_mean": 200.0, "T_s_std": 30.0, "S_mean": 100.0, "S_std": 80.0,
         "F_a_mean": 2.0, "F_a_std": 2.0}


@pytest.fixture(scope="module")
def cfgs():
    p2 = copy.deepcopy(train.load_phase2_config(
        os.path.join(ROOT, "config_phase2.yaml")))
    full = load_config(os.path.join(ROOT, "config.yaml"))
    T_frost = frost_point(full["co2"]["surface_pressure_pa"], full["co2"])
    return p2, full, T_frost


def _fixture(T_frost):
    """Seasonal polar night; half the S band control-pinned for t < PIN_UNTIL."""
    x = np.linspace(-0.98, 0.98, NCELL)
    lat = np.arcsin(x)
    t = np.arange(NT)
    season = np.sin(2 * np.pi * t / PERIOD)[:, None]
    S = np.maximum(0.0, 300 * np.cos(lat)[None, :]
                   + 300 * season * np.sin(lat)[None, :])
    bands = md.polar_bands(lat)
    pinned_cells = np.where(bands["S"])[0][::2]
    ctrl = np.full((NT, NCELL), 220.0)
    ctrl[:PIN_UNTIL, pinned_cells] = T_frost + 0.2
    pinned = ctrl <= T_frost + evalmod.PIN_TOL_K
    resp = np.where(pinned, RESP_PINNED, RESP_OTHER)
    traj = {"T_s": ctrl + resp, "S": S, "F_a": np.full(NT, 8.0), "lat": lat,
            "w": np.full(NCELL, 1.0 / NCELL)}
    lab = {"control_T_s": ctrl, "response_T_s": resp}
    return traj, lab, bands, pinned


def _expected_counts(traj, bands, pinned, K, H):
    """Independent of eval.py: windows / cell-steps per polar metric."""
    out = {}
    for h, band in bands.items():
        w = md.winter_steps(traj["S"], band)        # s_ref = full-record max
        for key, extra in ((f"polar_winter_{h}", True),
                           (f"polar_winter_{h}_pinned", pinned)):
            m = w[:, None] & band[None, :] & extra
            per_win = [int(m[s + K:s + K + H].sum())
                       for s in range(0, NT - K - H + 1, H)]
            out[key] = {"n_windows": sum(n > 0 for n in per_win),
                        "n_cell_steps": sum(per_win), "per_window": per_win}
    return out


class _ZeroResponse(torch.nn.Module):
    """Same rollout for forced and control -> predicted response 0."""
    stochastic = False

    def rollout(self, Ts_ctx, S, Fa, lat, sample=False):
        H = S.shape[1] - Ts_ctx.shape[1]
        return torch.full((Ts_ctx.shape[0], H, NCELL), 200.0), {}


def _patch_data(monkeypatch, traj, lab):
    monkeypatch.setattr(evalmod, "load_split", lambda *a, **k: [traj])
    monkeypatch.setattr(evalmod, "load_split_eval_labels", lambda *a, **k: [lab])


def test_polar_keys_renamed():
    assert evalmod.POLAR_KEYS == ("polar_winter_N", "polar_winter_S",
                                  "polar_winter_N_pinned",
                                  "polar_winter_S_pinned")
    for f in ("eval.py", "sweep.py", "analyze.py"):
        src = open(os.path.join(ROOT, f)).read()
        for old in ('"polar_N', '"polar_S', "resp_N", "resp_S"):
            assert old not in src, f"{f} still uses {old}"


def test_pinned_window_needs_a_pinned_winter_cell(cfgs, monkeypatch):
    p2, full, T_frost = cfgs
    traj, lab, bands, pinned = _fixture(T_frost)
    K, H = p2["protocol"]["K"], p2["protocol"]["H"]
    exp = _expected_counts(traj, bands, pinned, K, H)
    pw, ww = exp["polar_winter_S_pinned"], exp["polar_winter_S"]["per_window"]
    # fixture must contain: a winter window WITH pinned cells, and one WITHOUT
    assert any(p > 0 for p in pw["per_window"])
    assert any(w > 0 and p == 0 for w, p in zip(ww, pw["per_window"]))
    _patch_data(monkeypatch, traj, lab)
    monkeypatch.setattr(evalmod, "load_model", lambda rd, *a: (
        _ZeroResponse(), {"arm": "stub"}, STATS, p2["protocol"]))
    m = evalmod.evaluate(["stub"], "unused", "main", "test_tier2", p2, full,
                         "cpu")
    cnt, ir = m["intervention_response_counts"], m["intervention_response_rmse"]
    for k in evalmod.POLAR_KEYS:
        assert cnt[k]["n_windows"] == exp[k]["n_windows"], k
        assert cnt[k]["n_cell_steps"] == exp[k]["n_cell_steps"], k
    # error = -resp: pinned cells only -> exactly RESP_PINNED in every window
    assert np.isclose(ir["polar_winter_S_pinned"], RESP_PINNED)
    assert ir["polar_winter_S"] > RESP_PINNED          # includes unpinned cells
    assert cnt["polar_winter_N_pinned"]["n_windows"] == 0
    assert np.isnan(ir["polar_winter_N_pinned"])


@pytest.mark.parametrize("arm", ["linear", "ebm_only", "gru", "hybrid"])
def test_every_arm_scored_with_same_protocol(arm, cfgs, monkeypatch, tmp_path):
    """Baselines (LINEAR, EBM-only) and learned arms go through the same
    evaluate() path: same K/H/stride, windows, masks and counts."""
    p2, full, T_frost = cfgs
    traj, lab, bands, pinned = _fixture(T_frost)
    K, H = p2["protocol"]["K"], p2["protocol"]["H"]
    exp = _expected_counts(traj, bands, pinned, K, H)
    rd = str(tmp_path / arm)
    model = build_model(arm, STATS, NCELL, p2["arch"], seed=0)
    train.save_run(rd, model, STATS, {"arm": arm, "seed": 0}, p2)
    _patch_data(monkeypatch, traj, lab)
    m = evalmod.evaluate([rd], "unused", "main", "test_tier2", p2, full, "cpu")
    assert m["protocol"] == {"K": K, "H": H, "stride": H}
    assert m["n_windows"] == len(range(0, NT - K - H + 1, H))
    for k in evalmod.POLAR_KEYS:
        c = m["intervention_response_counts"][k]
        assert (c["n_windows"], c["n_cell_steps"]) == (
            exp[k]["n_windows"], exp[k]["n_cell_steps"]), (arm, k)
        assert np.isfinite(m["intervention_response_rmse"][k]) == (
            exp[k]["n_windows"] > 0)


def _tier(v):
    return {"intervention_response_rmse": {k: v for k in evalmod.POLAR_KEYS},
            "intervention_response_counts": {
                k: {"n_windows": 3, "n_cell_steps": 40 + i}
                for i, k in enumerate(evalmod.POLAR_KEYS)}}


def test_results_csv_records_counts_per_metric(tmp_path):
    for t in ("t1", "t2"):
        for k in evalmod.POLAR_KEYS:
            for suf in ("", "_n_windows", "_n_cell_steps"):
                assert f"{t}_{k}{suf}" in sweep.FIELDS
    cols = sweep.polar_columns("t2", _tier(1.5))
    assert cols["t2_polar_winter_S_pinned"] == 1.5
    assert cols["t2_polar_winter_S_pinned_n_windows"] == 3
    assert cols["t2_polar_winter_S_pinned_n_cell_steps"] == 43
    row = {k: "" for k in sweep.FIELDS}
    row.update(cols, tag="x", status="ok")
    out = str(tmp_path / "r.csv")
    sweep._append_row(out, row)
    got = next(csv.DictReader(open(out)))
    assert got["t2_polar_winter_S_pinned_n_cell_steps"] == "43"
    # an older results.csv (different columns) is refused, not appended to
    old = tmp_path / "old.csv"
    old.write_text("tag,t2_resp_S,status\n")
    with pytest.raises(RuntimeError):
        sweep._append_row(str(old), row)


def test_primary_vs_baselines_table(tmp_path):
    rows = []
    vals = {"gru/off": [1.0, 3.0], "linear/off": [4.0, 6.0],
            "hybrid/off": [2.0, np.nan], "gru/A": [0.5, 0.7]}
    for frac in (0.1, 1.0):
        for cfg, vs in vals.items():
            arm, phys = cfg.split("/")
            for sd, v in enumerate(vs):
                rows.append(dict(tag=f"{arm}_{phys}_{frac}_{sd}", arm=arm,
                                 physics=phys, fraction=frac, seed=sd,
                                 status="ok", t2_polar_winter_S_pinned=v * frac,
                                 t2_polar_winter_S=9.0))
    p = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    df = analyze.load_results(str(p))
    tab = analyze.primary_vs_baselines(df)
    assert tab["metric"] == "t2_polar_winter_S_pinned"
    assert len(tab["rows"]) == 8
    r = next(r for r in tab["rows"]
             if r["config"] == "gru/off" and r["fraction"] == 1.0)
    assert r["arm"]["mean"] == 2.0 and r["arm"]["per_seed"] == {0: 1.0, 1: 3.0}
    assert r["baselines"]["linear/off"]["mean"] == 5.0
    hy = r["baselines"]["hybrid/off"]
    assert hy["mean"] == 2.0 and hy["n_seeds_finite"] == 1
    r01 = next(r for r in tab["rows"]
               if r["config"] == "gru/off" and r["fraction"] == 0.1)
    assert np.isclose(r01["baselines"]["linear/off"]["mean"], 0.5)
    # descriptive only: no verdict / threshold fields anywhere
    flat = repr(tab).lower()
    assert "verdict" not in flat and "threshold" not in flat
    mdp = tmp_path / "a.md"
    analyze.write_report({}, {}, out_json=str(tmp_path / "a.json"),
                         out_md=str(mdp), primary=tab)
    text = mdp.read_text()
    assert "Primary metric vs baselines" in text and "s1=3" in text
    assert "linear/off mean" in text and "hybrid/off per seed" in text

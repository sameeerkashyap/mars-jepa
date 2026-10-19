"""eval.py polar-winter metric: winter-restricted, metric_defs masks, s_ref from
the full trajectory, control-pinned variant, and label isolation.

Synthetic fixtures only (no data/ access)."""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import eval as evalmod
import metric_defs as md

NCELL, NT, PERIOD = 64, 700, 668.6
T_FROST = 148.0


def _fixture():
    """Seasonal polar night in each hemisphere (S -> 0 at the winter pole)."""
    x = np.linspace(-0.98, 0.98, NCELL)
    lat = np.arcsin(x)
    t = np.arange(NT)
    season = np.sin(2 * np.pi * t / PERIOD)[:, None]
    S = np.maximum(0.0, 300 * np.cos(lat)[None, :]
                   + 300 * season * np.sin(lat)[None, :])
    T_s = 200.0 + 0 * S
    ctrl = np.full_like(S, 220.0)
    traj = {"T_s": T_s, "S": S, "F_a": np.full(NT, 8.0), "lat": lat}
    lab = {"control_T_s": ctrl, "response_T_s": T_s - ctrl}
    return traj, lab, md.polar_bands(lat)


def _masks_for(traj, lab, bands, K, H, stride, ctrl=None):
    S_full = np.asarray(traj["S"])
    s_refs = {h: md.seasonal_ref(S_full, b) for h, b in bands.items()}
    if ctrl is not None:
        lab = dict(lab, control_T_s=ctrl)
    wins = evalmod._windows_with_control(traj, lab, K, H, stride)
    return wins, [evalmod._winter_masks(w["S"][K:], w["ctrl"][K:], bands,
                                        s_refs, T_FROST) for w in wins]


def test_eval_winter_mask_equals_metric_defs():
    traj, lab, bands = _fixture()
    K, H = 40, 120
    wins, masks = _masks_for(traj, lab, bands, K, H, H)
    assert len(wins) >= 4
    for h, band in bands.items():
        full = md.winter_steps(traj["S"], band)       # s_ref = full-record max
        assert full.any() and not full.all()
        for i, m in enumerate(masks):
            s = i * H
            assert np.array_equal(m[h][:, band][:, 0], full[s + K:s + K + H])
            assert not m[h][:, ~band].any()
    # at least one window has no winter steps in some hemisphere -> NaN
    err = np.ones((H, NCELL))
    vals = [evalmod._masked_rmse(err, m["N"])[0] for m in masks]
    assert any(np.isnan(v) for v in vals) and any(np.isfinite(v) for v in vals)
    assert evalmod._nanmean(vals) == 1.0


def test_s_ref_from_full_trajectory_not_window():
    traj, lab, bands = _fixture()
    K, H = 40, 120
    wins, masks = _masks_for(traj, lab, bands, K, H, H)
    band = bands["N"]
    # referencing the window's own max would give a different mask somewhere
    differs = [not np.array_equal(m["N"][:, band][:, 0],
                                  md.winter_steps(w["S"][K:], band))
               for w, m in zip(wins, masks)]
    assert any(differs)


def test_winter_only_differs_from_all_steps():
    traj, lab, bands = _fixture()
    K, H = 40, 300
    wins, masks = _masks_for(traj, lab, bands, K, H, H)
    band = bands["S"]
    for w, m in zip(wins, masks):
        win_t = m["S"][:, band][:, 0]
        if not (win_t.any() and (~win_t).any()):
            continue
        err = np.where(win_t[:, None], 0.1, 10.0) * np.ones((H, NCELL))
        winter_rmse, n = evalmod._masked_rmse(err, m["S"])
        all_steps = float(np.sqrt(np.mean(err[:, band] ** 2)))
        assert np.isclose(winter_rmse, 0.1) and all_steps > 1.0
        assert n == win_t.sum() * band.sum()
        return
    raise AssertionError("fixture has no mixed winter/summer window")


def test_pinned_metric_uses_only_control_pinned_cells():
    traj, lab, bands = _fixture()
    K, H = 40, 300
    ctrl = np.array(lab["control_T_s"])
    band = bands["S"]
    pinned_cells = np.where(band)[0][::2]               # half the S band
    ctrl[:, pinned_cells] = T_FROST + 0.2
    wins, masks = _masks_for(traj, lab, bands, K, H, H, ctrl=ctrl)
    used = 0
    for m in masks:
        mp = m["S_pinned"]
        assert not mp[:, np.setdiff1d(np.arange(NCELL), pinned_cells)].any()
        err = np.full((H, NCELL), 50.0)
        err[:, pinned_cells] = 0.5                       # only pinned cells small
        v, n = evalmod._masked_rmse(err, mp)
        assert n == m["S"][:, pinned_cells].sum()
        if n:
            assert np.isclose(v, 0.5)
            used += 1
        assert evalmod._masked_rmse(err, m["S"])[0] > 0.5 or not m["S"].any()
    assert used >= 1
    # threshold is T_frost + PIN_TOL_K
    assert evalmod.PIN_TOL_K == 0.5


def test_no_eval_labels_outside_eval():
    """Only eval.py may use control_T_s / eval labels; models, training,
    calibration, data windows and metric_defs never reference them."""
    files = [os.path.join(ROOT, f) for f in
             ("train.py", "calibrate.py", "data_windows.py", "physics_torch.py",
              "metric_defs.py")]
    files += glob.glob(os.path.join(ROOT, "models", "*.py"))
    for f in files:
        src = open(f).read()
        for bad in ("load_eval_labels", "load_split_eval_labels", "control_T_s",
                    "response_T_s", "eval__", "import eval", "from eval "):
            assert bad not in src, f"{os.path.relpath(f, ROOT)} references {bad}"

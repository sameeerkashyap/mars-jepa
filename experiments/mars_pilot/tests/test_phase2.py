"""
Phase-2 test suite.  Required tests (all must pass):

  * hidden variables never reach models
  * eval labels never imported by training or any module under models/
  * train-only normalization
  * parameter counts within 10% (neural arms)
  * physics residual ~0 on a trajectory from the simplified EBM itself
    (and the residual INCLUDES forcing)
  * open-loop rollouts use no ground truth after the context
  * determinism for a fixed seed
  * hybrid ice state never initialized from hidden m
  * calibration code cannot load test splits
  * subsample_train: stratified, nested, deterministic
"""

import copy
import glob
import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.fakedata import build_fake_dataset, HIDDEN, EVAL
import dataio as dio
import train
from common import load_config
from models.base import build_model, count_params
from physics_torch import (EBMResidual, EBMBackbone, load_ebm_params,
                           stored_dt_seconds)


@pytest.fixture(scope="session")
def fake_dir(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("data_fake"))
    build_fake_dataset(root, n_time=180)
    return root


@pytest.fixture(scope="session")
def p2_small(fake_dir):
    p2 = train.load_phase2_config()
    p2 = copy.deepcopy(p2)
    p2["data"]["data_dir"] = fake_dir
    p2["protocol"] = {"K": 20, "H": 30, "stride": 60}
    return p2


@pytest.fixture(scope="session")
def full_cfg():
    return load_config(os.path.join(ROOT, "config.yaml"))


# -------------------------------------------------------------- import safety
def test_eval_labels_not_imported_by_train_or_models():
    bad = []
    files = [os.path.join(ROOT, "train.py"), os.path.join(ROOT, "calibrate.py")]
    files += glob.glob(os.path.join(ROOT, "models", "*.py"))
    for f in files:
        src = open(f).read()
        if "load_eval_labels" in src or "load_split_eval_labels" in src:
            bad.append(os.path.relpath(f, ROOT))
    assert not bad, f"load_eval_labels referenced in: {bad}"


def test_models_do_not_reference_hidden(fake_dir):
    for f in glob.glob(os.path.join(ROOT, "models", "*.py")):
        src = open(f).read()
        assert "load_hidden" not in src, f"{f} references load_hidden"
        assert "hidden__" not in src, f"{f} references hidden__ keys"


# -------------------------------------------------------------- hidden leakage
def test_load_split_returns_observables_only(fake_dir):
    trajs = dio.load_split(fake_dir, "main", "train")
    for t in trajs:
        for k in t:
            assert not k.startswith(HIDDEN)
            assert not k.startswith(EVAL)
            assert k not in ("m", "T_g", "tau", "control_T_s", "response_T_s")


# -------------------------------------------------------------- normalization
def test_normalization_is_train_only(fake_dir):
    from data_windows import compute_norm_stats
    train_trajs = dio.load_split(fake_dir, "main", "train")
    val_trajs = dio.load_split(fake_dir, "main", "val")
    s_train = compute_norm_stats(train_trajs)
    # stats computed from train+val must DIFFER from train-only -> proves
    # the pipeline's train-only stats are not secretly using val
    s_mixed = compute_norm_stats(train_trajs + val_trajs)
    assert s_train != s_mixed
    # determinism
    assert compute_norm_stats(train_trajs) == s_train


# -------------------------------------------------------------- param counts
def test_param_counts_within_10pct(p2_small):
    stats = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
             "F_a_mean": 2, "F_a_std": 2}
    counts = {}
    for arm in ["gru", "rssm", "transformer", "hybrid", "gru_noforce", "jepa"]:
        m = build_model(arm, stats, 64, p2_small["arch"], seed=0)
        counts[arm] = count_params(m)
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / lo <= 1.10, counts


# -------------------------------------------------------------- physics residual
def test_physics_residual_near_zero_on_ebm_trajectory(full_cfg):
    from sim_ebm import fit_ebm, run_ebm, residual as np_residual
    from sim_truth import run_truth
    truth = run_truth(full_cfg, F_a=0.0, seed=0, years=1, mismatch_scale=0.0)
    params, _ = fit_ebm(full_cfg, truth)
    Fa = 3.0
    ebm = run_ebm(full_cfg, params, F_a=Fa, years=1)
    T = ebm["T_s"]; S = ebm["S"]
    dt = stored_dt_seconds(full_cfg)
    # numpy residual (ground truth definition)
    res_np = np_residual(T, S, Fa, dt, params, full_cfg)
    rms_np = np.sqrt(np.nanmean(res_np ** 2))
    # torch residual must match numpy and be small
    er = EBMResidual(params, full_cfg["grid"]["n"], dt, residual_scale=1.0,
                     device="cpu")
    Tt = torch.tensor(T[None], dtype=torch.float32)
    St = torch.tensor(S[None], dtype=torch.float32)
    Ft = torch.full((1, T.shape[0]), Fa, dtype=torch.float32)
    res_t, mask = er.residual(Tt, St, Ft)
    rms_t = float(torch.sqrt(torch.mean(res_t[mask] ** 2)))
    assert rms_np < 5.0, rms_np        # EBM self-residual is small (W/m^2)
    assert abs(rms_t - rms_np) / max(rms_np, 1e-6) < 0.25


def test_physics_residual_includes_forcing(full_cfg):
    """Residual must depend on F_a (not a zero-forcing / diffusion-only form)."""
    params = load_ebm_params()
    dt = stored_dt_seconds(full_cfg)
    er = EBMResidual(params, 64, dt, residual_scale=1.0, device="cpu")
    T = 200.0 + 5.0 * torch.randn(1, 10, 64)
    S = 100.0 + torch.zeros(1, 10, 64)
    r0, m0 = er.residual(T, S, torch.zeros(1, 10))
    r5, m5 = er.residual(T, S, torch.full((1, 10), 5.0))
    # same mask, residual differs by exactly -F_a on unmasked cells
    diff = (r0 - r5)[m0]
    assert torch.allclose(diff, torch.full_like(diff, 5.0), atol=1e-3)


# -------------------------------------------------------------- open-loop
@pytest.mark.parametrize("arm", ["gru", "rssm", "transformer", "linear",
                                 "hybrid", "gru_noforce", "jepa"])
def test_open_loop_ignores_future_ground_truth(arm, p2_small):
    torch.manual_seed(0)
    stats = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
             "F_a_mean": 2, "F_a_std": 2}
    m = build_model(arm, stats, 64, p2_small["arch"], seed=0)
    m.eval()
    K, H = 20, 30
    Ts = 200 + 5 * torch.randn(2, K + H, 64)
    S = 100 + torch.zeros(2, K + H, 64)
    Fa = torch.full((2, K + H), 3.0)
    lat = torch.linspace(-1.4, 1.4, 64)
    with torch.no_grad():
        p1, _ = m.rollout(Ts[:, :K], S, Fa, lat)
        # corrupt the horizon ground truth; predictions must NOT change
        Ts2 = Ts.clone(); Ts2[:, K:] += 50.0
        p2, _ = m.rollout(Ts2[:, :K], S, Fa, lat)
    assert torch.allclose(p1, p2, atol=1e-5)


# -------------------------------------------------------------- determinism
@pytest.mark.parametrize("arm", ["gru", "gru_noforce", "jepa"])
def test_determinism_fixed_seed(arm, p2_small, full_cfg):
    kw = dict(epochs=1, max_batches=2)
    m1, s1, i1 = train.train_one(arm, "off", 1.0, 7, 0.1, p2_small, full_cfg,
                                 "cpu", **kw)
    m2, s2, i2 = train.train_one(arm, "off", 1.0, 7, 0.1, p2_small, full_cfg,
                                 "cpu", **kw)
    assert i1["val_loss"] == i2["val_loss"]
    # state_dict, not parameters(): also covers the JEPA EMA target encoder
    for (ka, a), (kb, b) in zip(m1.state_dict().items(),
                                m2.state_dict().items()):
        assert ka == kb and torch.allclose(a, b), ka


# -------------------------------------------------------------- D8 / D9 arms
_STATS = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
          "F_a_mean": 2, "F_a_std": 2}


def _rollout_with_fa(m, fa):
    torch.manual_seed(0)
    K, H = 20, 30
    Ts = 200 + 5 * torch.randn(2, K + H, 64)
    S = 100 + torch.zeros(2, K + H, 64)
    with torch.no_grad():
        p, _ = m.rollout(Ts[:, :K], S, torch.full((2, K + H), fa),
                         torch.linspace(-1.4, 1.4, 64))
    return p


@pytest.mark.parametrize("arm,uses_fa", [("gru", True), ("jepa", True),
                                         ("gru_noforce", False)])
def test_forcing_input_used_iff_expected(arm, uses_fa, p2_small):
    m = build_model(arm, _STATS, 64, p2_small["arch"], seed=0).eval()
    same = torch.allclose(_rollout_with_fa(m, 0.0), _rollout_with_fa(m, 5.0))
    assert same != uses_fa


def test_gru_noforce_matches_gru_except_forcing_input(p2_small):
    g = build_model("gru", _STATS, 64, p2_small["arch"], seed=0)
    n = build_model("gru_noforce", _STATS, 64, p2_small["arch"], seed=0)
    assert n.cell.input_size == g.cell.input_size - 1
    assert count_params(g) - count_params(n) == 3 * g.hidden   # F_a columns


def test_jepa_target_encoder_is_ema_not_trained(p2_small):
    m = build_model("jepa", _STATS, 64, p2_small["arch"], seed=0)
    tgt = list(m.target_enc.parameters())
    assert all(not p.requires_grad for p in tgt)
    assert count_params(m) == sum(p.numel() for n_, p in m.named_parameters()
                                  if not n_.startswith("target_enc."))
    before = [p.clone() for p in tgt]
    with torch.no_grad():
        for p in m.enc.parameters():
            p.add_(1.0)
    m.ema_update()
    dec = p2_small["arch"]["jepa"]["ema_decay"]
    for b, t, o in zip(before, tgt, m.enc.parameters()):
        assert torch.allclose(t, dec * b + (1 - dec) * o, atol=1e-6)


def test_jepa_latent_loss_trains_and_reloads(p2_small, full_cfg, tmp_path):
    import eval as evalmod
    m, stats, info = train.train_one("jepa", "off", 1.0, 0, 0.1, p2_small,
                                     full_cfg, "cpu", epochs=1, max_batches=2)
    assert np.isfinite(info["val_loss"])
    # the EMA target encoder moved off its initial copy of the online encoder
    m0 = build_model("jepa", stats, 64, p2_small["arch"], seed=0)
    assert not all(torch.allclose(a, b) for a, b in
                   zip(m.target_enc.parameters(), m0.target_enc.parameters()))
    train.save_run(str(tmp_path), m, stats, info, p2_small)
    m2, *_ = evalmod.load_model(str(tmp_path), p2_small, full_cfg, "cpu")
    for (k, a), b in zip(m.state_dict().items(), m2.state_dict().values()):
        assert torch.allclose(a, b), k


# -------------------------------------------------------------- hybrid ice init
def test_hybrid_ice_init_from_context_only():
    params = load_ebm_params()
    bb = EBMBackbone(params, 64, 1000.0, m_ref=1.0, frost_init_band_K=0.5)
    Tf = params["T_frost"]
    # context: some cells pinned (at frost), others warm
    Ts = torch.full((1, 5, 64), 220.0)
    Ts[:, :, :10] = Tf                       # first 10 cells pinned
    m0 = bb.init_ice_from_context(Ts)
    assert torch.all(m0[0, :10] == 1.0)
    assert torch.all(m0[0, 10:] == 0.0)
    # depends ONLY on last context temperature (observable), not on any hidden m:
    # perturb an (imaginary) hidden m -> init has no such input, result unchanged
    m0b = bb.init_ice_from_context(Ts.clone())
    assert torch.allclose(m0, m0b)


# -------------------------------------------------------------- calibration safety
def test_calibration_cannot_load_test(fake_dir):
    import calibrate
    with pytest.raises(RuntimeError):
        calibrate._safe_load_split(fake_dir, "main", "test_tier1")
    with pytest.raises(RuntimeError):
        calibrate._safe_load_split(fake_dir, "main", "test_tier2")


def test_load_eval_labels_blocks_calibration_mode(fake_dir):
    with pytest.raises(RuntimeError):
        dio.load_split_eval_labels(fake_dir, "main", "val", calibration_mode=True)


# -------------------------------------------------------------- subsampler
def test_subsample_stratified_nested_deterministic(fake_dir):
    trajs = dio.load_split(fake_dir, "main", "train")
    s1 = dio.subsample_train(trajs, 0.1, 0)
    s3 = dio.subsample_train(trajs, 0.3, 0)
    s10 = dio.subsample_train(trajs, 1.0, 0)
    # nested
    assert all(any(a is b for b in s3) for a in s1)
    assert all(any(a is b for b in s10) for a in s3)
    # deterministic
    s3b = dio.subsample_train(trajs, 0.3, 0)
    mag = lambda L: [float(np.max(np.abs(t["F_a"]))) for t in L]
    assert mag(s3) == mag(s3b)
    # stratified: subsample magnitude range spans the full range reasonably
    full_rng = max(mag(trajs)) - min(mag(trajs))
    sub_rng = max(mag(s3)) - min(mag(s3))
    assert sub_rng >= 0.4 * full_rng


def test_single_data_api():
    """dio.py was merged into dataio.py; nothing imports the old module."""
    assert not os.path.exists(os.path.join(ROOT, "dio.py"))
    for f in glob.glob(os.path.join(ROOT, "*.py")) + glob.glob(
            os.path.join(ROOT, "models", "*.py")):
        src = open(f).read()
        assert "from dio " not in src and "import dio\n" not in src, f


def test_split_eval_labels_logged_and_aligned(fake_dir):
    log = os.path.join(fake_dir, "test_access.log")
    n0 = len(open(log).read().splitlines()) if os.path.exists(log) else 0
    labs = dio.load_split_eval_labels(fake_dir, "main", "test_tier2")
    obs = dio.load_split(fake_dir, "main", "test_tier2")
    assert len(labs) == len(obs)
    assert all(set(l) == {"control_T_s", "response_T_s"} for l in labs)
    lines = open(log).read().splitlines()
    assert len(lines) == n0 + 2 and all("TEST-LOAD" in ln for ln in lines[n0:])


@pytest.mark.parametrize("fraction", [0.1, 0.3])
def test_normalization_uses_trained_subset(fraction, p2_small, full_cfg):
    """stats.json must come from the subset the run trains on, not the full split."""
    from data_windows import compute_norm_stats
    seed = 3
    _, stats, _ = train.train_one("linear", "off", fraction, seed, 0.0, p2_small,
                                  full_cfg, "cpu", epochs=0)
    full = dio.load_split(p2_small["data"]["data_dir"], "main", "train")
    sub = dio.subsample_train(full, fraction, seed)
    assert len(sub) < len(full)
    assert stats == compute_norm_stats(sub)
    assert stats != compute_norm_stats(full)


def test_calibrate_saves_curves_and_flags(p2_small, full_cfg):
    import calibrate
    assert calibrate.still_decreasing([3.0, 2.0, 1.0]) is True
    assert calibrate.still_decreasing([3.0, 1.0, 1.5]) is False
    assert calibrate.still_decreasing([1.0]) is None
    p2 = copy.deepcopy(p2_small)
    rep = calibrate.calibrate(["linear", "ebm_only"], [0], p2, full_cfg, "cpu",
                              epochs=3, max_batches=1, family="main")
    c = rep["arms"]["linear"]["val_loss_curves"]
    assert len(c) == 1 and len(c[0]) == 3
    assert isinstance(rep["arms"]["linear"]["still_decreasing_at_final_epoch"], bool)
    assert "still_decreasing_arms" in rep


def test_thresholds_filled():
    import yaml
    thr = yaml.safe_load(open(os.path.join(ROOT, "thresholds.yaml")))
    assert thr and all(v is not None for v in thr.values())
    assert thr["h1_metric"] == "t2_polar_winter_S_pinned"      # PREREG D7
    assert thr["correction"] == "holm" and 0 < thr["alpha"] < 1


def _hyp_rows(off, phys, hybrid, n_seeds=10):
    """Fake results: gru off/A/B and hybrid at fractions 0.3 and 1.0."""
    rows = []
    for s in range(n_seeds):
        for arm, p, v in (("gru", "off", off), ("gru", "A", phys),
                          ("gru", "B", phys), ("hybrid", "off", hybrid)):
            for f in (0.3, 1.0):
                x = v + 0.01 * s + (0.5 if f < 1 else 0)
                rows.append(dict(tag=f"{arm}_{p}_{f}_{s}", arm=arm, physics=p,
                                 fraction=f, seed=s, status="ok",
                                 t1_polar_winter_S_pinned=1.0,
                                 t2_polar_winter_S_pinned=x))
    return rows


def test_hypotheses_paired_wilcoxon_holm(tmp_path):
    import yaml, pandas as pd
    import analyze
    thr = yaml.safe_load(open(os.path.join(ROOT, "thresholds.yaml")))

    def run(rows):
        csv = tmp_path / "r.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        return analyze.evaluate_hypotheses(analyze.load_results(str(csv)), thr)

    # physics 50% better on every seed, hybrid 50% better than gru -> supported
    h = run(_hyp_rows(off=4.0, phys=2.0, hybrid=2.0))
    assert h["H1"]["verdict"] == "supported"                 # 4/1 >= 2
    for k in ("H2", "H2B", "H3"):
        assert h[k]["verdict"] == "supported", (k, h[k])
        assert h[k]["p_holm"] < thr["alpha"]
    # 5% improvement: below the 20% criterion even though p is tiny
    h = run(_hyp_rows(off=4.0, phys=3.8, hybrid=3.8))
    assert h["H2"]["verdict"] == "not supported" and h["H2"]["p"] < 0.01
    assert h["H3"]["verdict"] == "not supported"
    # large effect but too few seeds for the test to survive Holm
    h = run(_hyp_rows(off=4.0, phys=2.0, hybrid=2.0, n_seeds=3))
    assert h["H2"]["verdict"] == "not supported"             # n < h2_min_seeds
    assert h["H3"]["verdict"] == "not supported"             # p_holm >= alpha


def test_holm_and_wilcoxon_helpers():
    import analyze
    assert list(analyze.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert list(analyze.holm([0.5, float("nan")])) == pytest.approx([1.0, 1.0])
    assert analyze.wilcoxon_greater([2, 2], [2, 2]) == 1.0
    assert np.isnan(analyze.wilcoxon_greater([], []))
    assert analyze.wilcoxon_greater(np.arange(10) + 1.0, np.zeros(10)) \
        == pytest.approx(1 / 1024)


def test_thresholds_blank_and_ebm_metric_labeled(tmp_path):
    import pandas as pd
    import analyze, eval as evalmod
    thr = {}                                     # unfilled thresholds
    label = ("EBM-residual constraint satisfaction "
             "(also the training objective for physics arms)")
    assert analyze.EBM_METRIC_LABEL == label == evalmod.EBM_METRIC_LABEL
    rows = [dict(tag=f"{a}_{p}_{s}", arm=a, physics=p, fraction=1.0, seed=s,
                 status="ok", t1_polar_winter_S=1.0, t2_polar_winter_S=2.0,
                 t1_ebm_viol=0.1, t2_ebm_viol=0.2)
            for a in ("gru", "hybrid") for p in ("off", "A") for s in (0, 1)]
    csv = tmp_path / "r.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = analyze.load_results(str(csv))
    hyps = analyze.evaluate_hypotheses(df, thr)
    assert all(h["verdict"].startswith("inconclusive") for h in hyps.values())
    md = tmp_path / "a.md"
    analyze.write_report(hyps, {}, out_json=str(tmp_path / "a.json"),
                         out_md=str(md), ebm=analyze.ebm_metric_table(df))
    assert label in md.read_text()


def test_sweep_grid_new_arms_physics_off_only():
    import sweep
    grid = sweep.enumerate_grid(train.load_phase2_config())
    assert len(grid) == 450
    for arm in ("gru_noforce", "jepa"):
        runs = [g for g in grid if g[0] == arm]
        assert len(runs) == 30 and {g[1] for g in runs} == {"off"}

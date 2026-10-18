"""
eval.py -- evaluate trained arm(s) on the held-out TEST splits.

THIS IS THE ONLY MODULE THAT TOUCHES TEST SPLITS OR EVAL LABELS.
It imports load_split_eval_labels (train.py / models/ must never do so).

Metrics (Tier 1 and Tier 2 reported separately):
  1. Intervention-response error: predicted (T_a - T_0) vs true response, with
     T_0 the same-seed control rollout.  Primary polar-winter T_s (|lat|>60),
     N and S separately; secondary global-mean.
  2. Rollout error vs horizon.
  3. Calibration (stochastic models + ensemble): CRPS, 90% coverage, Spearman
     of spread vs OOD distance.
  4. EBM-residual constraint satisfaction (also the training objective for physics arms):
     mean normalized squared EBM residual of the model's forced trajectory
     (frost-masked).  NOTE: this is the SAME quantity the physics-loss arms
     minimize in training, so it is a constraint-satisfaction score, not an
     independent energy-budget check.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dataio import load_split, load_split_eval_labels   # eval-only imports
from common import load_config
from models.base import build_model
from physics_torch import EBMResidual, load_ebm_params, stored_dt_seconds
from train import load_phase2_config, pick_device

POLAR_DEG = 60.0
EBM_METRIC_LABEL = ("EBM-residual constraint satisfaction (also the training objective for physics arms)")
TRAIN_FA_MAX = 5.0


# --------------------------------------------------------------------------
def load_model(run_dir, p2, full_cfg, device):
    with open(os.path.join(run_dir, "info.json")) as fh:
        info = json.load(fh)
    with open(os.path.join(run_dir, "stats.json")) as fh:
        stats = json.load(fh)
    with open(os.path.join(run_dir, "arch.json")) as fh:
        ad = json.load(fh)
    n_cell = full_cfg["grid"]["n"]
    model = build_model(info["arm"], stats, n_cell, ad["arch"],
                        seed=info.get("seed", 0)).to(device)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, info, stats, ad["protocol"]


def _windows_with_control(traj, labels, K, H, stride):
    """Yield aligned (Ts_ctx, S, Fa, ctrl_ctx, true_resp, true_forced) windows."""
    Ts = np.asarray(traj["T_s"], np.float32)
    S = np.asarray(traj["S"], np.float32)
    Fa = np.asarray(traj["F_a"], np.float32)
    ctrl = np.asarray(labels["control_T_s"], np.float32)
    resp = np.asarray(labels["response_T_s"], np.float32)
    n = Ts.shape[0]
    W = K + H
    out = []
    for s in range(0, n - W + 1, stride):
        out.append(dict(
            Ts=Ts[s:s + W], S=S[s:s + W], Fa=Fa[s:s + W], ctrl=ctrl[s:s + W],
            resp=resp[s + K:s + W], forced=Ts[s + K:s + W]))
    return out


def _polar_masks(lat):
    thr = np.radians(POLAR_DEG)
    return lat > thr, lat < -thr          # (N band, S band)


def _band_rmse(err, band):
    """RMSE of err (time, ncell) over a latitude band (ncell bool)."""
    if band.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err[:, band] ** 2)))


# --------------------------------------------------------------------------
def _model_samples(model, Ts_ctx, S, Fa, lat, n_samples, stochastic):
    """Return array (n_samples, B, H, ncell) of PHYSICAL predictions."""
    outs = []
    for _ in range(max(1, n_samples) if stochastic else 1):
        with torch.no_grad():
            pred, _ = model.rollout(Ts_ctx, S, Fa, lat, sample=stochastic)
        outs.append(pred.cpu().numpy())
    return np.stack(outs, axis=0)


def _ensemble_samples(models, Ts_ctx, S, Fa, lat):
    outs = []
    for m in models:
        with torch.no_grad():
            pred, _ = m.rollout(Ts_ctx, S, Fa, lat, sample=False)
        outs.append(pred.cpu().numpy())
    return np.stack(outs, axis=0)


def _crps(samples, y):
    """CRPS for samples (n,) and scalar y (empirical estimator)."""
    n = samples.shape[0]
    term1 = np.mean(np.abs(samples - y))
    term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
    return term1 - 0.5 * term2


def evaluate(run_dirs, data_dir, family, split, p2, full_cfg, device,
             ensemble=False):
    """Evaluate one arm (or an ensemble of run_dirs) on a split."""
    models, infos, protos = [], [], []
    for rd in run_dirs:
        m, info, stats, proto = load_model(rd, p2, full_cfg, device)
        models.append(m)
        infos.append(info)
        protos.append(proto)
    K, H = protos[0]["K"], protos[0]["H"]
    stride = H
    stochastic = models[0].stochastic
    is_cal = ensemble or stochastic

    trajs = load_split(data_dir, family, split)                 # test access
    labels = load_split_eval_labels(data_dir, family, split)    # eval labels
    lat_np = np.asarray(trajs[0]["lat"], np.float32)
    lat = torch.tensor(lat_np, device=device)
    bandN, bandS = _polar_masks(lat_np)
    w = np.asarray(trajs[0]["w"], np.float32)

    res = EBMResidual(load_ebm_params(), full_cfg["grid"]["n"],
                      stored_dt_seconds(full_cfg),
                      residual_scale=p2["physics"]["residual_scale"],
                      frost_tol_K=p2["physics"]["frost_tol_K"], device=device)

    n_samp = (p2["calibration"]["ensemble_members"] if ensemble
              else p2["calibration"]["n_samples"])

    # accumulators
    respN, respS, respG = [], [], []
    horizon_sq = np.zeros(H)
    horizon_ct = 0
    crps_list, cover_list, spread_list, ood_list = [], [], [], []
    ebm_violation = []

    for traj, lab in zip(trajs, labels):
        wins = _windows_with_control(traj, lab, K, H, stride)
        for wd in wins:
            Ts_ctx = torch.tensor(wd["Ts"][None, :K], device=device)
            S_t = torch.tensor(wd["S"][None], device=device)
            Fa_t = torch.tensor(wd["Fa"][None], device=device)
            ctrl_ctx = torch.tensor(wd["ctrl"][None, :K], device=device)
            Fa_zero = torch.zeros_like(Fa_t)

            # forced + control model rollouts
            if ensemble:
                fs = _ensemble_samples(models, Ts_ctx, S_t, Fa_t, lat)
                cs = _ensemble_samples(models, ctrl_ctx, S_t, Fa_zero, lat)
            else:
                fs = _model_samples(models[0], Ts_ctx, S_t, Fa_t, lat, n_samp,
                                    stochastic)
                cs = _model_samples(models[0], ctrl_ctx, S_t, Fa_zero, lat,
                                    n_samp, stochastic)
            # mean prediction (B=1 -> drop)
            Ta = fs.mean(0)[0]                # (H, ncell)
            T0 = cs.mean(0)[0]
            pred_resp = Ta - T0
            err = pred_resp - wd["resp"]     # (H, ncell)

            respN.append(_band_rmse(err, bandN))
            respS.append(_band_rmse(err, bandS))
            respG.append(float(np.sqrt(np.mean((err @ w) ** 2))))

            # rollout error vs horizon (forced global-mean)
            ferr = (Ta - wd["forced"])       # (H, ncell)
            horizon_sq += (ferr @ w) ** 2
            horizon_ct += 1

            # EBM-residual constraint satisfaction (= physics-arm training loss)
            with torch.no_grad():
                Tcat = torch.tensor(Ta[None], device=device)
                r = res.loss(Tcat, S_t[:, K:], Fa_t[:, K:]).item()
            ebm_violation.append(r)

            # calibration (per cell, forced samples vs true forced)
            if is_cal:
                S_samp = fs[:, 0]            # (n, H, ncell)
                y = wd["forced"]            # (H, ncell)
                # CRPS averaged over polar bands + global
                band = bandN | bandS
                idx = np.where(band)[0]
                cr = np.mean([_crps(S_samp[:, h, c], y[h, c])
                              for h in range(0, H, max(1, H // 8))
                              for c in idx]) if idx.size else np.nan
                crps_list.append(cr)
                lo = np.percentile(S_samp, 5, axis=0)
                hi = np.percentile(S_samp, 95, axis=0)
                cover = np.mean((y >= lo) & (y <= hi))
                cover_list.append(float(cover))
                spread_list.append(float(np.mean(S_samp.std(0))))
                ood_list.append(max(0.0, float(np.max(np.abs(wd["Fa"])))
                                    - TRAIN_FA_MAX))

    metrics = {
        "split": split, "n_windows": horizon_ct,
        "intervention_response_rmse": {
            "polar_N": float(np.nanmean(respN)),
            "polar_S": float(np.nanmean(respS)),
            "global_mean": float(np.nanmean(respG)),
        },
        "rollout_rmse_vs_horizon": np.sqrt(horizon_sq / max(horizon_ct, 1)).tolist(),
        "energy_budget_violation_mean": float(np.mean(ebm_violation)),
        "energy_budget_violation_label": EBM_METRIC_LABEL,
    }
    if is_cal and crps_list:
        sp = np.array(spread_list)
        ood = np.array(ood_list)
        rho = _spearman(sp, ood)
        metrics["calibration"] = {
            "crps": float(np.nanmean(crps_list)),
            "coverage_90": float(np.nanmean(cover_list)),
            "spread_ood_spearman": rho,
            "n_samples": n_samp,
        }
    return metrics


def _spearman(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="run dir (repeat for an ensemble)")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--family", default="main")
    ap.add_argument("--splits", nargs="+", default=["test_tier1", "test_tier2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true",
                    help="do not print metric values (write --out only)")
    args = ap.parse_args()

    p2 = load_phase2_config()
    full_cfg = load_config()
    device = args.device or pick_device(p2["train"]["device"])
    data_dir = args.data_dir or p2["data"]["data_dir"]

    results = {"runs": args.run, "ensemble": args.ensemble, "tiers": {}}
    for split in args.splits:
        tier = "tier1" if split.endswith("tier1") else (
            "tier2" if split.endswith("tier2") else split)
        results["tiers"][tier] = evaluate(args.run, data_dir, args.family, split,
                                          p2, full_cfg, device,
                                          ensemble=args.ensemble)
    if not args.quiet:
        print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

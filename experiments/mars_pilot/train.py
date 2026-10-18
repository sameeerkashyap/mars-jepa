"""
train.py -- train one model arm (one configuration) for the Mars pilot.

Usage:
    python3 train.py --arm gru --physics off --fraction 1.0 --seed 0 \
        --out runs/gru_off_f100_s0

Imports ONLY load_split / subsample_train from the data layer (the eval-label
loader is never imported here; enforced by a test).  Normalization uses the
training subset actually used by the run (the data-fraction subsample).

Open-loop protocol: observe K context steps, roll out H steps feeding back the
model's OWN predictions (no ground truth after the context).

Physics loss (arms gru/rssm/transformer):
  off : data loss only
  A   : + lambda*mean(residual^2) on labeled training-range rollouts (strict)
  B   : A + residual on unlabeled rollouts with F_a sampled up to fa_max
Residual is the forcing-aware EBM residual, frost-pinned cells masked,
normalized by a fixed scale (config physics.residual_scale).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml

from common import load_config
from dataio import load_split, subsample_train       # NOTE: eval labels never imported here
from data_windows import compute_norm_stats, make_windows, WindowDataset
from models.base import build_model, count_params
from physics_torch import (EBMResidual, load_ebm_params, stored_dt_seconds)

PHYSICS_ARMS = {"gru", "rssm", "transformer"}
NO_TRAIN_ARMS = {"ebm_only"}


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_phase2_config(path="config_phase2.yaml"):
    with open(path) as fh:
        return yaml.safe_load(fh)


def _normalized_data_loss(pred_phys, target_phys, stats):
    m, s = stats["T_s_mean"], stats["T_s_std"]
    return torch.mean(((pred_phys - target_phys) / s) ** 2)


def train_one(arm, physics, fraction, seed, lam, p2, full_cfg, device,
              epochs=None, max_batches=None, verbose=False,
              calibration_mode=False, record_curve=False):
    """Train a single configuration.  Returns (model, stats, info)."""
    set_seed(seed)
    K = p2["protocol"]["K"]
    H = p2["protocol"]["H"]
    stride = p2["protocol"]["stride"]
    dcfg = p2["data"]
    n_cell = full_cfg["grid"]["n"]

    # ---- data (observables only; eval labels never touched) ----
    train_all = load_split(dcfg["data_dir"], dcfg["family"], dcfg["train_split"],
                           calibration_mode=calibration_mode)
    val_all = load_split(dcfg["data_dir"], dcfg["family"], dcfg["val_split"],
                         calibration_mode=calibration_mode)
    train_sub = subsample_train(train_all, fraction, seed)
    # normalization from the subset this run ACTUALLY trains on (10%/30%/100%)
    stats = compute_norm_stats(train_sub)

    tr_w = make_windows(train_sub, K, H, stride)
    va_w = make_windows(val_all, K, H, stride)
    tr_ds = WindowDataset(tr_w, stats, K, H)
    va_ds = WindowDataset(va_w, stats, K, H)
    lat = torch.tensor(tr_w["lat"], device=device)

    gen = torch.Generator().manual_seed(seed)
    bs = p2["train"]["batch_size"]
    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=True,
                                        generator=gen, num_workers=0)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=bs, shuffle=False,
                                        num_workers=0)

    # ---- model ----
    model = build_model(arm, stats, n_cell, p2["arch"], seed=seed).to(device)
    n_params = count_params(model)

    # ---- physics residual ----
    use_phys = physics in ("A", "B") and arm in PHYSICS_ARMS
    res = None
    if use_phys:
        res = EBMResidual(load_ebm_params(), n_cell, stored_dt_seconds(full_cfg),
                          residual_scale=p2["physics"]["residual_scale"],
                          frost_tol_K=p2["physics"]["frost_tol_K"], device=device)
    colloc = physics == "B" and arm in PHYSICS_ARMS
    coll_cfg = p2["physics"]["collocation"]
    coll_rng = np.random.default_rng(seed + 10007)

    if arm in NO_TRAIN_ARMS:
        info = {"arm": arm, "physics": physics, "fraction": fraction,
                "seed": seed, "lambda": 0.0, "n_params": n_params,
                "val_loss": float("nan"), "trained": False}
        val_loss = _eval_val(model, va_dl, stats, K, lat, device)
        info["val_loss"] = val_loss
        return model, stats, info

    opt = torch.optim.Adam(model.parameters(), lr=p2["train"]["lr"],
                           weight_decay=p2["train"]["weight_decay"])
    n_epochs = epochs if epochs is not None else p2["train"]["epochs"]
    clip = p2["train"]["grad_clip"]
    val_curve = []                      # per-epoch val loss (record_curve=True)

    for ep in range(n_epochs):
        model.train()
        for bi, (Ts, S, Fa) in enumerate(tr_dl):
            if max_batches is not None and bi >= max_batches:
                break
            Ts, S, Fa = Ts.to(device), S.to(device), Fa.to(device)
            Ts_ctx = Ts[:, :K]
            target = Ts[:, K:]
            pred, aux = model.rollout(Ts_ctx, S, Fa, lat)
            loss = _normalized_data_loss(pred, target, stats)
            if "kl" in aux:
                loss = loss + aux["kl"]
            if use_phys:
                S_h = S[:, K:]
                Fa_h = Fa[:, K:]
                loss = loss + lam * res.loss(pred, S_h, Fa_h)
                if colloc:
                    loss = loss + lam * _collocation_loss(
                        model, Ts_ctx, S, K, H, res, coll_cfg, coll_rng,
                        lat, device)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        if verbose or record_curve:
            vl = _eval_val(model, va_dl, stats, K, lat, device)
            val_curve.append(vl)
            if verbose:
                print(f"  ep {ep}: val_loss={vl:.4f}")

    val_loss = (val_curve[-1] if val_curve
                else _eval_val(model, va_dl, stats, K, lat, device))
    info = {"arm": arm, "physics": physics, "fraction": fraction, "seed": seed,
            "lambda": float(lam), "n_params": n_params, "val_loss": val_loss,
            "trained": True, "K": K, "H": H}
    if record_curve:
        info["val_curve"] = [float(v) for v in val_curve]
    return model, stats, info


def _collocation_loss(model, Ts_ctx, S, K, H, res, coll_cfg, rng, lat, device):
    """Variant-B: residual on UNLABELED rollouts, F_a sampled up to fa_max.

    Uses the real context and insolation; replaces F_a over the horizon with a
    constant drawn uniformly in [0, fa_max].  No truth labels are used."""
    B = Ts_ctx.shape[0]
    fa_val = float(rng.uniform(0.0, coll_cfg["fa_max"]))
    Fa = torch.full((B, K + H), fa_val, device=device)
    pred, _ = model.rollout(Ts_ctx, S, Fa, lat)
    return res.loss(pred, S[:, K:], Fa[:, K:])


def _eval_val(model, va_dl, stats, K, lat, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for Ts, S, Fa in va_dl:
            Ts, S, Fa = Ts.to(device), S.to(device), Fa.to(device)
            pred, _ = model.rollout(Ts[:, :K], S, Fa, lat)
            tot += _normalized_data_loss(pred, Ts[:, K:], stats).item() * Ts.shape[0]
            n += Ts.shape[0]
    return tot / max(n, 1)


def save_run(out, model, stats, info, p2):
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    with open(os.path.join(out, "info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    with open(os.path.join(out, "arch.json"), "w") as fh:
        json.dump({"arch": p2["arch"], "protocol": p2["protocol"]}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--physics", default="off", choices=["off", "A", "B"])
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--select-lambda", action="store_true",
                    help="search physics.lambda_grid on VALIDATION")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    p2 = load_phase2_config()
    if args.data_dir:
        p2["data"]["data_dir"] = args.data_dir
    full_cfg = load_config()
    device = args.device or pick_device(p2["train"]["device"])

    lam_default = p2["physics"]["lambda_default"]
    lam = args.lam if args.lam is not None else lam_default

    def _run(lm):
        return train_one(args.arm, args.physics, args.fraction, args.seed, lm,
                         p2, full_cfg, device, epochs=args.epochs,
                         max_batches=args.max_batches, verbose=args.verbose)

    if args.select_lambda and args.physics in ("A", "B") and args.arm in PHYSICS_ARMS:
        best = None
        for lm in p2["physics"]["lambda_grid"]:
            model, stats, info = _run(lm)
            print(f"lambda={lm}: val_loss={info['val_loss']:.5f}")
            if best is None or info["val_loss"] < best[2]["val_loss"]:
                best = (model, stats, info)
        model, stats, info = best
        info["lambda_selected"] = info["lambda"]
    else:
        model, stats, info = _run(lam)

    print(json.dumps(info, indent=2))
    out = args.out or (f"runs/{args.arm}_{args.physics}_"
                       f"f{int(args.fraction*100)}_s{args.seed}")
    save_run(out, model, stats, info, p2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

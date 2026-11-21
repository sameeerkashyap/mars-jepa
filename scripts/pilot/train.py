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
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import yaml

from benchmarks.registry import build_model, count_params
from data.pilot import (  # NOTE: eval labels never imported here
    load_split,
    subsample_train,
)
from data.windows import WindowDataset, compute_norm_stats, make_windows
from paths import PILOT_OUTPUT, PILOT_SPEC, REPO_ROOT
from physics.surface import load_config
from physics.surface_torch import EBMResidual, load_ebm_params, stored_dt_seconds

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_phase2_config(path=PILOT_SPEC / "config_phase2.yaml"):
    path = Path(path)
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if "base" in cfg:
        base = load_phase2_config(path.parent / cfg.pop("base"))
        _merge_config(base, cfg)
        cfg = base
    cfg["data"]["data_dir"] = str(REPO_ROOT / cfg["data"]["data_dir"])
    return cfg


def _merge_config(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _normalized_data_loss(pred_phys, target_phys, stats):
    _m, s = stats["T_s_mean"], stats["T_s_std"]
    return torch.mean(((pred_phys - target_phys) / s) ** 2)


def train_one(arm, physics, fraction, seed, lam, p2, full_cfg, device,
              epochs=None, max_batches=None, verbose=False,
              calibration_mode=False, record_curve=False, subset_seed=None,
              checkpoint_dir=None, resume=False):
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
    subset_seed = seed if subset_seed is None else subset_seed
    train_sub = subsample_train(train_all, fraction, subset_seed)
    observed_hash = hashlib.sha256()
    subset_indices = [i for i, traj in enumerate(train_all)
                      if any(traj is selected for selected in train_sub)]
    if checkpoint_dir:
        for traj in train_sub + val_all:
            for key in sorted(traj):
                observed_hash.update(key.encode())
                observed_hash.update(np.ascontiguousarray(traj[key]).tobytes())
    # normalization from the subset this run ACTUALLY trains on (10%/30%/100%)
    stats = compute_norm_stats(train_sub)

    tr_w = make_windows(train_sub, K, H, stride)
    va_w = make_windows(val_all, K, H, stride)
    tr_ds = WindowDataset(tr_w, stats, K, H)
    va_ds = WindowDataset(va_w, stats, K, H)
    lat = torch.tensor(tr_w["lat"], device=device)
    del tr_w, va_w, train_all, train_sub, val_all

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

    identity = None
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir) / "latest.pt"
        data_root = Path(dcfg["data_dir"])
        data_hash = hashlib.sha256((data_root / "manifest.json").read_bytes())
        if (data_root / "SHA256SUMS").exists():
            data_hash.update((data_root / "SHA256SUMS").read_bytes())
        identity = {"arm": arm, "physics": physics, "fraction": fraction,
                    "seed": seed, "subset_seed": subset_seed, "lambda": lam,
                    "config": copy.deepcopy(p2), "physics_config": full_cfg,
                    "data_sha256": data_hash.hexdigest(), "device": device,
                    "observed_sha256": observed_hash.hexdigest(),
                    "subset_indices": subset_indices,
                    "max_batches": max_batches,
                    "source": os.environ.get("MARSWORLD_SOURCE_SHA256", "local")}
        identity["config"]["train"].pop("epochs", None)
        if checkpoint_path.exists() and not resume:
            raise ValueError("Checkpoint exists; use --resume or a new output directory")

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
    val_curve = []
    best_loss, best_state, best_epoch = float("inf"), None, 0
    stop_best, bad_epochs, start_epoch = float("inf"), 0, 0
    early = p2["train"].get("early_stop", {})
    stopped = False
    epoch_seconds = []
    batch_seconds, validation_seconds = [], []
    if resume and checkpoint_path and checkpoint_path.exists():
        # Checkpoints are trusted artifacts produced by this training process.
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise ValueError("Checkpoint configuration, source, data, or device mismatch")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch = ck["epoch"]
        val_curve, epoch_seconds = ck["val_curve"], ck["epoch_seconds"]
        best_loss, best_state, best_epoch = ck["best_loss"], ck["best_model"], ck["best_epoch"]
        stop_best, bad_epochs, stopped = ck["stop_best"], ck["bad_epochs"], ck["stopped"]
        random.setstate(ck["random"])
        np.random.set_state(ck["numpy"])
        torch.set_rng_state(ck["torch"])
        gen.set_state(ck["loader"])
        coll_rng.bit_generator.state = ck["collocation"]
        if device == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda"])

    completed = start_epoch
    for ep in range(start_epoch, n_epochs):
        if stopped:
            break
        epoch_start = time.monotonic()
        model.train()
        for bi, (Ts, S, Fa) in enumerate(tr_dl):
            if max_batches is not None and bi >= max_batches:
                break
            batch_start = time.monotonic()
            Ts, S, Fa = Ts.to(device), S.to(device), Fa.to(device)
            Ts_ctx = Ts[:, :K]
            target = Ts[:, K:]
            pred, aux = model.rollout(Ts_ctx, S, Fa, lat)
            loss = _normalized_data_loss(pred, target, stats)
            if "kl" in aux:
                loss = loss + aux["kl"]
            if hasattr(model, "latent_loss"):      # JEPA: horizon truth used
                loss = loss + model.latent_loss(aux, target)   # as a TARGET only
            if use_phys:
                S_h = S[:, K:]
                Fa_h = Fa[:, K:]
                loss = loss + lam * res.loss(pred, S_h, Fa_h)
                if colloc:
                    loss = loss + lam * _collocation_loss(
                        model, Ts_ctx, S, K, H, res, coll_cfg, coll_rng,
                        lat, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            if hasattr(model, "ema_update"):
                model.ema_update()
            if max_batches is not None and device == "cuda":
                torch.cuda.synchronize()
            batch_seconds.append(time.monotonic() - batch_start)
            if verbose and (bi + 1) % 10 == 0:
                print(f"epoch={ep + 1} batch={bi + 1}/{len(tr_dl)} loss={loss.item():.6g}",
                      flush=True)
        if verbose or record_curve or checkpoint_dir:
            val_start = time.monotonic()
            vl = _eval_val(model, va_dl, stats, K, lat, device)
            validation_seconds.append(time.monotonic() - val_start)
            if not np.isfinite(vl):
                raise RuntimeError("Non-finite validation loss")
            val_curve.append(vl)
            if verbose:
                print(f"epoch={ep + 1} val_loss={vl:.6g} "
                      f"val_rmse_K={np.sqrt(vl) * stats['T_s_std']:.6g}", flush=True)
            if vl < best_loss:
                best_loss, best_epoch = vl, ep + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if vl < stop_best * (1 - early.get("min_delta", 0.001)):
                stop_best, bad_epochs = vl, 0
            elif ep + 1 > early.get("warmup", 5):
                bad_epochs += 1
            stopped = bool(early and bad_epochs >= early["patience"])
        completed = ep + 1
        epoch_seconds.append(time.monotonic() - epoch_start)
        if checkpoint_path:
            atomic_torch_save({"identity": identity, "model": model.state_dict(),
                               "optimizer": opt.state_dict(), "epoch": completed,
                               "val_curve": val_curve, "epoch_seconds": epoch_seconds,
                               "best_loss": best_loss, "best_model": best_state,
                               "best_epoch": best_epoch, "stop_best": stop_best,
                               "bad_epochs": bad_epochs, "stopped": stopped,
                               "random": random.getstate(), "numpy": np.random.get_state(),
                               "torch": torch.get_rng_state(), "loader": gen.get_state(),
                               "collocation": coll_rng.bit_generator.state,
                               "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else []},
                              checkpoint_path)

    val_loss = (val_curve[-1] if val_curve
                else _eval_val(model, va_dl, stats, K, lat, device))
    info = {"arm": arm, "physics": physics, "fraction": fraction, "seed": seed,
            "lambda": float(lam), "n_params": n_params, "val_loss": val_loss,
            "trained": True, "K": K, "H": H}
    info.update(train_windows=len(tr_ds), val_windows=len(va_ds))
    if checkpoint_dir:
        if best_state is None:
            raise RuntimeError("Budget ended before the first epoch could complete")
        model.load_state_dict(best_state)
        info.update(val_loss=best_loss, best_epoch=best_epoch, epochs_completed=completed,
                    subset_seed=subset_seed, early_stopped=stopped,
                    epoch_seconds=epoch_seconds,
                    train_windows=len(tr_ds), val_windows=len(va_ds), identity=identity)
    if record_curve:
        info["val_curve"] = [float(v) for v in val_curve]
        info.update(batch_seconds=batch_seconds, validation_seconds=validation_seconds)
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
        for bi, (Ts, S, Fa) in enumerate(va_dl):
            Ts, S, Fa = Ts.to(device), S.to(device), Fa.to(device)
            pred, _ = model.rollout(Ts[:, :K], S, Fa, lat)
            tot += _normalized_data_loss(pred, Ts[:, K:], stats).item() * Ts.shape[0]
            n += Ts.shape[0]
            if (bi + 1) % 10 == 0:
                print(f"validation batch={bi + 1}/{len(va_dl)}", flush=True)
    return tot / max(n, 1)


def save_run(out, model, stats, info, p2):
    os.makedirs(out, exist_ok=True)
    atomic_torch_save(model.state_dict(), os.path.join(out, "model.pt"))
    with open(os.path.join(out, "resolved_config.json"), "w") as fh:
        json.dump(p2, fh, indent=2)
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
    ap.add_argument("--config", default=str(PILOT_SPEC / "config_phase2.yaml"))
    ap.add_argument("--subset-seed", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint", action="store_true")
    args = ap.parse_args()

    p2 = load_phase2_config(args.config)
    if args.data_dir:
        p2["data"]["data_dir"] = args.data_dir
    full_cfg = load_config()
    device = args.device or pick_device(p2["train"]["device"])

    lam_default = p2["physics"]["lambda_default"]
    lam = args.lam if args.lam is not None else lam_default
    out = args.out or (f"{PILOT_OUTPUT}/runs/{args.arm}_{args.physics}_"
                       f"f{int(args.fraction*100)}_s{args.seed}")
    if args.select_lambda and (args.checkpoint or args.resume):
        ap.error("Checkpointed runs require one fixed lambda")

    def _run(lm):
        return train_one(args.arm, args.physics, args.fraction, args.seed, lm,
                         p2, full_cfg, device, epochs=args.epochs,
                         max_batches=args.max_batches, verbose=args.verbose,
                         subset_seed=args.subset_seed,
                         checkpoint_dir=out if args.checkpoint or args.resume else None,
                         resume=args.resume)

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
    save_run(out, model, stats, info, p2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

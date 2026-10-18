"""
calibrate.py -- calibration + validation data ONLY.

By construction this script CANNOT load a test split:
  * every data access goes through `_safe_load_split`, which forces
    calibration_mode=True (dataio raises on any test split in that mode);
  * it never imports the eval-label loaders, so intervention labels are untouchable;
  * all errors measured here are label-free VALIDATION rollout errors.

It measures, per arm:
  * seed-to-seed spread (val rollout RMSE across several training seeds),
  * the EBM-only and LINEAR baseline errors,
  * per-seed validation-loss curves (one value per epoch), and flags any arm
    whose val loss is STILL DECREASING at the final epoch (under-trained budget),
and writes calibration_report.json (+ <out>_curves.png).
"""

from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch

from common import load_config
from train import (load_phase2_config, pick_device, train_one, _eval_val,
                   set_seed)
from data_windows import compute_norm_stats, make_windows, WindowDataset
from dataio import load_split


def _safe_load_split(data_dir, family, split):
    """Load a split with the calibration guard ALWAYS on (test splits raise)."""
    return load_split(data_dir, family, split, calibration_mode=True)


def _val_rmse_kelvin(model, p2, full_cfg, device, family, val_split, stats):
    """Label-free validation rollout RMSE in Kelvin."""
    K, H, stride = (p2["protocol"]["K"], p2["protocol"]["H"],
                    p2["protocol"]["stride"])
    val = _safe_load_split(p2["data"]["data_dir"], family, val_split)
    vw = make_windows(val, K, H, stride)
    ds = WindowDataset(vw, stats, K, H)
    lat = torch.tensor(vw["lat"], device=device)
    dl = torch.utils.data.DataLoader(ds, batch_size=p2["train"]["batch_size"])
    norm_mse = _eval_val(model, dl, stats, K, lat, device)
    return float(np.sqrt(norm_mse) * stats["T_s_std"])


def still_decreasing(curve, rel_tol=0.0):
    """True if the last epoch improved val loss by more than rel_tol (relative)."""
    if len(curve) < 2:
        return None                     # undefined with < 2 epochs
    return bool(curve[-1] < curve[-2] * (1.0 - rel_tol))


def calibrate(arms, seeds, p2, full_cfg, device, epochs, max_batches,
              family="calibration", val_split="val", rel_tol=0.0):
    report = {"family": family, "val_split": val_split, "seeds": list(seeds),
              "arms": {}}
    # normalization from calibration-family train only
    train_all = _safe_load_split(p2["data"]["data_dir"], family, "train")
    stats = compute_norm_stats(train_all)

    p2c = copy.deepcopy(p2)
    p2c["data"]["family"] = family
    p2c["data"]["train_split"] = "train"
    p2c["data"]["val_split"] = val_split

    for arm in arms:
        rmses, curves = [], []
        n_params = None
        trainable = arm not in ("ebm_only",)
        use_seeds = seeds if trainable else seeds[:1]
        for sd in use_seeds:
            model, mstats, info = train_one(
                arm, "off", 1.0, sd, p2["physics"]["lambda_default"],
                p2c, full_cfg, device, epochs=epochs, max_batches=max_batches,
                calibration_mode=True, record_curve=True)
            n_params = info["n_params"]
            curves.append(info.get("val_curve", []))
            rmses.append(_val_rmse_kelvin(model, p2c, full_cfg, device, family,
                                          val_split, mstats))
        report["arms"][arm] = {
            "n_params": n_params,
            "val_rmse_K_mean": float(np.mean(rmses)),
            "val_rmse_K_std": float(np.std(rmses)),
            "val_rmse_K_per_seed": [float(x) for x in rmses],
            "seed_to_seed_spread_K": float(np.std(rmses)),
            "val_loss_curves": curves,
        }
        if trainable:
            dec = [still_decreasing(c, rel_tol) for c in curves]
            known = [d for d in dec if d is not None]
            report["arms"][arm]["still_decreasing_per_seed"] = dec
            report["arms"][arm]["still_decreasing_at_final_epoch"] = (
                any(known) if known else None)
    report["still_decreasing_arms"] = [
        a for a, r in report["arms"].items()
        if r.get("still_decreasing_at_final_epoch")]
    report["still_decreasing_rel_tol"] = rel_tol
    return report


def plot_curves(report, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm, r in report["arms"].items():
        for i, c in enumerate(r.get("val_loss_curves", [])):
            if c:
                flag = " (still decreasing)" if r.get(
                    "still_decreasing_at_final_epoch") else ""
                ax.plot(range(1, len(c) + 1), c, marker=".",
                        color=f"C{list(report['arms']).index(arm)}",
                        label=f"{arm}{flag}" if i == 0 else None)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (normalized MSE)")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("Calibration: per-arm validation-loss curves")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["gru", "rssm", "transformer", "hybrid", "linear",
                             "ebm_only"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--family", default="calibration")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--out", default="calibration_report.json")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--rel-tol", type=float, default=0.0,
                    help="flag an arm if its final-epoch val-loss drop exceeds "
                         "this relative amount (default: any drop)")
    args = ap.parse_args()

    p2 = load_phase2_config()
    if args.data_dir:
        p2["data"]["data_dir"] = args.data_dir
    full_cfg = load_config()
    device = args.device or pick_device(p2["train"]["device"])
    report = calibrate(args.arms, args.seeds, p2, full_cfg, device,
                       args.epochs, args.max_batches, family=args.family,
                       rel_tol=args.rel_tol)
    report["curves_png"] = plot_curves(
        report, args.out.rsplit(".", 1)[0] + "_curves.png")
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({a: {k: r[k] for k in ("val_rmse_K_mean",
                                            "still_decreasing_at_final_epoch")
                          if k in r} for a, r in report["arms"].items()}, indent=2))
    if report["still_decreasing_arms"]:
        print("WARNING: val loss still decreasing at final epoch for:",
              ", ".join(report["still_decreasing_arms"]))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

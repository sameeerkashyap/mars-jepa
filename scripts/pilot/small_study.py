"""One independently restartable small-study task. Training never evaluates tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import traceback

import torch

from physics.surface import load_config
from scripts.pilot.train import load_phase2_config, save_run, train_one

RUN_FILES = ("model.pt", "info.json", "stats.json", "arch.json", "resolved_config.json")


def artifact_hashes(run_dir):
    return {name: hashlib.sha256((Path(run_dir) / name).read_bytes()).hexdigest()
            for name in RUN_FILES}


def verify_frozen_run(run_dir, source_sha256):
    path = Path(run_dir) / "FROZEN.json"
    if not path.exists():
        raise ValueError("Final evaluation requires a frozen study decision")
    decision = json.loads(path.read_text())
    if "runs" in decision:
        if decision["source_sha256"] != source_sha256:
            raise ValueError("Frozen source mismatch")
        run = decision["runs"].get(Path(run_dir).name, {})
        expected = run.get("artifacts_sha256", {})
        if set(expected) != set(RUN_FILES) or artifact_hashes(run_dir) != expected:
            raise ValueError("Frozen checkpoint or metadata mismatch")
    return decision


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def variant_config(path, variant, data_dir):
    p2 = load_phase2_config(path)
    if variant not in p2["study"]["variants"] + ["ebm_only"]:
        raise ValueError(f"Unknown study variant: {variant}")
    p2["data"]["data_dir"] = data_dir
    if variant == "jepa_no_latent":
        p2["arch"]["jepa"]["latent_weight"] = 0.0
    return p2, "jepa" if variant == "jepa_no_latent" else variant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["benchmark", "train", "eval"])
    ap.add_argument("--config", default="data/specification/pilot/small_study.yaml")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / f"{args.mode}.log").open("a", buffering=1)
    console = sys.stdout
    sys.stdout = Tee(console, log)
    p2, arm = variant_config(args.config, args.variant, args.data_dir)
    if (args.seed not in p2["study"]["seeds"]
            or not 1 <= args.epochs <= p2["study"]["max_epochs"]):
        ap.error("Seed or epoch count is outside the registered small study")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    status_file = args.out / f"{args.mode}-status.json"
    status = {"mode": args.mode, "variant": args.variant, "seed": args.seed,
              "device": args.device, "source_sha256": os.environ.get("MARSWORLD_SOURCE_SHA256"),
              "status": "running"}
    write_json(status_file, status)
    result = {}
    try:
        if args.mode == "eval":
            # Test access is isolated to this explicit, final stage.
            verify_frozen_run(args.out, status["source_sha256"])
            from scripts.pilot.eval import evaluate

            p2 = json.loads((args.out / "resolved_config.json").read_text())
            tiers = {}
            for tier in ("tier1", "tier2"):
                tiers[tier] = evaluate([str(args.out)], args.data_dir, "main",
                                       f"test_{tier}", p2, load_config(), args.device)
            # Existing metrics use NaN for windows with no polar winter coverage.
            (args.out / "eval.json").write_text(json.dumps({"tiers": tiers}, indent=2))
            result = {"tiers": tiers}
        else:
            benchmark = args.mode == "benchmark"
            model, stats, info = train_one(
                arm, "off", p2["study"]["fraction"], args.seed, 0.0,
                p2, load_config(), args.device,
                epochs=1 if benchmark else args.epochs,
                max_batches=7 if benchmark else None, verbose=True, record_curve=True,
                calibration_mode=True, subset_seed=p2["study"]["subset_seed"],
                checkpoint_dir=None if benchmark or arm == "ebm_only" else args.out,
                resume=not benchmark)
            info.update(variant=args.variant, val_rmse_K=math.sqrt(info["val_loss"])
                        * stats["T_s_std"], torch_version=torch.__version__,
                        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / (1024**2 if sys.platform == "darwin" else 1024),
                        cuda_peak_mib=torch.cuda.max_memory_allocated() / 1024**2
                        if args.device == "cuda" else 0.0)
            if benchmark:
                steady_batch = statistics.median(info["batch_seconds"][2:])
                n_train_batches = math.ceil(info["train_windows"] / p2["train"]["batch_size"])
                info["projected_epoch_seconds"] = (
                    steady_batch * n_train_batches + info["validation_seconds"][0])
                write_json(args.out / "benchmark.json", info)
            else:
                save_run(str(args.out), model, stats, info, p2)
                info["artifacts_sha256"] = artifact_hashes(args.out)
            result = info
        status["status"] = "ok"
    except Exception:
        status.update(status="error", error=traceback.format_exc())
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        write_json(status_file, status)
        print("RESULT_JSON=" + json.dumps({"status": status, "result": result}), flush=True)
        sys.stdout = console
        log.close()


if __name__ == "__main__":
    main()

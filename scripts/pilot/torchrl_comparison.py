"""Train three TorchRL seeds, freeze validation checkpoints, then evaluate once."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics

from scripts.pilot.nautilus_study import ROOT, Study
from scripts.pilot.small_study import RUN_FILES, write_json
from scripts.pilot.train import load_phase2_config

CONFIG = ROOT / "data/specification/pilot/torchrl_comparison.yaml"
VARIANT = "torchrl_dreamer"
SEEDS = (0, 1, 2)
CONTROLS = ("gru", "jepa_no_latent", "jepa", "linear")
METRIC = "polar_winter_S_pinned"


def successful_records(state, mode):
    return [job["record"] for job in state["jobs"].values()
            if job.get("record", {}).get("status", {}).get("status") == "ok"
            and job["record"]["status"]["mode"] == mode]


class TorchRLComparison(Study):
    def __init__(self, state_dir, study_id, baseline_state, previous_state):
        super().__init__(state_dir, study_id)
        config = load_phase2_config(CONFIG)
        config["data"]["data_dir"] = "/results/pilot-20261109/data"
        self.spec = config["study"]
        if (self.spec["variants"] != [VARIANT] or self.spec["seeds"] != list(SEEDS)
                or self.spec["max_epochs"] != 20):
            raise ValueError("Comparison requires TorchRL, seeds 0/1/2, and a 20-epoch cap")
        baseline_bytes = Path(baseline_state).read_bytes()
        baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
        self.baseline = json.loads(baseline_bytes)
        if self.baseline["phase"] != "complete":
            raise ValueError("Baseline study must be complete")
        baseline_fits = successful_records(self.baseline, "train")
        if not baseline_fits:
            raise ValueError("Baseline training metadata is missing")
        self.reference = baseline_fits[0]["result"]["identity"]
        baseline_tests = successful_records(self.baseline, "eval")
        present = {(r["status"]["variant"], r["status"]["seed"]) for r in baseline_tests}
        if len(present) != len(baseline_tests):
            raise ValueError("Baseline test results contain duplicate model seeds")
        if not {(v, s) for v in CONTROLS for s in SEEDS} <= present:
            raise ValueError("Baseline test results must cover all control seeds")
        for record in baseline_fits:
            identity = record["result"].get("identity")
            if identity:
                self.check_data(identity, config)
        if "comparison_config" in self.state:
            if (self.state["comparison_config"] != config
                    or self.state["baseline_sha256"] != baseline_hash):
                raise ValueError("Comparison configuration or baseline snapshot changed")
        else:
            previous = json.loads(Path(previous_state).read_text())
            if (previous["phase"] != "trained"
                    or previous["prior_study"] != self.baseline["study_id"]):
                raise ValueError("Previous cost ledger must include the completed baseline")
            pilot = successful_records(previous, "train")
            if (len(pilot) != 1 or pilot[0]["result"]["identity"]["config"]["arch"][VARIANT]
                    != config["arch"][VARIANT]):
                raise ValueError("TorchRL architecture must match the five-epoch pilot")
            for device in ("cpu", "cuda"):
                jobs = [j for j in previous["jobs"].values() if j["device"] == device]
                if any("charged_seconds" not in j or j.get("accounting_unresolved") for j in jobs):
                    raise ValueError("Previous jobs need resolved allocation charges before importing their costs")
                self.state["jobs"]["previous-" + device] = {
                    "device": device, "reserved_seconds": 0,
                    "charged_seconds": sum(j["charged_seconds"] for j in jobs)}
            self.state.update(images=previous["images"], comparison_config=config,
                              baseline_sha256=baseline_hash, phase="training")
            (state_dir / "baseline-state.json").write_bytes(baseline_bytes)
            self.save()

    def check_data(self, identity, config):
        for key in ("fraction", "subset_seed", "subset_indices", "observed_sha256", "data_sha256"):
            if identity[key] != self.reference[key]:
                raise ValueError(f"Training data mismatch: {key}")
        if (identity["fraction"] != 0.3 or identity["subset_seed"] != 0
                or len(identity["subset_indices"]) != 30
                or identity["config"]["protocol"] != config["protocol"]
                or any(identity["config"]["train"][k] != config["train"][k]
                       for k in ("batch_size", "lr", "weight_decay", "grad_clip", "early_stop"))):
            raise ValueError("Training protocol or early stopping differs from the baseline")

    def manifest(self, task):
        name, manifest = super().manifest(task)
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        container["env"].extend({"name": k, "value": v} for k, v in {
            "MARSWORLD_CONFIG": "data/specification/pilot/torchrl_comparison.yaml",
            "MARSWORLD_TORCH_VERSION": "2.10.0",
            "MARSWORLD_EXTRA_REQUIREMENTS": "k8s/nautilus/requirements-torchrl.txt",
        }.items())
        if task[0] == "train":
            container["args"][0] = (
                "python - <<'OUTPUT_GUARD'\nimport os\nfrom pathlib import Path\n"
                "output = Path(os.environ['MARSWORLD_RUN_DIR'])\n"
                "assert not output.exists(), f'Fresh output directory required: {output}'\n"
                "OUTPUT_GUARD\n" + container["args"][0])
        return name, manifest

    def freeze(self, records):
        runs = {}
        for record in records:
            status, info = record["status"], record["result"]
            seed = status["seed"]
            if (status["status"] != "ok" or status["mode"] != "train"
                    or status["variant"] != VARIANT or seed not in SEEDS
                    or status["source_sha256"] != self.state["source_sha256"]
                    or info["identity"]["source"] != self.state["source_sha256"]
                    or info["identity"]["seed"] != seed or info["identity"]["arm"] != VARIANT
                    or info["identity"]["config"]["arch"][VARIANT]
                    != self.state["comparison_config"]["arch"][VARIANT]
                    or info["identity"]["config"]["study"]["max_epochs"] != 20):
                raise ValueError("Training result identity mismatch")
            self.check_data(info["identity"], self.state["comparison_config"])
            if (not 1 <= info["best_epoch"] <= info["epochs_completed"] <= 20
                    or not (info["epochs_completed"] == 20 or info["early_stopped"])
                    or not math.isfinite(info["val_rmse_K"])
                    or set(info["artifacts_sha256"]) != set(RUN_FILES)):
                raise ValueError("Invalid validation checkpoint metadata")
            key = f"{VARIANT}_s{seed}"
            if key in runs:
                raise ValueError("Duplicate training seed")
            runs[key] = {k: info[k] for k in ("best_epoch", "epochs_completed", "val_rmse_K",
                                             "artifacts_sha256")}
        if set(runs) != {f"{VARIANT}_s{s}" for s in SEEDS}:
            raise ValueError("All three training seeds must finish before test evaluation")
        decision = {"source_sha256": self.state["source_sha256"],
                    "baseline_sha256": self.state["baseline_sha256"], "seeds": list(SEEDS),
                    "selection": "lowest validation MSE", "metric": METRIC, "runs": runs}
        if "decision" in self.state and self.state["decision"] != decision:
            raise ValueError("Frozen decision changed")
        self.state.update(decision=decision, phase="frozen")
        write_json(self.state_dir / "FROZEN.json", decision)
        self.save()

    def run(self):
        if self.state["phase"] == "complete":
            return
        self.prepare()
        training = self.batch([("train", VARIANT, seed, 20, "cpu") for seed in SEEDS])
        self.freeze(training)
        evaluated = self.batch([("eval", VARIANT, seed, 20, "cpu") for seed in SEEDS])
        self.report(evaluated)
        self.state["phase"] = "complete"
        self.save()

    def report(self, records):
        rows, scores = [], {}
        for record in successful_records(self.baseline, "eval") + records:
            variant, seed = record["status"]["variant"], record["status"]["seed"]
            tiers = record["result"]["tiers"]
            values = [tiers[t]["intervention_response_rmse"][METRIC] for t in ("tier1", "tier2")]
            values = [v if math.isfinite(v) else None for v in values]
            rows.append({"variant": variant, "seed": seed, "tier1": values[0], "tier2": values[1]})
            scores[variant, seed] = values
        paired = {}
        for control in CONTROLS:
            paired[control] = {}
            for i, tier in enumerate(("tier1", "tier2")):
                seeds = [s for s in SEEDS if scores[VARIANT, s][i] is not None
                         and scores[control, s][i] is not None]
                differences = [scores[VARIANT, s][i] - scores[control, s][i] for s in seeds]
                paired[control][tier] = {
                    "seeds": seeds, "differences_K": differences,
                    "mean_K": statistics.mean(differences) if differences else None,
                    "std_K": statistics.stdev(differences) if len(differences) > 1 else None}
        note = ("Exploratory three-seed comparison on one fixed training subset. "
                "Baselines used PyTorch 2.5.1; TorchRL uses 2.10.0. "
                "No original ten-seed hypothesis claims.")
        write_json(self.state_dir / "report.json", {"metric": METRIC, "rows": rows,
                   "paired_torchrl_minus_control": paired, "limitations": note})
        show = lambda v: "missing" if v is None else f"{v:.4g}"
        lines = ["# TorchRL comparison", "", note, "", "South-pole control-pinned winter heating-response RMSE (K).",
                 "", "| Model | Seed | Tier 1 | Tier 2 |", "|---|---:|---:|---:|"]
        lines += [f"| {r['variant']} | {r['seed']} | {show(r['tier1'])} | {show(r['tier2'])} |" for r in rows]
        lines += ["", "Paired differences: TorchRL minus control; negative favors TorchRL.", "",
                  "| Control | Tier | Paired seeds | Mean difference (K) | Sample SD (K) |",
                  "|---|---|---:|---:|---:|"]
        lines += [f"| {control} | {tier} | {len(v['seeds'])} | {show(v['mean_K'])} | {show(v['std_K'])} |"
                  for control, tiers in paired.items() for tier, v in tiers.items()]
        lines += ["", f"Remaining CPU/GPU admission hours: {self.remaining()}"]
        (self.state_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--study-id", default="mars-torchrl-match-v1")
    ap.add_argument("--baseline-state", type=Path,
                    default=ROOT / "outputs/pilot/small-study-v2/state.json")
    ap.add_argument("--previous-state", type=Path,
                    default=ROOT / "outputs/pilot/torchrl-small-v1/state.json")
    args = ap.parse_args()
    if (not args.study_id.replace("-", "").isalnum() or not args.study_id.islower()
            or len(args.study_id) > 22):
        ap.error("Study ID must be at most 22 lowercase letters, digits, and hyphens")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error("This comparison already has a launcher")
        TorchRLComparison(args.state_dir, args.study_id, args.baseline_state, args.previous_state).run()


if __name__ == "__main__":
    main()

"""Run the staged study from a durable local process; at most two worker Jobs."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile
import time
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "data/specification/pilot/small_study.yaml"
VARIANTS = ("gru", "jepa_no_latent", "jepa", "linear")


def kubectl(*args, value=None, check=True):
    proc = subprocess.run(["rtk", "proxy", "kubectl", "--context", "nautilus", "-n",
                           "razvanlab", *args], input=json.dumps(value) if value else None,
                          text=True, capture_output=True, timeout=90)
    if check and proc.returncode:
        raise RuntimeError(proc.stderr or proc.stdout)
    return proc


def source_bundle():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("src", "scripts", "benchmarks", "data/specification", "k8s/nautilus",
                     "pyproject.toml", "README.md"):
            path = ROOT / name
            files = sorted(path.rglob("*")) if path.is_dir() else [path]
            for file in files:
                if file.is_file() and "__pycache__" not in file.parts and file.suffix != ".pyc":
                    archive.add(file, arcname=str(file.relative_to(ROOT)), recursive=False)
    return buffer.getvalue()


def pinned_image(repository, tag):
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repository}:pull"
    token = json.load(urllib.request.urlopen(url, timeout=30))["token"]
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repository}/manifests/{tag}",
        headers={"Authorization": f"Bearer {token}", "Accept":
                 "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        digest = response.headers["Docker-Content-Digest"]
    return f"{repository}@{digest}"


def choose_device(benchmarks, remaining_cpu, remaining_gpu):
    cpu = max(b["result"]["projected_epoch_seconds"] for b in benchmarks
              if b["status"]["device"] == "cpu")
    gpu = max((b["result"]["projected_epoch_seconds"] for b in benchmarks
               if b["status"]["device"] == "cuda"), default=float("inf"))
    # Reserve all 12 fits through 20 epochs, startup, and serial final evaluation.
    cpu_projection = (12 * (20 * cpu + 300) + 13 * (128 * cpu + 300)) / 3600 * 2
    gpu_projection = (12 * (20 * gpu + 300)) / 3600
    if cpu / gpu >= 2 and gpu_projection <= remaining_gpu and (
            2 * gpu_projection + 13 * (128 * cpu + 300) / 3600 * 2 <= remaining_cpu):
        return "cuda"
    if cpu_projection <= remaining_cpu:
        return "cpu"
    raise RuntimeError("Projected study exceeds remaining compute budget; no pilot admitted")


def pilot_passes(results, improvement=0.05):
    linear = results["linear"]["result"]["val_rmse_K"]
    return any(results[v]["result"]["val_rmse_K"] <= (1 - improvement) * linear
               for v in ("gru", "jepa_no_latent", "jepa"))


class Study:
    def __init__(self, state_dir, study_id):
        self.state_dir = state_dir
        self.path = state_dir / "state.json"
        self.study_id = study_id
        self.spec = yaml.safe_load(CONFIG.read_text())["study"]
        if self.spec["variants"] != list(VARIANTS) or self.spec["seeds"] != [0, 1, 2]:
            raise ValueError("This runner implements the fixed four-variant, three-seed study")
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "study_id": study_id, "jobs": {}, "images": {}, "phase": "benchmark"}
        if self.state["study_id"] != study_id:
            raise ValueError("State directory belongs to a different study")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.path)

    def remaining(self):
        cpu = gpu = 0.0
        for job in self.state["jobs"].values():
            seconds = job.get("charged_seconds", max(job["reserved_seconds"],
                                                     time.time() - job.get("created", time.time())))
            cpu += seconds * 2 / 3600
            if job["device"] == "cuda":
                gpu += seconds / 3600
        return self.spec["cpu_hours"] - cpu, self.spec["gpu_hours"] - gpu

    def prepare(self):
        if "source_sha256" not in self.state:
            archive = source_bundle()
            digest = hashlib.sha256(archive).hexdigest()
            (self.state_dir / "source.tar.gz").write_bytes(archive)
            self.state.update(source_sha256=digest,
                              configmap=f"mars-small-source-{digest[:12]}")
            self.save()
        archive = (self.state_dir / "source.tar.gz").read_bytes()
        if hashlib.sha256(archive).hexdigest() != self.state["source_sha256"]:
            raise ValueError("Source archive changed")
        cm = {"apiVersion": "v1", "kind": "ConfigMap", "immutable": True,
              "metadata": {"name": self.state["configmap"], "labels": {
                  "app": "marsworld-small-study"}},
              "binaryData": {"source.tar.gz": base64.b64encode(archive).decode()}}
        kubectl("apply", "-f", "-", value=cm)
        for device, repo, tag in (("cpu", "library/python", "3.11-slim-bookworm"),
                                  ("cuda", "pytorch/pytorch", "2.5.1-cuda12.4-cudnn9-runtime")):
            if device not in self.state["images"]:
                self.state["images"][device] = pinned_image(repo, tag)
                self.save()
        self.save()

    def manifest(self, task):
        mode, variant, seed, epochs, device = task
        name = f"{self.study_id}-{mode[:3]}-{variant.replace('_', '-')}-s{seed}-e{epochs}-{device}"
        manifest = yaml.safe_load((ROOT / "k8s/nautilus/small-study-job.yaml").read_text())
        manifest["metadata"]["name"] = name
        manifest["metadata"]["labels"]["study"] = self.study_id
        pod = manifest["spec"]["template"]["spec"]
        pod["volumes"][1]["configMap"]["name"] = self.state["configmap"]
        container = pod["containers"][0]
        container["image"] = self.state["images"][device]
        if device == "cuda":
            for field in ("requests", "limits"):
                container["resources"][field]["nvidia.com/gpu"] = "1"
        else:
            pod["affinity"] = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{"matchExpressions": [{
                    "key": "feature.node.kubernetes.io/pci-10de.present", "operator": "NotIn",
                    "values": ["true"]}]}]}}}
        run = f"/results/{self.study_id}/{variant}_s{seed}"
        if mode == "benchmark":
            run = f"/results/{self.study_id}/benchmarks/{variant}_{device}"
        env = {"MARSWORLD_MODE": mode, "MARSWORLD_VARIANT": variant,
               "MARSWORLD_SEED": str(seed), "MARSWORLD_EPOCHS": str(epochs),
               "MARSWORLD_DEVICE": device, "MARSWORLD_SOURCE_SHA256": self.state["source_sha256"],
               "MARSWORLD_RUN_DIR": run, "MARSWORLD_TASK_DIR": f"/results/{self.study_id}/jobs/{name}"}
        container["env"].extend({"name": k, "value": v} for k, v in env.items())
        container["args"][0] = ('mkdir -p "$MARSWORLD_TASK_DIR"\n' + container["args"][0])
        if mode == "eval":
            frozen = json.dumps(self.state["decision"])
            container["args"][0] = (
                'python -c \'import os,pathlib; pathlib.Path(os.environ["MARSWORLD_RUN_DIR"], '
                '"FROZEN.json").write_text(os.environ["MARSWORLD_DECISION"])\'\n'
                + container["args"][0])
            container["env"].append({"name": "MARSWORLD_DECISION", "value": frozen})
        return name, manifest

    def collect(self, name):
        job = self.state["jobs"][name]
        while True:
            remote = json.loads(kubectl("get", "job", name, "-o", "json").stdout)
            status = remote.get("status", {})
            elapsed = time.time() - job["created"]
            terminal = any(c["type"] in ("Complete", "Failed") and c["status"] == "True"
                           for c in status.get("conditions", []))
            if terminal:
                job["charged_seconds"] = elapsed
                output = kubectl("logs", f"job/{name}", "--pod-running-timeout=5s", check=False)
                logs = output.stdout + getattr(output, "stderr", "")
                (self.state_dir / f"{name}.log").write_text(logs)
                records = [line[len("RESULT_JSON="):] for line in logs.splitlines()
                           if line.startswith("RESULT_JSON=")]
                if not records:
                    job["error"] = status
                    raise RuntimeError(f"{name} ended without a result; inspect retained Job and local log")
                record = json.loads(records[-1])
                job["startup_seconds"] = max(0.0, elapsed - record["status"]["wall_seconds"])
                job["record"] = record
                if record["status"]["status"] != "ok" or not status.get("succeeded"):
                    raise RuntimeError(f"{name} failed: {record['status']}")
                print(f"DONE {name}: {elapsed:.0f}s", flush=True)
                return record
            time.sleep(10)

    def admit(self, tasks):
        cpu_cost = gpu_cost = 0.0
        records = [j["record"] for j in self.state["jobs"].values() if "record" in j]
        startup = max((j.get("startup_seconds", 0) for j in self.state["jobs"].values()),
                      default=300)
        for task in tasks:
            mode, variant, seed, epochs, device = task
            if mode == "benchmark":
                continue
            name, _ = self.manifest(task)
            if self.state["jobs"].get(name, {}).get("record", {}).get("status", {}).get("status") == "ok":
                continue
            benchmarks = [r["result"]["projected_epoch_seconds"] for r in records
                          if r["status"]["mode"] == "benchmark" and r["status"]["device"] == device]
            epoch = max(benchmarks, default=60)
            matching = [r["result"] for r in records if r["status"]["mode"] == "train"
                        and r["status"]["variant"] == variant and r["status"]["seed"] == seed]
            done = 0
            if matching:
                info = matching[-1]
                done = info.get("epochs_completed", 0)
                epoch = max(info.get("epoch_seconds", [epoch])[-3:])
                if info.get("early_stopped"):
                    done = epochs
            seconds = (128 * epoch if mode == "eval" else max(0, epochs - done) * epoch)
            if variant == "ebm_only" and mode == "train":
                seconds = epoch
            hours = (seconds * 1.5 + max(startup, 60)) / 3600
            cpu_cost += 2 * hours
            if device == "cuda":
                gpu_cost += hours
        cpu, gpu = self.remaining()
        if cpu_cost > cpu or gpu_cost > gpu:
            raise RuntimeError(f"Stage not admitted: projected CPU/GPU hours {cpu_cost:.2f}/{gpu_cost:.2f} "
                               f"exceed remaining {cpu:.2f}/{gpu:.2f}")

    def batch(self, tasks):
        self.admit(tasks)
        results = []
        for start in range(0, len(tasks), 2):
            names = []
            chunk = tasks[start:start + 2]
            for task in chunk:
                cpu, gpu = self.remaining()
                mode, _, _, _, device = task
                reservation = cpu * 3600 / (2 * len(chunk))
                if device == "cuda":
                    reservation = min(reservation, gpu * 3600 / len(chunk))
                if mode == "benchmark":
                    cap = self.spec["benchmark_gpu_hours"] * 3600 / 2 if device == "cuda" else (
                        self.spec["benchmark_cpu_hours"] * 3600 / 4)
                    reservation = min(reservation, cap)
                name, manifest = self.manifest(task)
                old = self.state["jobs"].get(name)
                if old and old.get("record", {}).get("status", {}).get("status") == "ok":
                    results.append(old["record"])
                    continue
                if old and ("error" in old or "charged_seconds" in old):
                    raise RuntimeError(f"Previous task {name} failed; inspect before spending more compute")
                if not old:
                    if reservation <= 0:
                        raise RuntimeError("Compute budget exhausted")
                    active = json.loads(kubectl("get", "jobs", "-l", "app=marsworld-small-study",
                                               "-o", "json").stdout)
                    if sum(not any(c["type"] in ("Complete", "Failed") and c["status"] == "True"
                                   for c in j.get("status", {}).get("conditions", []))
                           for j in active["items"]) >= 2:
                        raise RuntimeError("Two study jobs already active; no additional job admitted")
                    self.state["jobs"][name] = {"created": time.time(), "device": device,
                                                "reserved_seconds": reservation}
                    self.save()
                    kubectl("apply", "-f", "-", value=manifest)
                    print(f"START {name}", flush=True)
                names.append(name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.collect, name) for name in names]
                failures = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as error:
                        failures.append(error)
                    finally:
                        self.save()
                if failures:
                    raise failures[0]
        return results

    def run(self, cpu_only=False):
        cpu_only = cpu_only or self.state.get("cpu_only", False)
        if cpu_only:
            if self.state.get("device") not in (None, "cpu"):
                raise ValueError("Cannot switch an existing GPU training study to CPU")
            self.state["cpu_only"] = True
            self.save()
        self.prepare()
        devices = ("cpu",) if cpu_only else ("cpu", "cuda")
        benchmarks = self.batch([("benchmark", v, 0, 1, d)
                                 for d in devices for v in ("gru", "jepa")])
        if "device" not in self.state:
            self.state["device"] = choose_device(benchmarks, *self.remaining())
            self.save()
        device = self.state["device"]
        print(f"Selected training device: {device}; remaining CPU/GPU hours: {self.remaining()}",
              flush=True)
        pilot_epochs, max_epochs = self.spec["pilot_epochs"], self.spec["max_epochs"]
        pilot = self.batch([("train", v, 0, pilot_epochs, device) for v in VARIANTS])
        self.batch([("train", "ebm_only", 0, pilot_epochs, "cpu")])
        if "decision" not in self.state:
            passed = pilot_passes({r["status"]["variant"]: r for r in pilot},
                                   self.spec["min_rmse_improvement"])
            self.state["decision"] = {"expanded": passed, "seeds": [0, 1, 2] if passed else [0],
                                       "criterion": "neural validation RMSE at least 5% below linear",
                                       "source_sha256": self.state["source_sha256"]}
            self.save()
        decision = self.state["decision"]
        if decision["expanded"]:
            self.batch([("train", v, seed, max_epochs, device)
                        for seed in decision["seeds"] for v in VARIANTS])
        self.state["phase"] = "frozen"
        self.save()
        final = self.batch([("eval", v, seed, max_epochs if decision["expanded"] else pilot_epochs, "cpu")
                            for seed in decision["seeds"] for v in VARIANTS]
                           + [("eval", "ebm_only", 0, pilot_epochs, "cpu")])
        self.report(final)
        self.state["phase"] = "complete"
        self.save()

    def report(self, records):
        lines = ["# Exploratory JEPA study", "", json.dumps(self.state["decision"]), "",
                 "No original ten-seed hypothesis claims. Seeds share one fixed training subset.",
                 "", "| Variant | Seed | Tier-1 south pinned RMSE | Tier-2 south pinned RMSE |",
                 "|---|---:|---:|---:|"]
        by_variant = {}
        for record in records:
            variant, seed = record["status"]["variant"], record["status"]["seed"]
            tiers = record["result"]["tiers"]
            scores = [tiers[t]["intervention_response_rmse"]["polar_winter_S_pinned"]
                      for t in ("tier1", "tier2")]
            lines.append(f"| {variant} | {seed} | {scores[0]:.4g} | {scores[1]:.4g} |")
            by_variant.setdefault(variant, {})[seed] = scores[1]
        lines += ["", "Paired Tier-2 differences (JEPA minus comparator; negative favors JEPA):"]
        for comparator in ("gru", "jepa_no_latent"):
            diffs = [score - by_variant[comparator][seed]
                     for seed, score in by_variant["jepa"].items()]
            lines.append(f"- {comparator}: {diffs}; mean {statistics.mean(diffs):.4g} K")
        lines += ["", f"Remaining CPU/GPU hours: {self.remaining()}",
                  "", "Full metrics, validation curves, checkpoints, and settings remain on the PVC."]
        training = {}
        for job in self.state["jobs"].values():
            record = job.get("record", {})
            if record.get("status", {}).get("mode") == "train":
                key = record["status"]["variant"], record["status"]["seed"]
                training[key] = record["result"]
        lines += ["", "| Variant | Seed | Validation RMSE (K) | Best epoch | Epochs finished |",
                  "|---|---:|---:|---:|---:|"]
        for (variant, seed), info in training.items():
            lines.append(f"| {variant} | {seed} | {info['val_rmse_K']:.4g} | "
                         f"{info.get('best_epoch', '—')} | {info.get('epochs_completed', 0)} |")
        curves = {f"{variant}_s{seed}": info.get("val_curve", [])
                  for (variant, seed), info in training.items()}
        (self.state_dir / "validation_curves.json").write_text(json.dumps(curves, indent=2))
        (self.state_dir / "report.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--study-id", default="mars-small-jepa-v1")
    ap.add_argument("--cpu-only", action="store_true",
                    help="Use successful CPU benchmarks; inspect any GPU failure before resuming")
    args = ap.parse_args()
    if (not args.study_id.replace("-", "").isalnum() or not args.study_id.islower()
            or len(args.study_id) > 22):
        ap.error("Study ID must be at most 22 lowercase letters, digits, and hyphens")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / "launcher.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            ap.error("This study already has a launcher")
        Study(args.state_dir, args.study_id).run(cpu_only=args.cpu_only)


if __name__ == "__main__":
    main()

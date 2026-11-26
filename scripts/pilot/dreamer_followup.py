"""Run the validation and one-use confirmation stages of the Dreamer follow-up."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from scripts.pilot.dreamer_control import (
    DreamerControlStudy, FINAL_SEEDS, control_report, digest, freeze_confirmation,
)
from scripts.pilot.dreamer_control_forecast import readiness
from scripts.pilot.nautilus_study import Study, kubectl
from scripts.pilot.small_study import write_json


SOURCE = "cd7b517b33fe18b26df2ffa4fd1a972540b6316ea3d180491b6dfff2b324c556"
DATA = ["c5899c45029eff57bb7603802bd3e984edead82ade4824c1ed8c449207b65a2a",
        "78c1a3ebd9d9af9bab6f1220fc35c220c934ee16c5e77547596226df31bbba0d"]
ORIGINAL_CONFIG = "5cda42890b94a98a98616195a6137434e115dafd71ba67c940d36196fc4ce24a"
BASE = "/results/mars-dreamer-ctrl-v1"
DEV = (101, 102)


class FollowupStudy(DreamerControlStudy):
    def __init__(self, state_dir, previous_state):
        Study.__init__(self, Path(state_dir), "mars-dreamer-long-v1")
        raw = Path(previous_state).read_bytes()
        previous = json.loads(raw)
        previous_sha = hashlib.sha256(raw).hexdigest()
        if (previous["phase"] != "complete" or previous["study_id"] != "mars-dreamer-ctrl-v1"
                or previous["rung"] != 24 or previous["data_sha256"] != DATA
                or previous["source_sha256"] != SOURCE or previous["config_sha256"] != ORIGINAL_CONFIG
                or digest(previous["control_config"]) != ORIGINAL_CONFIG
                or previous["configmap"] != f"mars-control-source-{SOURCE[:12]}"
                or self.state.get("previous_sha256", previous_sha) != previous_sha):
            raise ValueError("Original pilot identity or ledger changed")
        self.original = previous["control_config"]
        if "previous_sha256" not in self.state:
            for device in ("cpu", "cuda"):
                jobs = [j for j in previous["jobs"].values() if j["device"] == device]
                if any(j.get("accounting_unresolved") or not math.isfinite(j["charged_seconds"])
                       or j["charged_seconds"] < 0 for j in jobs):
                    raise ValueError("Unresolved original job charges")
                self.state["jobs"]["previous-"+device] = dict(device=device, reserved_seconds=0,
                    charged_seconds=sum(j["charged_seconds"] for j in jobs))
            self.state.update(previous_sha256=previous_sha, images=previous["images"],
                              meter=previous["meter"], baseline=previous["baseline"],
                              source_sha256=SOURCE, data_sha256=DATA, configmap=previous["configmap"])
            self.save()
        self.select("long")

    def select(self, trial, horizon=3):
        self.trial = trial
        self.spec = {**self.original, "model_checkpoints": [500, 1000, 2000, 5000, 10000]}
        if trial.startswith("full"):
            self.spec["learner"] = {**self.spec["learner"], "sequence_length": 13}
        if trial.endswith("-more"):
            self.spec["behavior_checkpoints"] = [100, 200, 400, 800, 1600]
        self.recipe = dict(learning_rate=self.spec["model_learning_rates"]["fast"], horizon=horizon)

    def manifest(self, task):
        mode, variant, seed, rung, _ = task
        if mode == "final" and "decision" not in self.state:
            raise ValueError("Final job requires five frozen checkpoints")
        _, manifest = super().manifest(task)
        tag = f"{self.trial}-{mode}-{variant}-s{seed}"
        name = f"mars-drf-v1-{tag}"
        if self.trial == "long" and mode == "model":
            name, tag = f"mars-dreamer-long-v1-s{seed}", f"long-s{seed}"
        out = f"{BASE}/followup-v1/{tag}"
        manifest["metadata"]["name"] = name
        worker = manifest["spec"]["template"]["spec"]["containers"][0]
        env = {x["name"]: x["value"] for x in worker["env"]}
        old_out = env["MARSWORLD_TASK_DIR"]
        recipe = {**self.recipe, "horizon": int(variant[1:])} if mode == "actor" else self.recipe
        env.update(MARSWORLD_TASK_DIR=out, MARSWORLD_RUN_DIR=out, MARSWORLD_RECIPE=json.dumps(recipe),
                   MARSWORLD_FROZEN=json.dumps(self.state["decision"]) if mode == "final" else "{}")
        worker["env"] = [{"name": k, "value": v} for k, v in env.items()]
        command = worker["args"][0].replace(old_out, out)
        worker["args"][0] = command.replace("--base /results/"+self.study_id, "--base "+BASE)
        return name, manifest

    def adopt(self, task):
        mode, _, seed, _, _ = task
        name, _ = self.manifest(task)
        if name in self.state["jobs"]:
            return
        existing = (f"long-{seed}.json" if self.trial == "long" and mode == "model"
                    else name.removeprefix("mars-drf-v1-")+".job.json")
        path = self.state_dir / existing
        if not path.exists():
            return
        declared = json.loads(path.read_text())
        if declared["metadata"]["name"] != name:
            raise ValueError("Existing manifest has a different job identity")
        cached = self.state_dir / (name+".record")
        if cached.exists():
            retained = json.loads(cached.read_text())
            record = retained["record"]
            status = record["status"]
            seconds = retained["cpu_core_hours"]*3600/2
            if (retained["job"] != name or status["source_sha256"] != SOURCE
                    or status["device"] != "cpu" or status["allocated_cpu_cores"] != 2
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError("Invalid retained job identity or charge")
            job = dict(charged_seconds=seconds, reserved_seconds=0, record=record,
                       startup_seconds=max(0, seconds-status["wall_seconds"]),
                       allocation_evidence=f"Job start through completion from {cached.name}")
        else:
            remote = kubectl("get", "job", name, "-o", "json", check=False)
            if remote.returncode:
                if "notfound" in (remote.stdout+remote.stderr).lower().replace(" ", ""):
                    return
                raise RuntimeError(remote.stderr or remote.stdout)
            if json.loads(remote.stdout)["metadata"]["name"] != name:
                raise ValueError("Remote job identity changed")
            job = dict(created=time.time(), reserved_seconds=self.estimated_seconds(task)*1.5+300)
        self.state["jobs"][name] = dict(**job, device="cpu", submitted=True)
        self.save()

    def rows(self, mode, variant, seeds):
        tasks = [(mode, variant, seed, 24, "cpu") for seed in seeds]
        for task in tasks:
            self.adopt(task)
        rows = self.stage(tasks, self.trial+"-"+mode+"-"+variant)
        by_seed = {row["seed"]: row for row in rows}
        if (len(by_seed) != len(rows) or not set(by_seed) <= set(seeds)
                or (set(by_seed) != set(seeds) and not any(r.get("readiness_failed") for r in rows))):
            raise ValueError("Wrong or missing seed identity")
        for seed, row in by_seed.items():
            recipe = dict(self.recipe, horizon=int(variant[1:])) if mode == "actor" else self.recipe
            identity = dict(mode=mode, variant=variant, seed=seed, rung=24, data_sha256=DATA,
                            source_sha256=SOURCE, config_sha256=digest(self.spec), recipe_sha256=digest(recipe))
            if any(row.get(k) != v for k, v in identity.items()):
                raise ValueError("Model or policy result identity changed")
            model = [c for c in row["curves"] if c["phase"] == "model"]
            if [c["updates"] for c in model] != self.spec["model_checkpoints"]:
                raise ValueError("Incomplete model validation curve")
            selected = min(model, key=lambda c: c["validation"]["model_response"]["rmse"])["validation"]
            ready = readiness(selected, self.spec["readiness_response_improvement"])
            if mode == "model":
                if row["ready"] != ready or row["validation"] != selected:
                    raise ValueError("Model readiness disagrees with selected forecast")
            elif not ready:
                if (row.get("readiness_failed") is not True or row["actor_updates"] != 0
                        or row["validation"] != selected):
                    raise ValueError("Policy training crossed its model readiness gate")
                continue
            else:
                behavior = [c for c in row["curves"] if c["phase"] == "behavior"]
                if [c["updates"] for c in behavior] != self.spec["behavior_checkpoints"]:
                    raise ValueError("Incomplete policy validation curve")
                best = min(behavior, key=lambda c: c["validation_rmse_K"])
                observed = row["validation"]
                if (observed != best["rows"] or [r["condition_seed"] for r in observed] != list(range(5100, 5108))
                        or not math.isclose(row["validation_rmse_K"], best["validation_rmse_K"])
                        or not math.isclose(row["validation_rmse_K"],
                                            statistics.mean(r["tracking_rmse_K"] for r in observed))):
                    raise ValueError("Policy validation rows or score changed")
            name, manifest = self.manifest((mode, variant, seed, 24, "cpu"))
            env = manifest["spec"]["template"]["spec"]["containers"][0]["env"]
            output = next(e["value"] for e in env if e["name"] == "MARSWORLD_TASK_DIR")
            if (row["checkpoint"] != str(Path(output).relative_to(BASE)/"best.pt")
                    or len(row["checkpoint_sha256"]) != 64):
                raise ValueError(f"Unexpected checkpoint for {name}")
        return rows

    def estimated_seconds(self, task):
        mode, variant, _, _, _ = task
        meter = self.state["meter"]
        seconds = super().estimated_seconds(task)
        if mode == "final":
            return seconds
        # Actual end-to-end times include replay loading and every validation pass.
        records = [j["record"] for j in self.state["jobs"].values() if "record" in j]
        comparable = [r["status"]["wall_seconds"] for r in records
                      if r["result"].get("config_sha256") == digest(self.spec)
                      and r["result"].get("recipe_sha256") == digest(dict(
                          self.recipe, horizon=int(variant[1:]) if mode == "actor" else self.recipe["horizon"]))
                      and r["result"].get("mode") in (("actor", "confirm") if mode != "model" else ("model",))
                      and not r["result"].get("readiness_failed") and r["status"]["status"] == "ok"]
        if comparable:
            return max(comparable)
        if self.trial.startswith("full"):
            seconds += meter["model_update_seconds"]*10000*(13/8-1)
        return seconds

    def outcome(self, reason, *, final=None):
        report = dict(outcome=reason, final_test_opened=final is not None, passed=False,
                      selected_trial=self.trial, recipe=self.recipe, baseline=self.state["baseline"],
                      development_screen_K=self.screen, validation_scores=self.state.get("validation_scores", {}),
                      previous_sha256=self.state["previous_sha256"], source_sha256=SOURCE, data_sha256=DATA,
                      config_sha256=digest(self.spec), remaining_cpu_gpu_hours=self.remaining())
        if final is not None:
            report.update(final)
        write_json(self.state_dir/"report.json", report)
        self.state.update(phase="complete", outcome=reason)
        self.save()

    @property
    def screen(self):
        baseline = self.state["baseline"]
        return (1-self.spec["final_improvement"])*baseline["validation_means"][baseline["selected"]]

    def policy_score(self, horizon):
        rows = self.rows("actor", f"h{horizon}", DEV)
        if any(r.get("readiness_failed") or any(v["violations"] for v in r["validation"]) for r in rows):
            score = None
        else:
            score = statistics.mean(r["validation_rmse_K"] for r in rows)
        self.state.setdefault("validation_scores", {})[f"{self.trial}-h{horizon}"] = score
        self.save()
        return score

    def run(self):
        if self.state["phase"] == "complete":
            return
        for trial in ("long", "full"):
            self.select(trial)
            if all(r["ready"] for r in self.rows("model", "fast", DEV)):
                break
        else:
            return self.outcome("Both model recipes missed forecast readiness.")
        self.state["readiness_passed"] = True
        self.save()
        scores = {h: self.policy_score(h) for h in (3, 6)}
        feasible = {h: score for h, score in scores.items() if score is not None}
        if not feasible:
            return self.outcome("Neither horizon produced a feasible validation pair.")
        horizon = min(feasible, key=feasible.get)
        if feasible[horizon] > self.screen:
            self.select(trial+"-more", horizon)
            score = self.policy_score(horizon)
            if score is None or score > self.screen:
                return self.outcome("The longer behavior candidate missed the development screen.")
        else:
            self.select(trial, horizon)
        self.state["qualified_recipe"] = dict(trial=self.trial, **self.recipe)
        self.save()
        confirmed = self.rows("confirm", "selected", FINAL_SEEDS)
        if any(r.get("readiness_failed") for r in confirmed):
            return self.outcome("A fresh confirmation seed missed model readiness.")
        fields = ("mode", "variant", "seed", "rung", "data_sha256", "source_sha256", "config_sha256",
                  "recipe_sha256", "checkpoint", "checkpoint_sha256", "validation_rmse_K")
        runs = [{key: r[key] for key in fields} for r in confirmed]
        decision = freeze_confirmation(runs, SOURCE, digest(self.spec), DATA, self.state["baseline"], self.recipe)
        freeze_path = self.state_dir/"FROZEN.json"
        if (self.state.get("decision", decision) != decision
                or (freeze_path.exists() and json.loads(freeze_path.read_text()) != decision)):
            raise ValueError("Previously frozen final decision changed")
        self.state["decision"] = decision
        write_json(freeze_path, decision)
        self.save()
        final, = self.stage([("final", "frozen", 0, 24, "cpu")], "final-once")
        if final["frozen_sha256"] != digest(decision):
            raise ValueError("Held-out result did not use the frozen decision")
        report = control_report(final["records"])
        self.outcome("Five-seed held-out gate " + ("passed." if report["passed"] else "failed."), final=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--previous-state", required=True, type=Path)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir/"launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = FollowupStudy(args.state_dir, args.previous_state)
        try:
            study.run()
        except Exception as error:
            study.state["blocker"] = repr(error)
            study.save()
            raise


if __name__ == "__main__":
    main()

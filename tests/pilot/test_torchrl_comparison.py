"""Checkpoint freezing, fixed-seed orchestration, and paired reports without a cluster."""

import copy
import json

import pytest

from scripts.pilot.small_study import RUN_FILES, artifact_hashes, verify_frozen_run
from scripts.pilot.torchrl_comparison import (
    CONFIG, CONTROLS, METRIC, SEEDS, VARIANT, TorchRLComparison,
)
from scripts.pilot.train import load_phase2_config


def training_record(config, variant, seed, source="a" * 64):
    return {
        "status": {"mode": "train", "status": "ok", "variant": variant,
                   "seed": seed, "source_sha256": source},
        "result": {"best_epoch": 12, "epochs_completed": 16, "early_stopped": True,
                   "val_rmse_K": 2.0, "artifacts_sha256": {n: "f" * 64 for n in RUN_FILES},
                   "identity": {"arm": variant, "seed": seed, "source": source,
                                "fraction": 0.3, "subset_seed": 0,
                                "subset_indices": list(range(30)), "observed_sha256": "data",
                                "data_sha256": "manifest", "config": copy.deepcopy(config)}}}


def evaluation_record(variant, seed):
    error = 0.5 if variant == VARIANT else 1.0
    return {"status": {"mode": "eval", "status": "ok", "variant": variant, "seed": seed},
            "result": {"tiers": {tier: {"intervention_response_rmse": {METRIC: factor * error + seed}}
                                 for tier, factor in (("tier1", 1), ("tier2", 2))}}}


@pytest.fixture
def comparison(tmp_path):
    config = load_phase2_config(CONFIG)
    baseline = {"study_id": "baseline", "phase": "complete", "jobs": {}}
    for variant in CONTROLS:
        for seed in SEEDS:
            for mode, record in (("train", training_record(config, variant, seed)),
                                 ("eval", evaluation_record(variant, seed))):
                baseline["jobs"][f"{variant}-{seed}-{mode}"] = {"record": record}
    previous = {"phase": "trained", "prior_study": "baseline",
                "images": {"cpu": "cpu@sha256:abc", "cuda": "gpu@sha256:def"},
                "jobs": {"pilot": {"device": "cpu", "charged_seconds": 100,
                                    "record": training_record(config, VARIANT, 0)}}}
    baseline_path, previous_path = tmp_path / "baseline.json", tmp_path / "previous.json"
    baseline_path.write_text(json.dumps(baseline))
    previous_path.write_text(json.dumps(previous))
    study = TorchRLComparison(tmp_path / "state", "mars-test", baseline_path, previous_path)
    study.state.update(source_sha256="a" * 64, configmap="source")
    study.save()
    return study, baseline_path, previous_path


def test_comparison_matches_pilot_and_baseline_protocol(comparison):
    study, *_ = comparison
    config = study.state["comparison_config"]
    pilot = load_phase2_config("data/specification/pilot/torchrl_small.yaml")
    assert config["arch"][VARIANT] == pilot["arch"][VARIANT]
    assert config["protocol"] == {"K": 40, "H": 120, "stride": 120}
    assert config["train"]["early_stop"] == {"warmup": 5, "patience": 3, "min_delta": 0.001}
    assert config["study"]["seeds"] == [0, 1, 2] and config["study"]["max_epochs"] == 20
    assert 0 < study.remaining()[0] < 48


def test_comparison_rejects_unresolved_previous_charge(comparison, tmp_path):
    _, baseline, previous = comparison
    ledger = json.loads(previous.read_text())
    ledger["jobs"]["pilot"]["accounting_unresolved"] = True
    previous.write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="resolved allocation charges"):
        TorchRLComparison(tmp_path / "new-state", "mars-new", baseline, previous)
    assert json.loads(previous.read_text()) == ledger


def test_no_test_job_before_all_three_training_seeds_finish(comparison, monkeypatch):
    study, *_ = comparison
    calls = []
    monkeypatch.setattr(study, "prepare", lambda: None)

    def batch(tasks):
        calls.extend(tasks)
        assert all(t[0] == "train" for t in tasks), "test data touched before freezing"
        return [training_record(study.state["comparison_config"], VARIANT, s) for s in (0, 1)]

    monkeypatch.setattr(study, "batch", batch)
    with pytest.raises(ValueError, match="All three"):
        study.run()
    assert not (study.state_dir / "FROZEN.json").exists()
    assert len(calls) == 3


def test_freeze_then_evaluate_once_and_report_paired_differences(comparison, monkeypatch):
    study, *_ = comparison
    calls = []
    monkeypatch.setattr(study, "prepare", lambda: None)

    def batch(tasks):
        calls.append(tasks)
        if tasks[0][0] == "train":
            assert not (study.state_dir / "FROZEN.json").exists()
            return [training_record(study.state["comparison_config"], VARIANT, s) for s in SEEDS]
        assert study.state["phase"] == "frozen"
        frozen = json.loads((study.state_dir / "FROZEN.json").read_text())
        assert set(frozen["runs"]) == {f"{VARIANT}_s{s}" for s in SEEDS}
        return [evaluation_record(VARIANT, s) for s in SEEDS]

    monkeypatch.setattr(study, "batch", batch)
    study.run()
    assert study.state["phase"] == "complete" and len(calls) == 2
    study.run()
    assert len(calls) == 2
    report = json.loads((study.state_dir / "report.json").read_text())
    assert len(report["rows"]) == 15
    paired = report["paired_torchrl_minus_control"]["gru"]["tier2"]
    assert paired == {"seeds": [0, 1, 2], "differences_K": [-1.0] * 3, "mean_K": -1.0, "std_K": 0.0}
    assert "PyTorch 2.5.1" in report["limitations"]


def test_rejects_data_change_and_unfinished_epoch_budget(comparison):
    study, *_ = comparison
    records = [training_record(study.state["comparison_config"], VARIANT, s) for s in SEEDS]
    records[1]["result"]["identity"]["subset_indices"][0] = 99
    with pytest.raises(ValueError, match="data mismatch"):
        study.freeze(records)
    records[1] = training_record(study.state["comparison_config"], VARIANT, 1)
    records[1]["result"].update(epochs_completed=5, best_epoch=5, early_stopped=False)
    with pytest.raises(ValueError, match="Invalid validation"):
        study.freeze(records)


def test_resume_rejects_changed_baseline_snapshot(comparison):
    study, baseline, previous = comparison
    baseline.write_text(baseline.read_text() + " ")
    with pytest.raises(ValueError, match="baseline snapshot changed"):
        TorchRLComparison(study.state_dir, study.study_id, baseline, previous)


def test_recovers_report_from_durable_results_without_accessing_test_again(comparison, monkeypatch):
    study, *_ = comparison
    monkeypatch.setattr(study, "prepare", lambda: None)
    for seed in SEEDS:
        name, _ = study.manifest(("train", VARIANT, seed, 20, "cpu"))
        study.state["jobs"][name] = {
            "device": "cpu", "reserved_seconds": 0, "charged_seconds": 0,
            "record": training_record(study.state["comparison_config"], VARIANT, seed)}
    study.freeze([j["record"] for j in study.state["jobs"].values() if "record" in j])
    for seed in SEEDS:
        name, _ = study.manifest(("eval", VARIANT, seed, 20, "cpu"))
        study.state["jobs"][name] = {"device": "cpu", "reserved_seconds": 0,
                                      "charged_seconds": 0, "record": evaluation_record(VARIANT, seed)}
    study.save()
    resumed = TorchRLComparison(study.state_dir, study.study_id, comparison[1], comparison[2])
    monkeypatch.setattr(resumed, "prepare", lambda: None)
    resumed.run()
    assert resumed.state["phase"] == "complete"
    assert len(json.loads((study.state_dir / "report.json").read_text())["rows"]) == 15


def test_manifests_use_fixed_torchrl_runtime_and_no_time_limits(comparison):
    study, *_ = comparison
    study.freeze([training_record(study.state["comparison_config"], VARIANT, s) for s in SEEDS])
    for mode in ("train", "eval"):
        _, manifest = study.manifest((mode, VARIANT, 0, 20, "cpu"))
        assert "activeDeadlineSeconds" not in manifest["spec"]
        assert "ttlSecondsAfterFinished" not in manifest["spec"]
        pod = manifest["spec"]["template"]["spec"]
        assert "activeDeadlineSeconds" not in pod
        container = pod["containers"][0]
        env = {e["name"]: e["value"] for e in container["env"]}
        assert env["MARSWORLD_TORCH_VERSION"] == "2.10.0"
        assert env["MARSWORLD_CONFIG"].endswith("torchrl_comparison.yaml")
        assert container["resources"]["requests"]["cpu"] == "2"
        if mode == "train":
            assert "Fresh output directory required" in container["args"][0]
        else:
            assert json.loads(env["MARSWORLD_DECISION"])["runs"] == study.state["decision"]["runs"]


@pytest.mark.parametrize("filename", RUN_FILES)
def test_worker_rejects_checkpoint_or_metadata_changed_after_freeze(tmp_path, filename):
    run = tmp_path / "torchrl_dreamer_s0"
    run.mkdir()
    for name in RUN_FILES:
        (run / name).write_text("original " + name)
    decision = {"source_sha256": "source", "runs": {run.name: {"artifacts_sha256": artifact_hashes(run)}}}
    (run / "FROZEN.json").write_text(json.dumps(decision))
    assert verify_frozen_run(run, "source") == decision
    with pytest.raises(ValueError, match="source mismatch"):
        verify_frozen_run(run, "different-source")
    (run / filename).write_text("changed")
    with pytest.raises(ValueError, match="checkpoint or metadata mismatch"):
        verify_frozen_run(run, "source")

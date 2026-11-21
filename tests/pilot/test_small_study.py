"""Small-study recovery, input timing, ablation, and resource gates."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.registry import build_model
from scripts.pilot import train
from scripts.pilot import nautilus_study
from scripts.pilot.nautilus_study import Study, choose_device, pilot_passes
from scripts.pilot.small_study import variant_config
from tests.pilot.fakedata import build_fake_dataset


@pytest.fixture
def config(tmp_path):
    root = build_fake_dataset(str(tmp_path / "data"), n_time=40)
    p2 = train.load_phase2_config("data/specification/pilot/small_study.yaml")
    p2["data"]["data_dir"] = root
    p2["protocol"] = {"K": 4, "H": 8, "stride": 8}
    p2["train"]["batch_size"] = 4
    return p2


@pytest.mark.parametrize("arm", ["gru", "jepa", "linear"])
def test_resume_matches_uninterrupted_including_ema(config, tmp_path, arm):
    torch.set_num_threads(1)
    kw = dict(arm=arm, physics="off", fraction=0.3, seed=2, lam=0.0,
              p2=config, full_cfg=train.load_config(), device="cpu", subset_seed=0,
              max_batches=2, record_curve=True, calibration_mode=True)
    train.train_one(**kw, epochs=2, checkpoint_dir=tmp_path / "full")
    train.train_one(**kw, epochs=1, checkpoint_dir=tmp_path / "resumed")
    train.train_one(**kw, epochs=2, checkpoint_dir=tmp_path / "resumed", resume=True)
    full = torch.load(tmp_path / "full/latest.pt", weights_only=False)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    assert full["val_curve"] == resumed["val_curve"]
    assert full["epoch"] == resumed["epoch"] == 2
    for key in full["model"]:
        assert torch.equal(full["model"][key], resumed["model"][key]), key
    assert torch.equal(full["loader"], resumed["loader"])
    with pytest.raises(ValueError, match="mismatch"):
        train.train_one(**dict(kw, subset_seed=1), epochs=3,
                        checkpoint_dir=tmp_path / "resumed", resume=True)
    assert not list((tmp_path / "resumed").glob("*.tmp"))


def test_subset_is_independent_of_initialization_seed(config):
    kw = dict(arm="linear", physics="off", fraction=0.3, lam=0.0, p2=config,
              full_cfg=train.load_config(), device="cpu", subset_seed=0, epochs=0)
    _, stats0, _ = train.train_one(**kw, seed=0)
    _, stats2, _ = train.train_one(**kw, seed=2)
    assert stats0 == stats2


def test_early_stop_exports_best_and_does_not_restart(config, tmp_path, monkeypatch):
    config["train"]["early_stop"] = {"warmup": 1, "patience": 2, "min_delta": 0.001}
    values = iter([1.0, 0.5, 0.7, 0.8])
    monkeypatch.setattr(train, "_eval_val", lambda *args: next(values))
    kw = dict(arm="jepa", physics="off", fraction=0.3, seed=0, lam=0,
              p2=config, full_cfg=train.load_config(), device="cpu", max_batches=1,
              subset_seed=0, checkpoint_dir=tmp_path / "run", epochs=20)
    model, _, info = train.train_one(**kw)
    assert info["early_stopped"] and info["best_epoch"] == 2
    assert info["epochs_completed"] == 4 and info["val_loss"] == 0.5
    ck = torch.load(tmp_path / "run/latest.pt", weights_only=False)
    for key, value in model.state_dict().items():
        assert torch.equal(value, ck["best_model"][key])
    _, _, resumed = train.train_one(**kw, resume=True)
    assert resumed["epochs_completed"] == 4


def test_jepa_ablation_only_changes_loss_weight(config):
    path = "data/specification/pilot/small_study.yaml"
    normal, arm = variant_config(path, "jepa", config["data"]["data_dir"])
    ablated, ablated_arm = variant_config(path, "jepa_no_latent", config["data"]["data_dir"])
    assert arm == ablated_arm == "jepa"
    assert ablated["arch"]["jepa"]["latent_weight"] == 0.0
    ablated["arch"]["jepa"]["latent_weight"] = 1.0
    assert normal == ablated


@pytest.mark.parametrize("arm", ["gru", "jepa", "linear", "ebm_only"])
def test_actions_are_applied_to_previous_interval(config, arm):
    stats = {"T_s_mean": 200, "T_s_std": 30, "S_mean": 100, "S_std": 80,
             "F_a_mean": 2, "F_a_std": 2}
    model = build_model(arm, stats, 64, config["arch"]).eval()
    if arm == "linear":
        model.bF.data.fill_(1)
    ts = torch.full((1, 4, 64), 250.0)
    sun = torch.full((1, 12, 64), 100.0)
    fa = torch.zeros(1, 12)
    lat = torch.linspace(-1.4, 1.4, 64)
    with torch.no_grad():
        baseline, _ = model.rollout(ts, sun, fa, lat)
        fa[:, 4] = 5
        changed, _ = model.rollout(ts, sun, fa, lat)
        assert torch.equal(baseline[:, 0], changed[:, 0])
        assert not torch.allclose(baseline[:, 1], changed[:, 1])
        fa.zero_()
        fa[:, 3] = 5
        previous, _ = model.rollout(ts, sun, fa, lat)
        assert not torch.allclose(baseline[:, 0], previous[:, 0])


def test_validation_gate_and_budget_admission():
    result = lambda x: {"result": {"val_rmse_K": x}}
    pilot = {v: result(10) for v in ("gru", "jepa_no_latent", "jepa", "linear")}
    assert not pilot_passes(pilot)
    pilot["jepa"] = result(9)
    assert pilot_passes(pilot)
    bench = [{"status": {"device": d}, "result": {"projected_epoch_seconds": t}}
             for d, t in (("cpu", 10), ("cpu", 8), ("cuda", 2), ("cuda", 3))]
    assert choose_device(bench, 48, 12) == "cuda"
    assert choose_device(bench, 48, 0) == "cpu"
    assert choose_device(bench[:2], 48, 12) == "cpu"
    with pytest.raises(RuntimeError, match="budget"):
        choose_device(bench, 0, 0)


def test_job_resources_source_and_deadline(tmp_path):
    study = Study(tmp_path, "mars-test")
    study.state.update(configmap="test-source", source_sha256="a" * 64,
                       images={"cpu": "cpu@sha256:abc", "cuda": "gpu@sha256:def"})
    for device in ("cpu", "cuda"):
        name, manifest = study.manifest(("train", "jepa_no_latent", 0, 5, device))
        assert len(name) <= 63
        assert "activeDeadlineSeconds" not in manifest["spec"]
        assert "activeDeadlineSeconds" not in manifest["spec"]["template"]["spec"]
        assert "ttlSecondsAfterFinished" not in manifest["spec"]
        container = manifest["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["cpu"] == "2"
        assert ("nvidia.com/gpu" in container["resources"]["limits"]) == (device == "cuda")
        env = {e["name"]: e["value"] for e in container["env"]}
        assert container["command"] == ["/bin/sh", "-ec"]
        assert "MARSWORLD_MAX_SECONDS" not in env
        assert env["MARSWORLD_SOURCE_SHA256"] == "a" * 64
        assert "pilot-20261109/data" in container["args"][0]
        assert env["MARSWORLD_RUN_DIR"] == "/results/mars-test/jepa_no_latent_s0"
    study.state["jobs"] = {"cpu": {"device": "cpu", "reserved_seconds": 3600},
                           "gpu": {"device": "cuda", "reserved_seconds": 3600}}
    assert study.remaining() == (44, 11)
    study.state["jobs"] = {"spent": {"device": "cpu", "reserved_seconds": 48 * 1800}}
    with pytest.raises(RuntimeError, match="Stage not admitted"):
        study.admit([("train", "gru", 0, 5, "cpu")])


def test_config_preserves_legacy_defaults():
    legacy = train.load_phase2_config()
    small = train.load_phase2_config("data/specification/pilot/small_study.yaml")
    assert legacy["protocol"]["stride"] == 20
    assert "action_offset" not in legacy["arch"]["jepa"]
    assert small["protocol"]["stride"] == 120
    assert small["arch"]["jepa"]["embed"] == 48
    assert small["arch"]["jepa"]["action_offset"] == -1


def test_running_job_is_not_cancelled_when_estimate_is_exceeded(tmp_path, monkeypatch):
    study = Study(tmp_path, "mars-test")
    study.state["jobs"]["long-job"] = {"device": "cpu", "reserved_seconds": 1,
                                       "created": 0}
    running = {"status": {"active": 1}}
    finished = {"status": {"succeeded": 1, "conditions": [
        {"type": "Complete", "status": "True"}]}}
    states = iter([running, finished])
    calls = []
    record = {"status": {"status": "ok", "wall_seconds": 99}, "result": {}}

    def fake_kubectl(*args, **kwargs):
        calls.append(args)
        if args[0] == "get":
            return SimpleNamespace(stdout=json.dumps(next(states)))
        assert args[0] == "logs"
        return SimpleNamespace(stdout="RESULT_JSON=" + json.dumps(record))

    monkeypatch.setattr(nautilus_study, "kubectl", fake_kubectl)
    monkeypatch.setattr(nautilus_study.time, "time", lambda: 100)
    monkeypatch.setattr(nautilus_study.time, "sleep", lambda seconds: None)
    assert study.collect("long-job") == record
    assert study.state["jobs"]["long-job"]["charged_seconds"] == 100
    assert not any(call[0] in ("delete", "patch") for call in calls)


def test_failed_job_without_pod_logs_is_charged_and_recorded(tmp_path, monkeypatch):
    study = Study(tmp_path, "mars-test")
    study.state["jobs"]["evicted"] = {"device": "cuda", "reserved_seconds": 1,
                                      "created": 0}
    failed = {"status": {"failed": 1, "conditions": [
        {"type": "Failed", "status": "True"}]}}

    def fake_kubectl(*args, **kwargs):
        if args[0] == "get":
            return SimpleNamespace(stdout=json.dumps(failed))
        assert args[0] == "logs" and kwargs["check"] is False
        return SimpleNamespace(stdout="", stderr="pod was evicted", returncode=1)

    monkeypatch.setattr(nautilus_study, "kubectl", fake_kubectl)
    monkeypatch.setattr(nautilus_study.time, "time", lambda: 100)
    with pytest.raises(RuntimeError, match="without a result"):
        study.collect("evicted")
    job = study.state["jobs"]["evicted"]
    assert job["charged_seconds"] == 100 and job["error"] == failed["status"]
    assert "pod was evicted" in (tmp_path / "evicted.log").read_text()


def test_cpu_only_continuation_skips_gpu_and_persists(tmp_path, monkeypatch):
    study = Study(tmp_path, "mars-test")
    calls = []
    monkeypatch.setattr(study, "prepare", lambda: None)

    def batch(tasks):
        calls.extend(tasks)
        if tasks[0][0] == "train":
            raise RuntimeError("pilot reached")
        return [{"status": {"device": "cpu"},
                 "result": {"projected_epoch_seconds": 2}} for _ in tasks]

    monkeypatch.setattr(study, "batch", batch)
    with pytest.raises(RuntimeError, match="pilot reached"):
        study.run(cpu_only=True)
    assert study.state["device"] == "cpu"
    assert all(task[-1] == "cpu" for task in calls)
    assert json.loads(study.path.read_text())["cpu_only"] is True
    study.state["device"] = "cuda"
    with pytest.raises(ValueError, match="Cannot switch"):
        study.run(cpu_only=True)

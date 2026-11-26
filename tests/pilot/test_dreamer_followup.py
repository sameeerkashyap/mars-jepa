import json

import pytest

from scripts.pilot.dreamer_control import CONFIG, digest
from scripts.pilot.dreamer_followup import FollowupStudy, SOURCE, DATA


@pytest.fixture
def study(tmp_path):
    prior = tmp_path / "previous.json"
    config = json.loads(CONFIG.read_text())
    prior.write_text(json.dumps(dict(
        study_id="mars-dreamer-ctrl-v1", phase="complete", rung=24,
        jobs={"old": {"device": "cpu", "charged_seconds": 3600}},
        control_config=config, config_sha256=digest(config), source_sha256=SOURCE,
        configmap=f"mars-control-source-{SOURCE[:12]}", data_sha256=DATA,
        baseline={"selected": "simple_model", "validation_means": {"simple_model": 0.5819}},
        images={"cpu": "fixture@sha256:" + "a"*64},
        meter={"model_update_seconds": .1, "behavior_update_seconds": .1,
               "forecast_validation_seconds": 1., "collection_seconds": 5.},
    )))
    return FollowupStudy(tmp_path / "followup", prior)


def test_followup_uses_frozen_source_and_separate_paths(study):
    assert study.remaining() == (46., 12.)
    for seed in (101, 102):
        name, job = study.manifest(("model", "fast", seed, 24, "cpu"))
        assert name == f"mars-dreamer-long-v1-s{seed}"
        pod = job["spec"]["template"]["spec"]
        worker = pod["containers"][0]
        env = {x["name"]: x["value"] for x in worker["env"]}
        assert env["MARSWORLD_SOURCE_SHA256"] == SOURCE
        assert env["MARSWORLD_TASK_DIR"] == f"/results/mars-dreamer-ctrl-v1/followup-v1/long-s{seed}"
        assert "--base /results/mars-dreamer-ctrl-v1 " in worker["args"][0]
        assert "--config /dev/stdin" in worker["args"][0]
        config = json.loads(worker["args"][0].split("<<'PILOT_CONFIG'\n")[1].splitlines()[0])
        assert config["model_checkpoints"] == [500, 1000, 2000, 5000, 10000]
        assert "activeDeadlineSeconds" not in json.dumps(job)
    name, _ = study.manifest(("actor", "h6", 101, 24, "cpu"))
    assert name == "mars-drf-v1-long-actor-h6-s101"
    with pytest.raises(ValueError, match="frozen"):
        study.manifest(("final", "frozen", 0, 24, "cpu"))


def test_existing_model_records_are_charged_once_and_wrong_seed_is_rejected(study):
    task = ("model", "fast", 101, 24, "cpu")
    name, manifest = study.manifest(task)
    path = study.state_dir / "long-101.json"
    path.write_text(json.dumps(manifest))
    row = dict(seed=102, variant="fast", mode="model", rung=24, source_sha256=SOURCE,
               config_sha256=digest(study.spec), data_sha256=DATA)
    record = dict(result=row, status=dict(status="ok", wall_seconds=10, device="cpu",
                                         allocated_cpu_cores=2, source_sha256=SOURCE))
    (study.state_dir / (name+".record")).write_text(json.dumps(
        dict(job=name, cpu_core_hours=.25, record=record)))
    for _ in range(2):
        with pytest.raises(ValueError, match="identity"):
            study.rows("model", "fast", [101])
        assert study.remaining() == (45.75, 12.)


@pytest.mark.parametrize("extended_score,opens_final", [(0.6, False), (0.5, True)])
def test_development_screen_guards_confirmation_and_one_frozen_final(study, monkeypatch, extended_score, opens_final):
    calls = []
    def workers(tasks):
        results = []
        for task in tasks:
            mode, variant, seed, _, _ = task
            name, manifest = study.manifest(task)
            calls.append((study.trial, mode, variant, seed))
            worker = manifest["spec"]["template"]["spec"]["containers"][0]
            env = {x["name"]: x["value"] for x in worker["env"]}
            if mode == "final":
                frozen = json.loads(env["MARSWORLD_FROZEN"])
                assert [r["seed"] for r in frozen["runs"]] == list(range(201, 206))
                result = dict(frozen_sha256=digest(frozen), records=[
                    dict(seed=s, rows=[dict(condition_seed=c, tracking_rmse_K=.4, baseline_rmse_K=.6,
                        violations=0) for c in range(5200, 5220)]) for s in frozen["seeds"]])
            else:
                metrics = {key: {"rmse": 1.} for key in ("model_response", "model_reward", "model_temperature",
                    "linear_response", "zero_response", "linear_reward", "zero_reward", "persistence_temperature")}
                metrics["model_reward"] = metrics["model_response"] = {"rmse": .3}
                curves = [dict(phase="model", updates=u, validation=metrics) for u in study.spec["model_checkpoints"]]
                result = dict(mode=mode, variant=variant, seed=seed, rung=24, data_sha256=DATA,
                    source_sha256=SOURCE, config_sha256=digest(study.spec),
                    recipe_sha256=digest(json.loads(env["MARSWORLD_RECIPE"])), checkpoint_sha256="d"*64,
                    checkpoint=env["MARSWORLD_TASK_DIR"].split("mars-dreamer-ctrl-v1/")[1]+"/best.pt",
                    ready=True, curves=curves, validation=metrics, validation_rmse_K=.3)
                if mode != "model":
                    score = extended_score if study.trial.endswith("-more") else .59 if variant == "h6" else .61
                    rows = [dict(condition_seed=c, tracking_rmse_K=score, violations=0) for c in range(5100,5108)]
                    result.update(validation_rmse_K=score, validation=rows)
                    curves.extend(dict(phase="behavior", updates=u, validation_rmse_K=score, rows=rows)
                                  for u in study.spec["behavior_checkpoints"])
            results.append(dict(result=result))
        return results
    monkeypatch.setattr(study, "batch", workers)
    study.run()
    assert study.state["phase"] == "complete"
    assert [(v,s) for t,m,v,s in calls if t == "long-more" and m == "actor"] == [("h6",101), ("h6",102)]
    assert [s for _,m,_,s in calls if m == "confirm"] == (list(range(201,206)) if opens_final else [])
    assert sum(m == "final" for _,m,_,_ in calls) == int(opens_final)
    count = len(calls)
    study.run()
    assert len(calls) == count

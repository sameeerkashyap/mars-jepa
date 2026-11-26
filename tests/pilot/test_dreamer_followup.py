import json

import pytest

from scripts.pilot.dreamer_control import CONFIG, digest
from scripts.pilot.dreamer_followup import FollowupStudy, SOURCE, DATA


@pytest.fixture
def study(tmp_path, monkeypatch):
    import hashlib
    prior = tmp_path / "previous.json"
    config = json.loads(CONFIG.read_text())
    prior.write_text(json.dumps(dict(
        study_id="mars-dreamer-ctrl-v1", phase="complete", rung=24,
        jobs={"old": {"device": "cpu", "charged_seconds": 3600}},
        control_config=config, config_sha256=digest(config), source_sha256=SOURCE,
        configmap=f"mars-control-source-{SOURCE[:12]}", data_sha256=DATA,
        baseline={"selected": "simple_model", "validation_means": {"simple_model": 0.5819}},
        images={"cpu": "fixture@sha256:" + "a"*64},
        meter={"generation_condition_seconds": 2., "model_update_seconds": .1, "behavior_update_seconds": .1,
               "forecast_validation_seconds": 1., "collection_seconds": 5.},
    )))
    monkeypatch.setattr("scripts.pilot.dreamer_followup.ORIGINAL_LEDGER",
                        hashlib.sha256(prior.read_bytes()).hexdigest())
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


def test_all_followup_recipes_share_the_original_one_use_final(study):
    study.state['decision'] = {'fixture': True}
    for trial in ('long', 'full', 'long-more', 'full-more'):
        study.select(trial)
        name, manifest = study.manifest(('final', 'frozen', 0, 24, 'cpu'))
        worker = manifest['spec']['template']['spec']['containers'][0]
        env = {item['name']: item['value'] for item in worker['env']}
        assert name == 'mars-dreamer-ctrl-v1-fin-frozen-s0-e24-cpu'
        assert env['MARSWORLD_TASK_DIR'] == '/results/mars-dreamer-ctrl-v1/final-frozen-s0-r24'


@pytest.mark.parametrize('change', ['decision', 'opened', 'intent', 'modified'])
def test_initial_import_requires_the_original_sealed_ledger(study, tmp_path, change):
    prior = tmp_path/'previous.json'
    data = json.loads(prior.read_text())
    if change == 'decision':
        data['decision'] = {}
    elif change == 'opened':
        data['final_test_opened'] = True
    elif change == 'intent':
        data['jobs']['mars-dreamer-ctrl-v1-fin-frozen-s0-e24-cpu'] = {
            'device': 'cpu', 'charged_seconds': 0, 'submitted': False}
    else:
        data['jobs']['old']['charged_seconds'] = 1
    prior.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='ledger|final'):
        FollowupStudy(tmp_path/'other', prior)


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


def test_sealed_ledger_with_separate_final_access_evidence_is_rejected(study, tmp_path):
    (tmp_path/'report.json').write_text(json.dumps({'final_test_opened': True}))
    with pytest.raises(ValueError, match='ledger'):
        FollowupStudy(tmp_path/'other', tmp_path/'previous.json')


@pytest.mark.parametrize('wall,seconds,valid', [(10.9, 10., True), (11.1, 10., False),
                                                 (-1., 10., False), (float('nan'), 10., False)])
def test_retained_charge_covers_the_worker_runtime(study, wall, seconds, valid):
    task = ('model', 'fast', 101, 24, 'cpu')
    name, manifest = study.manifest(task)
    (study.state_dir/'long-101.json').write_text(json.dumps(manifest))
    status = dict(device='cpu', allocated_cpu_cores=2, source_sha256=SOURCE, wall_seconds=wall)
    record = dict(job=name, record=dict(status=status), cpu_core_hours=seconds*2/3600)
    (study.state_dir/(name+'.record')).write_text(json.dumps(record))
    if valid:
        study.adopt(task)
        assert study.state['jobs'][name]['charged_seconds'] == pytest.approx(seconds)
    else:
        with pytest.raises(ValueError, match='charge'):
            study.adopt(task)
        assert name not in study.state['jobs']


def worker_records(study, calls, extended_score, tasks):
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


@pytest.mark.parametrize("extended_score,opens_final", [(0.6, False), (0.5, True)])
def test_development_screen_guards_confirmation_and_one_frozen_final(study, monkeypatch, extended_score, opens_final):
    calls = []
    def workers(tasks):
        return worker_records(study, calls, extended_score, tasks)
    monkeypatch.setattr(study, "batch", workers)
    study.run()
    assert study.state["phase"] == "complete"
    assert [(v,s) for t,m,v,s in calls if t == "long-more" and m == "actor"] == [("h6",101), ("h6",102)]
    assert [s for _,m,_,s in calls if m == "confirm"] == (list(range(201,206)) if opens_final else [])
    assert sum(m == "final" for _,m,_,_ in calls) == int(opens_final)
    report = json.loads((study.state_dir/'report.json').read_text())
    assert report['last_attempt']['trial'] == 'long-more'
    if opens_final:
        assert report['qualified_recipe'] == study.state['qualified_recipe']
        proposal = report['larger_proposal']
        assert proposal['authorized'] is False and proposal['estimated_cpu_core_hours'] > 0
        assert proposal['startup_allowance_core_hours'] > 0
    else:
        assert report['qualified_recipe'] is None and 'larger_proposal' not in report
    count = len(calls)
    study.run()
    assert len(calls) == count


@pytest.mark.parametrize('bad', ['forecast', 'forecast_sol', 'forecast_missing', 'condition', 'curve_score', 'reported_score'])
def test_followup_rejects_malformed_checkpoint_metrics(study, monkeypatch, bad):
    mode, variant = ('model', 'fast') if bad.startswith('forecast') else ('actor', 'h3')

    def workers(tasks):
        records = worker_records(study, [], .5, tasks)
        row = records[0]['result']
        if bad.startswith('forecast'):
            curve = row['curves'][-1]
            curve['validation'] = {**curve['validation'], 'model_temperature': {
                'rmse': -0.01 if bad == 'forecast' else 1., 'at_sols': {'10': float('inf') if bad == 'forecast_sol' else 1.}}}
            if bad == 'forecast_missing':
                del curve['validation']['model_reward']
        else:
            curve = next(c for c in row['curves'] if c['phase'] == 'behavior')
            if bad == 'condition':
                curve['rows'] = [dict(r) for r in curve['rows']]
                curve['rows'][0]['tracking_rmse_K'] = -1.
                curve['validation_rmse_K'] = 1.
            elif bad == 'curve_score':
                curve['validation_rmse_K'] = float('inf')
            else:
                row['validation_rmse_K'] = -1.
        return records

    monkeypatch.setattr(study, 'batch', workers)
    with pytest.raises(ValueError):
        study.rows(mode, variant, [101])

def test_infeasible_horizons_report_the_actual_last_attempt(study, monkeypatch):
    def workers(tasks):
        results = worker_records(study, [], .5, tasks)
        for record in results:
            row = record['result']
            if row['mode'] == 'actor':
                for curve in row['curves']:
                    if curve['phase'] == 'behavior':
                        curve['rows'][0]['violations'] = 1
        return results

    monkeypatch.setattr(study, 'batch', workers)
    study.run()
    report = json.loads((study.state_dir/'report.json').read_text())
    assert report['last_attempt']['horizon'] == 6
    assert report['qualified_recipe'] is None

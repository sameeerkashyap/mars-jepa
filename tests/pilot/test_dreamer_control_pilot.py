"""Public forecast, admission, freeze and report contracts for response control."""
import numpy as np
import pytest
import torch

from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
from benchmarks.dreamer_control.climate import climate_imagination
from scripts.pilot.dreamer_control_forecast import prior_forecast


def test_forecast_accepts_only_reset_and_actions_and_preserves_rng():
    torch.set_num_threads(1)
    learner = DreamerLearner(LearnerConfig(obs_dim=133), imagination_env_factory=climate_imagination)
    reset = np.zeros(133, np.float32)
    reset[-2:] = 1
    actions = np.full((12, 1), -0.5, np.float32)
    state = torch.get_rng_state().clone()
    first = prior_forecast(learner, reset, actions)
    assert first['observations'].shape == (12, 133)
    assert first['rewards'].shape == (12,)
    assert torch.equal(state, torch.get_rng_state())
    # Future labels have no argument or path into this forecast seam.
    poison = np.full((13, 133), np.nan)
    poison[0] = reset
    again = prior_forecast(learner, poison[0], actions)
    np.testing.assert_array_equal(first['observations'], again['observations'])
    np.testing.assert_array_equal(first['rewards'], again['rewards'])


def test_final_report_pooled_gate_and_paired_uncertainty():
    from scripts.pilot.dreamer_control import control_report
    records = [{'seed': seed, 'rows': [dict(condition_seed=c, tracking_rmse_K=.8,
                  baseline_rmse_K=1., heat_used_Wm2_sol=250., violations=0)
                  for c in range(5200, 5220)]} for seed in range(201, 206)]
    result = control_report(records)
    assert result['passed']
    assert result['pooled_improvement'] == pytest.approx(.2)
    assert result['bootstrap_95'] == pytest.approx([.2, .2])
    records[0]['rows'][0]['violations'] = 1
    assert not control_report(records)['passed']
    with pytest.raises(ValueError, match='five'):
        control_report(records[:-1])
    for record in records:
        record['rows'][0]['baseline_rmse_K'] = -1.
    with pytest.raises(ValueError, match='RMSE'):
        control_report(records)


def test_freeze_requires_all_five_matching_hashes(tmp_path):
    from scripts.pilot.dreamer_control import freeze_confirmation, digest
    recipe = dict(learning_rate=.00004, horizon=3)
    rows = [dict(seed=s, checkpoint_sha256='a'*64, source_sha256='source',
                 config_sha256='config', data_sha256=['train', 'validation'],
                 validation_rmse_K=.5, checkpoint=f'confirm-{s}/best.pt', mode='confirm', variant='selected',
                 recipe_sha256=digest(recipe)) for s in range(201, 206)]
    with pytest.raises(ValueError, match='five'):
        freeze_confirmation(rows[:-1], 'source', 'config', ['train', 'validation'], 'constant_2.5', recipe)
    rows[0]['validation_rmse_K'] = -1.
    with pytest.raises(ValueError, match='identity'):
        freeze_confirmation(rows, 'source', 'config', ['train', 'validation'], 'constant_2.5', recipe)
    rows[0]['validation_rmse_K'] = .5
    rows[3]['data_sha256'] = ['changed']
    with pytest.raises(ValueError, match='identity'):
        freeze_confirmation(rows, 'source', 'config', ['train', 'validation'], 'constant_2.5', recipe)
    rows[3]['data_sha256'] = ['train', 'validation']
    for changed in (dict(learning_rate=.00008, horizon=3), dict(learning_rate=.00004, horizon=6)):
        with pytest.raises(ValueError, match='identity'):
            freeze_confirmation(rows, 'source', 'config', ['train', 'validation'], 'constant_2.5', changed)
    assert freeze_confirmation(rows, 'source', 'config', ['train', 'validation'], 'constant_2.5', recipe)['seeds'] == list(range(201, 206))
    paired_recipe = dict(recipe, paired_response_weight=1.)
    paired_rows = [dict(row, paired_response_weight=1., recipe_sha256=digest(paired_recipe))
                   for row in rows]
    assert freeze_confirmation(paired_rows, 'source', 'config', ['train', 'validation'],
                               'constant_2.5', paired_recipe)['recipe'] == paired_recipe
    with pytest.raises(ValueError, match='identity'):
        freeze_confirmation(paired_rows, 'source', 'config', ['train', 'validation'], 'constant_2.5', recipe)


def previous_ledger(tmp_path, phase='complete'):
    import json
    path = tmp_path/'previous.json'
    path.write_text(json.dumps(dict(phase=phase, images={'cpu': 'python@sha256:fixture', 'cuda': 'unused'},
        jobs={'cpu': dict(device='cpu', reserved_seconds=0, charged_seconds=3600),
              'gpu': dict(device='cuda', reserved_seconds=0, charged_seconds=1800)})))
    return path


def test_cost_import_once_admission_and_manifest_constraints(tmp_path):
    import json
    from scripts.pilot.dreamer_control import DreamerControlStudy
    prior = previous_ledger(tmp_path, 'frozen')
    study = DreamerControlStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', prior)
    assert not study.import_previous()
    assert not study.state['jobs']
    previous_ledger(tmp_path)
    assert study.import_previous()
    assert study.remaining() == pytest.approx((44.75, 11.5))
    assert study.import_previous()
    assert study.remaining() == pytest.approx((44.75, 11.5))
    study.state.update(source_sha256='a'*64, configmap='fixture')
    with pytest.raises(ValueError, match='readiness'):
        study.admit([('actor', 'h3', 101, 12, 'cpu')])
    study.state['readiness_passed'] = True
    with pytest.raises(ValueError, match='frozen'):
        study.admit([('final', 'frozen', 0, 12, 'cpu')])
    _, manifest = study.manifest(('model', 'base', 101, 12, 'cpu'))
    payload = json.dumps(manifest)
    assert 'activeDeadlineSeconds' not in payload
    assert 'ttlSecondsAfterFinished' not in payload
    assert 'timeout ' not in payload
    assert manifest['spec']['backoffLimit'] == 0
    assert 'scripts.pilot.dreamer_control_worker' in payload
    assert 'torch==${MARSWORLD_TORCH_VERSION' in payload
    study.state['jobs']['spent'] = dict(device='cpu', reserved_seconds=0, charged_seconds=44.8*3600/2)
    with pytest.raises(RuntimeError, match='not admitted'):
        study.admit([('data', 'base', 101, 24, 'cpu')])
    prior.write_text(prior.read_text()+'\n')
    with pytest.raises(ValueError, match='ledger changed'):
        study.import_previous()


def test_manifest_passes_validated_configuration_to_worker(tmp_path):
    import json
    import subprocess
    from scripts.pilot.dreamer_control import CONFIG, DreamerControlStudy, digest

    config = dict(json.loads(CONFIG.read_text()), paired_response_weight=1.)
    custom = tmp_path/'custom.json'
    custom.write_text(json.dumps(config))
    study = DreamerControlStudy(tmp_path/'study', 'fixture', previous_ledger(tmp_path), custom)
    study.import_previous()
    study.state.update(source_sha256='a'*64, configmap='fixture')
    _, manifest = study.manifest(('meter', 'base', 101, 12, 'cpu'))
    script = manifest['spec']['template']['spec']['containers'][0]['args'][0]
    worker = script[script.index('exec python -u -m scripts.pilot.dreamer_control_worker'):]
    # Read the worker's config argument without running its training stages.
    probe = worker.replace('python -u -m scripts.pilot.dreamer_control_worker',
                           """sh -c 'cat "$3"' probe""", 1)
    delivered = json.loads(subprocess.check_output(['sh', '-c', probe], text=True))
    assert digest(delivered) == study.state['config_sha256']


def test_readiness_failure_finishes_without_actor_or_test(tmp_path):
    from scripts.pilot.dreamer_control import DreamerControlStudy
    executed = []

    class FixtureStudy(DreamerControlStudy):
        def prepare(self):
            self.state['source_sha256'] = 'fixture'

        def batch(self, tasks):
            executed.extend(tasks)
            rows = []
            for mode, variant, seed, rung, device in tasks:
                if mode == 'meter':
                    result = {'model_update_seconds': .03, 'behavior_update_seconds': .03,
                              'collection_seconds': 2., 'generation_condition_seconds': 5.}
                elif mode == 'data':
                    result = {'data_sha256': ['train', 'validation'], 'baseline': {'selected': 'constant_2.5'}}
                else:
                    result = dict(variant=variant, seed=seed, ready=False,
                        validation={'model_response': {'rmse': 1.}}, curves=[
                            {'updates': n, 'validation': {'model_response': {'rmse': 1.}}}
                            for n in (500, 1000, 2000)])
                rows.append({'result': result})
            return rows

    study = FixtureStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', previous_ledger(tmp_path))
    study.run()
    assert [task[0] for task in executed] == ['meter', 'data', 'model', 'model', 'model', 'model']
    assert study.state['phase'] == 'complete'
    import json
    report = json.loads((tmp_path/'study'/'report.json').read_text())
    assert report['passed'] is False and report['final_test_opened'] is False
    study.run()
    assert len(executed) == 6


def test_simple_model_controller_uses_training_weights_and_respects_heat():
    from scripts.pilot.dreamer_control_worker import simple_model_plan
    coefficients = np.tile([0., 1.], (64, 1))
    # Daily linear interpolation from zero makes the first interval need more heat.
    plan = simple_model_plan(coefficients, np.arange(4), np.ones(4))
    assert len(plan) == 12 and all(0 <= h <= 10 for h in plan)
    assert sum(plan)*10 <= 300+1e-5
    assert 1.5 < np.mean(plan) < 3


@pytest.mark.parametrize('mode', ['actor', 'confirm'])
def test_paired_pretraining_requires_weight_in_selected_recipe(tmp_path, monkeypatch, mode):
    from argparse import Namespace
    from scripts.pilot.dreamer_control_worker import execute

    monkeypatch.setenv('MARSWORLD_RECIPE', '{"learning_rate": 0.00004, "horizon": 3}')
    args = Namespace(mode=mode, rung=12, base=tmp_path)
    with pytest.raises(ValueError, match='recipe'):
        execute(args, {'paired_response_weight': 1.})


def test_meter_reports_loss_from_the_paired_update_it_timed(tmp_path):
    import json
    from argparse import Namespace
    from scripts.pilot.dreamer_control import CONFIG
    from scripts.pilot.dreamer_control_worker import execute

    torch.set_num_threads(2)
    config = dict(json.loads(CONFIG.read_text()), paired_response_weight=1.)
    report = execute(Namespace(mode='meter', out=tmp_path, source='fixture'), config)
    assert report['last_model_update_loss']['loss_paired_response'] > 0
    assert report['model_update_seconds'] > 0


def test_worker_recovery_and_final_freeze_are_checked_before_any_test_outcome(tmp_path, monkeypatch):
    import argparse
    import json
    from data.control import generate_dataset
    from scripts.pilot import dreamer_control_worker as worker
    from scripts.pilot.dreamer_control import CONFIG, digest
    base = tmp_path/'results'
    generate_dataset(base/'data-12'/'train', split='train', condition_seeds=[5000])
    generate_dataset(base/'data-12'/'validation', split='validation', condition_seeds=[5100])
    hashes, _, _ = worker.datasets(base, 12)
    config = json.loads(CONFIG.read_text())
    config['model_checkpoints'] = [1, 2]
    config['learner']['warmup_steps'] = 1
    out = base/'model'
    out.mkdir()
    args = argparse.Namespace(base=base, out=out, rung=12, source='fixture', mode='model', variant='base', seed=101)
    first = worker.train_stage(args, config)
    resumed = worker.train_stage(args, config)
    assert first['checkpoint_sha256'] == resumed['checkpoint_sha256']
    assert first['validation'] == resumed['validation']
    assert 'loss_paired_response' not in first['curves'][0]['loss']
    paired_config = dict(config, paired_response_weight=1.)
    args.out = base/'paired'
    args.out.mkdir()
    paired = worker.train_stage(args, paired_config)
    assert paired['paired_response_weight'] == 1.
    assert paired['curves'][0]['loss']['loss_paired_response'] > 0
    assert paired['validation'] != first['validation']
    assert paired['recipe_sha256'] != first['recipe_sha256']
    assert worker.train_stage(args, paired_config)['checkpoint_sha256'] == paired['checkpoint_sha256']
    with pytest.raises(ValueError, match='identity changed'):
        worker.train_stage(args, config)
    actor_out = base/'actor'
    actor_out.mkdir()
    args.out, args.mode, args.variant = actor_out, 'confirm', 'selected'
    monkeypatch.setenv('MARSWORLD_RECIPE', json.dumps({'learning_rate': .00004, 'horizon': 3}))
    failed = worker.train_stage(args, config)
    assert failed['readiness_failed'] is True and failed['actor_updates'] == 0
    guard_checkpoint = torch.load(actor_out/'resume.pt', weights_only=False)
    assert guard_checkpoint['behavior_updates'] == 0
    assert not (actor_out/'best.pt').exists()
    for changed in (dict(learning_rate=.00008, horizon=3), dict(learning_rate=.00004, horizon=6)):
        monkeypatch.setenv('MARSWORLD_RECIPE', json.dumps(changed))
        with pytest.raises(ValueError, match='identity changed'):
            worker.train_stage(args, config)
    args.out = base/'paired-confirm'
    args.out.mkdir()
    recipe = dict(learning_rate=.00004, horizon=3, paired_response_weight=1.)
    monkeypatch.setenv('MARSWORLD_RECIPE', json.dumps(recipe))
    confirmed = worker.train_stage(args, paired_config)
    assert confirmed['readiness_failed'] and confirmed['actor_updates'] == 0
    assert confirmed['recipe_sha256'] == paired['recipe_sha256']
    assert confirmed['validation'] == paired['validation']
    assert [c['loss'] for c in confirmed['curves']] == [c['loss'] for c in paired['curves']]
    with pytest.raises(ValueError, match='recipe'):
        worker.train_stage(args, config)
    learner = worker.new_learner(config, 201, .00004)
    rows = []
    for seed in range(201, 206):
        path = base/f'confirm-{seed}.pt'
        identity = dict(source_sha256='fixture', config_sha256=digest(config), data_sha256=hashes,
                        seed=seed, mode='confirm', variant='selected', rung=12,
                        recipe_sha256=digest(dict(learning_rate=.00004, horizon=3)))
        worker.save_recovery(learner, path, {'identity': identity, 'seed_cursor': 3})
        rows.append(dict(**identity, checkpoint=path.name, checkpoint_sha256=worker.file_hash(path),
                         validation_rmse_K=.5))
    frozen = dict(source_sha256='fixture', config_sha256=digest(config), data_sha256=hashes,
                  seeds=list(range(201, 206)), baseline='no_heat', runs=rows,
                  recipe=dict(learning_rate=.00004, horizon=3))
    final = base/'final'
    final.mkdir()
    args.out = final
    seen = []
    monkeypatch.setattr(worker, 'evaluate', lambda learner, seeds, baseline, initial_states, **kwargs:
                        seen.append(list(seeds)) or [])
    invalid_runs = [rows[:-1], rows[:-1]+[rows[0]], [{**row, 'source_sha256': 'changed'} for row in rows]]
    with monkeypatch.context() as guard:
        def no_datasets(*args):
            raise AssertionError('Invalid freeze reached dataset access')
        guard.setattr(worker, 'datasets', no_datasets)
        for runs in invalid_runs:
            guard.setenv('MARSWORLD_FROZEN', json.dumps({**frozen, 'runs': runs}))
            with pytest.raises(ValueError, match='five|identity'):
                worker.final_evaluation(args, config)
    assert not seen and not (final/'EVALUATION_STARTED.json').exists()
    original = rows[-1]['checkpoint_sha256']
    rows[-1]['checkpoint_sha256'] = '0'*64
    monkeypatch.setenv('MARSWORLD_FROZEN', json.dumps(frozen))
    with pytest.raises(ValueError, match='checkpoint changed'):
        worker.final_evaluation(args, config)
    assert not seen and not (final/'EVALUATION_STARTED.json').exists()
    rows[-1]['checkpoint_sha256'] = original
    for changed in (dict(learning_rate=.00008, horizon=3), dict(learning_rate=.00004, horizon=6),
                    dict(learning_rate=.00004, horizon=3, paired_response_weight=1.)):
        monkeypatch.setenv('MARSWORLD_FROZEN', json.dumps({**frozen, 'recipe': changed}))
        with pytest.raises(ValueError, match='identity'):
            worker.final_evaluation(args, config)
    assert not seen and not (final/'EVALUATION_STARTED.json').exists()
    monkeypatch.setenv('MARSWORLD_FROZEN', json.dumps(frozen))
    assert len(worker.final_evaluation(args, config)['records']) == 5
    assert seen == [list(range(5200, 5220))]*5
    with pytest.raises(FileExistsError):
        worker.final_evaluation(args, config)
    assert len(seen) == 5


def test_interrupted_submission_resumes_same_job_and_reuses_completed_result(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts.pilot import dreamer_control as driver
    study = driver.DreamerControlStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', previous_ledger(tmp_path))
    study.import_previous()
    study.state.update(source_sha256='a'*64, configmap='fixture')
    task = ('meter', 'base', 101, 12, 'cpu')
    submitted = []
    remote_exists = False

    def transport(*args, **kwargs):
        nonlocal remote_exists
        if args[:2] == ('get', 'jobs'):
            return SimpleNamespace(stdout=json.dumps({'items': []}), stderr='', returncode=0)
        if args[:2] == ('get', 'job'):
            return SimpleNamespace(stdout='{}' if remote_exists else '', stderr='' if remote_exists else 'NotFound',
                                   returncode=0 if remote_exists else 1)
        submitted.append(args)
        remote_exists = True
        raise RuntimeError('API response lost after server accepted apply')

    monkeypatch.setattr(driver, 'kubectl', transport)
    with pytest.raises(RuntimeError, match='response lost'):
        study.batch([task])
    assert len(submitted) == 1
    resumed = driver.DreamerControlStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', study.state['previous_state'])

    def collect(name):
        record = {'status': {'status': 'ok'}, 'result': {'recovery_next_update_exact': True}}
        resumed.state['jobs'][name].update(record=record, charged_seconds=1)
        resumed.save()
        return record

    monkeypatch.setattr(resumed, 'collect', collect)
    assert resumed.batch([task])[0]['result']['recovery_next_update_exact']
    assert resumed.batch([task])[0]['result']['recovery_next_update_exact']
    assert len(submitted) == 1


@pytest.fixture
def completed_transport(monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts.pilot import dreamer_control as driver
    applied, existing = [], set()
    result = {'collected': True}

    def transport(*args, **kwargs):
        if args[:2] == ('apply', '-f'):
            name = kwargs['value']['metadata']['name']
            applied.append(name)
            existing.add(name)
            payload = {}
        elif args[:2] == ('get', 'jobs'):
            payload = {'items': []}
        elif args[:2] == ('get', 'job'):
            if args[2] not in existing:
                return SimpleNamespace(stdout='', stderr='NotFound', returncode=1)
            payload = {'status': {'succeeded': 1, 'startTime': '2026-01-01T00:00:00Z',
                       'completionTime': '2026-01-01T12:30:00Z',
                       'conditions': [{'type': 'Complete', 'status': 'True'}]}}
        elif args[:2] == ('get', 'pods'):
            payload = {'items': []}
        elif args[0] == 'logs':
            record = {'status': {'status': 'ok', 'wall_seconds': 45000}, 'result': result}
            return SimpleNamespace(stdout='RESULT_JSON='+json.dumps(record), stderr='', returncode=0)
        else:
            raise AssertionError(args)
        return SimpleNamespace(stdout=json.dumps(payload), stderr='', returncode=0)

    monkeypatch.setattr(driver, 'kubectl', transport)
    monkeypatch.setattr('scripts.pilot.nautilus_study.kubectl', transport)
    return applied, existing, result


def test_batch_rechecks_actual_charges_before_next_worker_pair(tmp_path, completed_transport):
    from scripts.pilot.dreamer_control import DreamerControlStudy
    study = DreamerControlStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', previous_ledger(tmp_path))
    study.import_previous()
    study.state.update(source_sha256='a'*64, configmap='fixture')
    tasks = [('meter', 'base', seed, 12, 'cpu') for seed in range(101, 105)]
    with pytest.raises(RuntimeError, match='not admitted'):
        study.batch(tasks)
    applied, _, _ = completed_transport
    assert len(applied) == 2
    assert all(study.state['jobs'][name]['record']['result']['collected'] for name in applied)
    assert study.remaining()[0] < 0


@pytest.mark.parametrize('include_new', [False, True])
def test_overbudget_resume_collects_existing_work_without_new_jobs(tmp_path, completed_transport, include_new):
    from scripts.pilot.dreamer_control import DreamerControlStudy
    study = DreamerControlStudy(tmp_path/'study', 'mars-dreamer-ctrl-v1', previous_ledger(tmp_path))
    study.import_previous()
    study.state.update(source_sha256='a'*64, configmap='fixture')
    applied, existing, result = completed_transport
    tasks = [('meter', 'base', seed, 12, 'cpu') for seed in range(101, 105)]
    record = {'status': {'status': 'ok'}, 'result': {'reused': True}}
    if include_new:
        record['result']['readiness_failed'] = True
        result['readiness_failed'] = True
    done, _ = study.manifest(tasks[1])
    running, _ = study.manifest(tasks[3])
    study.state['jobs'][done] = dict(device='cpu', reserved_seconds=0, charged_seconds=90000, record=record)
    study.state['jobs'][running] = dict(device='cpu', reserved_seconds=1200, submitted=True)
    existing.add(running)
    study.save()
    resumed = DreamerControlStudy(study.state_dir, study.study_id, study.state['previous_state'])
    if include_new:
        with pytest.raises(RuntimeError, match='not admitted'):
            resumed.batch(tasks)
    else:
        study.admit([tasks[1], tasks[3]])
        assert resumed.batch([tasks[1], tasks[3]]) == [record, resumed.state['jobs'][running]['record']]
    assert resumed.state['jobs'][running]['record']['result']['collected']
    assert resumed.state['jobs'][done]['record'] == record
    assert not applied

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


def test_freeze_requires_all_five_matching_hashes(tmp_path):
    from scripts.pilot.dreamer_control import freeze_confirmation
    rows = [dict(seed=s, checkpoint_sha256='a'*64, source_sha256='source',
                 config_sha256='config', data_sha256=['train', 'validation'],
                 validation_rmse_K=.5, checkpoint=f'confirm-{s}/best.pt') for s in range(201, 206)]
    with pytest.raises(ValueError, match='five'):
        freeze_confirmation(rows[:-1], 'source', 'config', ['train', 'validation'], 'constant_2.5')
    rows[3]['data_sha256'] = ['changed']
    with pytest.raises(ValueError, match='identity'):
        freeze_confirmation(rows, 'source', 'config', ['train', 'validation'], 'constant_2.5')


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
    actor_out = base/'actor'
    actor_out.mkdir()
    args.out, args.mode, args.variant = actor_out, 'confirm', 'selected'
    monkeypatch.setenv('MARSWORLD_RECIPE', json.dumps({'learning_rate': .00004, 'horizon': 3}))
    failed = worker.train_stage(args, config)
    assert failed['readiness_failed'] is True and failed['actor_updates'] == 0
    guard_checkpoint = torch.load(actor_out/'resume.pt', weights_only=False)
    assert guard_checkpoint['behavior_updates'] == 0
    assert not (actor_out/'best.pt').exists()
    learner = worker.new_learner(config, 201, .00004)
    rows = []
    for seed in range(201, 206):
        path = base/f'confirm-{seed}.pt'
        identity = dict(source_sha256='fixture', config_sha256=digest(config), data_sha256=hashes,
                        seed=seed, mode='confirm', variant='selected', rung=12)
        worker.save_recovery(learner, path, {'identity': identity, 'seed_cursor': 3})
        rows.append(dict(**identity, checkpoint=path.name, checkpoint_sha256=worker.file_hash(path)))
    frozen = dict(source_sha256='fixture', config_sha256=digest(config), data_sha256=hashes,
                  seeds=list(range(201, 206)), baseline='no_heat', runs=rows)
    final = base/'final'
    final.mkdir()
    args.out = final
    seen = []
    monkeypatch.setattr(worker, 'evaluate', lambda learner, seeds, baseline, initial_states, **kwargs:
                        seen.append(list(seeds)) or [])
    original = rows[-1]['checkpoint_sha256']
    rows[-1]['checkpoint_sha256'] = '0'*64
    monkeypatch.setenv('MARSWORLD_FROZEN', json.dumps(frozen))
    with pytest.raises(ValueError, match='checkpoint changed'):
        worker.final_evaluation(args, config)
    assert not seen and not (final/'EVALUATION_STARTED.json').exists()
    rows[-1]['checkpoint_sha256'] = original
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

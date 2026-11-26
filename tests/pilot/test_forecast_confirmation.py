import numpy as np
import pytest


def test_confirmatory_gain_uses_pooled_squared_error_and_paired_conditions():
    from scripts.pilot.dreamer_forecast_confirmation import forecast_report

    records = {}
    for seed in range(201, 206):
        rows = {}
        for condition in range(5300, 5332):
            baseline = 2. if condition % 2 else 4.
            ratio = 0.5 if seed == 201 else 0.8
            rows[condition] = {name: {'rmse': error} for name, error in (
                ('model_response', baseline * ratio), ('linear_response', baseline),
                ('model_reward', 3.), ('linear_reward', 2.),
                ('model_temperature', 2.), ('persistence_temperature', 4.))}
            rows[condition]['model_response']['at_sols'] = {str(i): baseline * ratio for i in range(10, 121, 10)}
        records[seed] = rows
    result = forecast_report(records)
    response = result['comparisons']['response']
    assert response['model_rmse'] == pytest.approx(np.sqrt(5.62))
    assert response['baseline_rmse'] == pytest.approx(np.sqrt(10.))
    assert response['bootstrap_95'][0] == pytest.approx(.2)
    assert response['bootstrap_95'][1] > .25
    assert result['per_seed'][201]['response']['gain'] == pytest.approx(.5)
    assert result['passed'] and result['comparisons']['reward']['gain'] == pytest.approx(-.5)
    assert result['at_sols']['model_response']['70'] == pytest.approx(np.sqrt(5.62))
    for seed in records:
        for c, row in records[seed].items():
            row['linear_response']['rmse'] = 1.
            row['model_response']['rmse'] = 1.24 if c % 2 == 0 else 0.1
    uncertain = forecast_report(records)
    assert uncertain['comparisons']['response']['gain'] > .1
    assert uncertain['comparisons']['response']['bootstrap_95'][0] < 0
    assert not uncertain['passed']
    del records[205][5300]
    with pytest.raises(ValueError, match='conditions'):
        forecast_report(records)


def test_open_freezes_all_best_validation_checkpoints_once_even_if_unready(tmp_path):
    import json
    import torch
    from scripts.pilot.dreamer_forecast_confirmation import (
        CHECKPOINTS, CONFIG_SHA, DATA, RECIPE_SHA, SEEDS, SOURCE, open_forecast,
    )

    from scripts.pilot import dreamer_control_forecast as scorer, dreamer_control_worker as worker
    from benchmarks.dreamer_control import DreamerLearner, LearnerConfig

    torch.set_num_threads(1)
    learner = DreamerLearner(LearnerConfig(obs_dim=133), imagination_env_factory=worker.climate_imagination)
    learner.updates = 2000
    episode = validation_episode()
    args = ([episode], [episode], np.arange(4), np.ones(4))
    expected = scorer.forecast_validation(learner, *args)
    runs = []
    for seed in SEEDS:
        identity = dict(seed=seed, mode='model', variant='fast', rung=24, source_sha256=SOURCE,
                        config_sha256=CONFIG_SHA, data_sha256=DATA, recipe_sha256=RECIPE_SHA)
        curves = [dict(phase='model', updates=step, validation=expected if step == 2000 else {
            **expected, 'model_response': {'rmse': expected['model_response']['rmse'] + 1.}})
                  for step in CHECKPOINTS]
        path = tmp_path / f'seed-{seed}.pt'
        worker.save_recovery(learner, path, dict(identity=identity, model_updates=2000,
                              behavior_updates=0, best_model_metrics=expected))
        runs.append((dict(**identity, curves=curves, ready=False,
                          checkpoint_sha256=worker.file_hash(path)), path))
    preflight = (DATA, {}, scorer, args)
    with pytest.raises(ValueError, match='five'):
        open_forecast(tmp_path, runs[:-1], *preflight)
    assert not (tmp_path / 'FROZEN.json').exists()
    altered = runs[0][0].copy()
    altered['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checkpoint'):
        open_forecast(tmp_path, [(altered, runs[0][1]), *runs[1:]], *preflight)
    assert not (tmp_path / 'FROZEN.json').exists()
    bad_path = tmp_path / 'damaged.pt'
    payload = torch.load(runs[-1][1], weights_only=False)
    del payload['loss']
    torch.save(payload, bad_path)
    damaged = dict(runs[-1][0], checkpoint_sha256=worker.file_hash(bad_path))
    with pytest.raises(KeyError, match='loss'):
        open_forecast(tmp_path, [*runs[:-1], (damaged, bad_path)], *preflight)
    assert not (tmp_path / 'FROZEN.json').exists()
    with pytest.raises(ValueError, match='validation'):
        shifted = validation_episode()
        shifted['policy']['rewards'] *= 2
        open_forecast(tmp_path, runs, DATA, {}, scorer, ([episode], [shifted], *args[2:]))
    assert not (tmp_path / 'FROZEN.json').exists()
    frozen, learners = open_forecast(tmp_path, runs, *preflight)
    assert list(learners) == list(SEEDS)
    assert [r['updates'] for r in frozen['runs']] == [2000] * 5
    assert json.loads((tmp_path / 'FROZEN.json').read_text()) == frozen
    with pytest.raises(FileExistsError):
        open_forecast(tmp_path, runs, *preflight)


def validation_episode():
    obs = np.zeros((13, 133), np.float32)
    obs[:, -2] = np.arange(12, -1, -1) / 12
    obs[:, -1] = 1
    return dict(policy=dict(observations=obs, actions=np.full((12, 1), -1.), rewards=-np.ones(12)),
                oracle_diagnostics=dict(daily_response_T_s=np.zeros((12, 10, 64))))


def test_confirmation_reports_twelve_boundaries_without_changing_validation_points():
    from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
    from benchmarks.dreamer_control.climate import climate_imagination
    from scripts.pilot.dreamer_control_forecast import forecast_validation
    import torch

    torch.set_num_threads(1)
    learner = DreamerLearner(LearnerConfig(obs_dim=133), imagination_env_factory=climate_imagination)
    episode = validation_episode()
    args = (learner, [episode], [episode], np.arange(4), np.ones(4))
    validation = forecast_validation(*args)
    confirmation = forecast_validation(*args, sols=iter(range(10, 121, 10)))
    assert list(confirmation['model_temperature']['at_sols']) == [str(i) for i in range(10, 121, 10)]
    for name, values in validation.items():
        assert confirmation[name]['rmse'] == values['rmse']
        for sol, value in values.get('at_sols', {}).items():
            assert confirmation[name]['at_sols'][sol] == value


@pytest.mark.parametrize('sols', [(0,), (15,), (-10,), (130,), (10, 10), (20.0,), ()])
def test_scorer_rejects_invalid_ten_sol_boundaries(sols):
    from scripts.pilot.dreamer_control_forecast import forecast_validation
    with pytest.raises(ValueError, match='ten-sol'):
        forecast_validation(None, [], [], None, None, sols=sols)


@pytest.mark.parametrize('retained', ['FROZEN.json', 'conditions', 'forecast-s203.json', 'report.json'])
def test_worker_refuses_prior_test_output_before_training(tmp_path, retained):
    from scripts.pilot.dreamer_forecast_confirmation import run
    out = tmp_path / 'forecast-confirmation-v1'
    out.mkdir()
    path = out / retained
    if retained == 'conditions':
        path.mkdir()
    else:
        path.touch()
    with pytest.raises(FileExistsError, match='already opened'):
        run(tmp_path, tmp_path / 'unused-archive', None)


def test_forecast_generation_restores_original_splits_when_destination_exists(tmp_path):
    import data.control as data
    from scripts.pilot.dreamer_forecast_confirmation import fresh_conditions
    original = data.SPLITS.copy()
    with pytest.raises(FileExistsError):
        fresh_conditions(tmp_path)
    assert data.SPLITS == original

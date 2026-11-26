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
    import hashlib
    import json
    import torch
    from scripts.pilot.dreamer_forecast_confirmation import (
        CHECKPOINTS, CONFIG_SHA, DATA, RECIPE_SHA, SEEDS, SOURCE, open_forecast,
    )

    runs = []
    for seed in SEEDS:
        identity = dict(seed=seed, mode='model', variant='fast', rung=24, source_sha256=SOURCE,
                        config_sha256=CONFIG_SHA, data_sha256=DATA, recipe_sha256=RECIPE_SHA)
        curves = [dict(phase='model', updates=step, validation={'model_response': {'rmse': error}})
                  for step, error in zip(CHECKPOINTS, (2., 1., .5, .6, .9))]
        best = curves[2]
        path = tmp_path / f'seed-{seed}.pt'
        torch.save({'pilot': dict(identity=identity, model_updates=2000, behavior_updates=0,
                                  best_model_metrics=best['validation'])}, path)
        runs.append((dict(**identity, curves=curves, ready=False,
                          checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), path))
    with pytest.raises(ValueError, match='five'):
        open_forecast(tmp_path, runs[:-1], DATA, {})
    assert not (tmp_path / 'FROZEN.json').exists()
    altered = runs[0][0].copy()
    altered['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checkpoint'):
        open_forecast(tmp_path, [(altered, runs[0][1]), *runs[1:]], DATA, {})
    assert not (tmp_path / 'FROZEN.json').exists()
    frozen = open_forecast(tmp_path, runs, DATA, {})
    assert [r['updates'] for r in frozen['runs']] == [2000] * 5
    assert json.loads((tmp_path / 'FROZEN.json').read_text()) == frozen
    with pytest.raises(FileExistsError):
        open_forecast(tmp_path, runs, DATA, {})


def test_confirmation_reports_twelve_boundaries_without_changing_validation_points():
    from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
    from benchmarks.dreamer_control.climate import climate_imagination
    from scripts.pilot.dreamer_control_forecast import forecast_validation
    import torch

    torch.set_num_threads(1)
    learner = DreamerLearner(LearnerConfig(obs_dim=133), imagination_env_factory=climate_imagination)
    obs = np.zeros((13, 133), np.float32)
    obs[:, -2] = np.arange(12, -1, -1) / 12
    obs[:, -1] = 1
    episode = dict(policy=dict(observations=obs, actions=np.full((12, 1), -1.), rewards=-np.ones(12)),
                   oracle_diagnostics=dict(daily_response_T_s=np.zeros((12, 10, 64))))
    args = (learner, [episode], [episode], np.arange(4), np.ones(4))
    validation = forecast_validation(*args)
    confirmation = forecast_validation(*args, sols=range(10, 121, 10))
    assert list(confirmation['model_temperature']['at_sols']) == [str(i) for i in range(10, 121, 10)]
    for name, values in validation.items():
        assert confirmation[name]['rmse'] == values['rmse']
        for sol, value in values.get('at_sols', {}).items():
            assert confirmation[name]['at_sols'][sol] == value

import numpy as np
import pytest

from physics.control import HeatingControl, ResponseHeatingControl
from physics.reference import run_truth


def test_zero_heat_has_zero_response_and_exact_two_kelvin_rmse():
    env = ResponseHeatingControl()
    env.reset(5000)
    for _ in range(12):
        transition = env.step(0)
        assert transition['reward'] == -40.0
    assert env.metrics()['tracking_rmse_K'] == 2.0
    assert env.metrics()['heat_used_Wm2_sol'] == 0


def test_response_reward_matches_independent_full_no_heat_trajectory_and_resume():
    env = ResponseHeatingControl()
    initial_obs = env.reset(5001)
    initial = env.sim.state_dict()
    reference = run_truth(env.cfg, 0.0, years=120/env.cfg['planet']['year_sols'],
                          mismatch_scale=.10, start_step=initial['step_index'],
                          storms=initial['storms'], **initial['initial'])
    actions = [7.5, 10, 10, 10] + [0]*8
    first = env.step(actions[0])
    checkpoint = env.state_dict()
    rest = [env.step(a) for a in actions[1:]]
    metrics = env.metrics()
    env.load_state_dict(checkpoint)
    replay = [env.step(a) for a in actions[1:]]
    for expected, restored in zip(rest, replay):
        assert expected['reward'] == restored['reward']
        np.testing.assert_array_equal(expected['daily_T_s'], restored['daily_T_s'])
    assert env.metrics() == metrics
    transitions = [first, *rest]
    assert [t['applied_heat'] for t in transitions] == [7.5, 10, 10, 2.5] + [0]*8
    assert metrics['heat_used_Wm2_sol'] == 300
    daily = np.concatenate([t['daily_T_s'] for t in transitions])
    error = (daily-reference['T_s'][1:])[:, env.band] - 2
    daily_error = np.average(error**2, axis=1, weights=env.grid.w[env.band])
    np.testing.assert_allclose([t['reward'] for t in transitions],
                               -daily_error.reshape(12, 10).sum(axis=1), rtol=1e-14)
    assert metrics['tracking_rmse_K'] == pytest.approx(np.sqrt(daily_error.mean()))
    np.testing.assert_array_equal(np.concatenate([t['daily_t_sol'] for t in transitions]),
                                  reference['t_sol'][1:])
    assert set(initial_obs) == {'T_s', 'S', 'previous_heat', 'Ls', 'remaining_sols', 'remaining_budget'}
    assert set(first) == {'observation', 'reward', 'terminated', 'truncated',
                          'requested_heat', 'applied_heat', 'daily_T_s', 'daily_S', 'daily_t_sol'}
    assert rest[-1]['terminated'] and not rest[-1]['truncated']
    with pytest.raises(RuntimeError):
        env.step(0)
    for bad in [np.nan, np.inf, -1, 11]:
        env.load_state_dict(checkpoint)
        with pytest.raises(ValueError):
            env.step(bad)
    # Invalid actions must not advance either world.
    actual = env.step(10)
    assert actual['reward'] == rest[0]['reward']
    obs = env.reset(5001)
    np.testing.assert_array_equal(obs['T_s'], initial_obs['T_s'])


def test_response_and_original_task_share_only_heated_policy_observations():
    response, original = ResponseHeatingControl(), HeatingControl()
    response.reset(5002)
    original.reset(5002)
    for action in [10, 0, 7.5]:
        a, b = response.step(action), original.step(action)
        for field in a['observation']:
            np.testing.assert_array_equal(a['observation'][field], b['observation'][field])
        np.testing.assert_array_equal(a['daily_T_s'], b['daily_T_s'])
        assert a['reward'] != b['reward']

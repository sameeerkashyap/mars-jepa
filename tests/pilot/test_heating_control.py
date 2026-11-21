import numpy as np
import pytest

from physics.reference import TruthSimulation, run_truth
from physics.surface import load_config


def test_persistent_truth_matches_full_trajectory():
    cfg = load_config()
    full = run_truth(cfg, F_a=lambda t: 1.0 if t < 10 else 4.0,
                     seed=42, years=20 / cfg['planet']['year_sols'])
    sim = TruthSimulation(cfg, seed=42, total_sol=20)
    first = sim.advance(10, 1.0)
    second = sim.advance(10, 4.0)
    for key in ('T_s', 'T_g', 'm', 'tau', 'S', 't_sol'):
        joined = np.concatenate([first[key], second[key][1:]])
        np.testing.assert_array_equal(joined, full[key])
    np.testing.assert_allclose(first['netflux_integral'] + second['netflux_integral'],
                               full['netflux_integral'], rtol=1e-14)


def test_control_budget_observations_and_resume():
    from physics.control import HeatingControl

    env = HeatingControl()
    obs = env.reset(seed=5000)
    assert set(obs) == {'T_s', 'S', 'previous_heat', 'Ls', 'remaining_sols', 'remaining_budget'}
    assert obs['T_s'].shape == (64,)
    checkpoint = env.state_dict()
    actions = [10] * 12
    first = [env.step(a) for a in actions]
    assert [t['applied_heat'] for t in first] == [10, 10, 10] + [0] * 9
    assert sum(t['applied_heat'] * 10 for t in first) == 300
    assert not any(t['terminated'] for t in first[:-1])
    assert first[-1]['terminated'] and not first[-1]['truncated']
    assert first[-1]['observation']['remaining_budget'] == 0
    assert first[-1]['observation']['remaining_sols'] == 0
    assert all(t['reward'] <= 0 for t in first)
    daily = np.concatenate([t['daily_T_s'] for t in first])
    band = np.degrees(env.grid.lat) < -60
    expected = np.sqrt(np.average((daily[:, band] - env.target)**2,
                                  axis=1, weights=env.grid.w[band]).mean())
    assert env.metrics()['tracking_rmse_K'] == pytest.approx(expected)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.load_state_dict(checkpoint)
    second = [env.step(a) for a in actions]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a['daily_T_s'], b['daily_T_s'])
        assert a['reward'] == b['reward']
    env.load_state_dict(checkpoint)
    env.step(7.5)
    mid_episode = env.state_dict()
    expected_next = env.step(3.25)
    env.load_state_dict(mid_episode)
    restored_next = env.step(3.25)
    np.testing.assert_array_equal(expected_next['daily_T_s'], restored_next['daily_T_s'])
    assert restored_next['observation']['remaining_budget'] == 192.5
    for bad in [np.nan, np.inf, -1, 11]:
        env.load_state_dict(checkpoint)
        with pytest.raises(ValueError):
            env.step(bad)


@pytest.mark.parametrize('actions', [[0]*12, [10]*12, [0, 10]*6, [2.5]*12])
def test_control_schedules_match_monolithic_truth(actions):
    from physics.control import HeatingControl

    env = HeatingControl()
    env.reset(seed=5001)
    initial = env.sim.initial.copy()
    start_step = env.sim.step_index
    start_sol = env.sim.t_sol
    remaining = 300
    applied = []
    for action in actions:
        heat = min(action, remaining / 10)
        applied.append(heat)
        remaining -= 10 * heat

    def forcing(t):
        index = min(int(round((t - start_sol) / .05)) // 200, 11)
        return applied[index]

    full = run_truth(env.cfg, forcing, years=120 / env.cfg['planet']['year_sols'],
                     mismatch_scale=.10, start_step=start_step, storms=env.sim.storms,
                     **initial)
    stepped = [env.step(a) for a in actions]
    np.testing.assert_array_equal(np.concatenate([s['daily_T_s'] for s in stepped]),
                                  full['T_s'][1:])
    assert env.remaining_budget == pytest.approx(remaining)
    assert [s['applied_heat'] for s in stepped] == applied


def test_reset_isolation_and_south_winter():
    from physics.control import HeatingControl, south_winter_start

    env = HeatingControl()
    first = env.reset(5002)
    env.step(10)
    second = env.reset(5002)
    np.testing.assert_array_equal(first['T_s'], second['T_s'])
    assert second['remaining_budget'] == 300
    assert second['previous_heat'] == 0
    assert 200 < south_winter_start(env.cfg) < 500
    # Observations own their arrays; mutating a measurement cannot alter truth.
    second['T_s'][:] = 0
    np.testing.assert_array_equal(env.observation()['T_s'], first['T_s'])


def test_reserved_conditions_are_rejected():
    from physics.control import HeatingControl

    with pytest.raises(ValueError, match='reserved'):
        HeatingControl().reset(9000)

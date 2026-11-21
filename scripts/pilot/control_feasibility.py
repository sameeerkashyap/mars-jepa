"""Small development-only truth screen; no learning or held-out data access.

Run with python -m scripts.pilot.control_feasibility --out <small JSON path>.
"""

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from physics.control import HeatingControl, ResponseHeatingControl
from physics.surface import frost_point, insolation


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = ['src/physics/reference.py', 'src/physics/control.py',
             'src/physics/surface.py', 'data/specification/pilot/config.yaml',
             'scripts/pilot/control_feasibility.py']
    return dict(source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},
                python=sys.version, numpy=version('numpy'), scipy=version('scipy'),
                command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m '
                        'scripts.pilot.control_feasibility --out docs/pilot/results/control_feasibility.json')


def screen_later_windows():
    """Exploratory onset offsets; keep original storms and accepted v1 intact."""
    wall, cpu = time.perf_counter(), time.process_time()
    env = HeatingControl()
    annual_max = max(np.average(insolation(env.grid.x, d, env.cfg['planet'])[env.band],
                                weights=env.grid.w[env.band]) for d in range(669))
    rows = []
    for seed in range(5000, 5005):
        env.reset(seed)
        initial = env.state_dict()
        for offset in (20, 40):
            env.load_state_dict(initial)
            # Keep precisely the original finite disturbance draw for paired windows.
            env.sim.total_steps += env.sim._steps(offset)
            env.sim.advance(offset, 0.0, store=False)
            start = env.sim.t_sol
            assert all(np.average(insolation(env.grid.x, start+d, env.cfg['planet'])[env.band],
                                   weights=env.grid.w[env.band]) <= .1*annual_max
                       for d in range(121))
            window = env.state_dict()
            schedules = {'zero': [0.0]*12, 'constant': [2.5]*12,
                         'seasonal': [0.0]*4 + [10.0]*3 + [0.0]*5}
            schedules.update({f'truth_pulse_{s:02d}': [10.0 if s <= i < s+3 else 0.0
                                                       for i in range(12)] for s in range(10)})
            for name in [*schedules, 'threshold']:
                env.load_state_dict(window)
                for i in range(12):
                    if name == 'threshold':
                        obs = env.observation()
                        temperature = np.average(obs['T_s'][env.band], weights=env.grid.w[env.band])
                        action = 10.0 if temperature < env.target else 0.0
                    else:
                        action = schedules[name][i]
                    env.step(action)
                assert env.metrics()['heat_used_Wm2_sol'] <= 300
                rows.append(dict(seed=seed, start_offset_sol=offset, controller=name,
                                 **env.metrics()))
    means = {offset: {name: float(np.mean([r['tracking_rmse_K'] for r in rows
                                          if r['start_offset_sol'] == offset and r['controller'] == name]))
                      for name in [*schedules, 'threshold']} for offset in (20, 40)}
    return dict(exploratory=True, accepted_task_changed=False,
                disturbance_rule='Original onset-screen draw; no redraw beyond original nominal horizon',
                development_seeds=list(range(5000, 5005)), start_offsets_sol=[20, 40],
                mean_tracking_rmse_K=means, rows=rows,
                wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                allocated_cpu_cores=1, gpu_hours=0, agent_training_admitted=False,
                provenance=provenance())


def screen(response=False):
    wall, cpu = time.perf_counter(), time.process_time()
    env = ResponseHeatingControl() if response else HeatingControl()
    co2 = env.cfg['co2']
    threshold = frost_point(co2['surface_pressure_pa'], co2) + 2
    schedules = {'zero': [0.0]*12, 'constant': [2.5]*12,
                 'seasonal': [0.0]*4 + [10.0]*3 + [0.0]*5}
    for start in range(10):
        schedules[f'truth_pulse_{start:02d}'] = [10.0 if start <= i < start+3 else 0.0
                                                for i in range(12)]
    assert all(len(a) == 12 and 10*sum(a) <= 300 for a in schedules.values())
    rows = []
    for seed in range(5000, 5005):
        env.reset(seed)
        initial = env.state_dict()
        for name, actions in schedules.items():
            env.load_state_dict(initial)
            for action in actions:
                env.step(action)
            metrics = env.metrics()
            assert metrics['heat_used_Wm2_sol'] <= 300
            rows.append(dict(seed=seed, controller=name, actions_Wm2=actions, **metrics))
        env.load_state_dict(initial)
        actions = []
        for _ in range(12):
            obs = env.observation()
            temperature = np.average(obs['T_s'][env.band], weights=env.grid.w[env.band])
            action = 10.0 if temperature < threshold else 0.0
            actions.append(action)
            env.step(action)
        rows.append(dict(seed=seed, controller='threshold', actions_Wm2=actions,
                         **env.metrics()))
    means = {name: float(np.mean([r['tracking_rmse_K'] for r in rows
                                  if r['controller'] == name]))
             for name in [*schedules, 'threshold']}
    best = min((name for name in schedules if name.startswith('truth_pulse')), key=means.get)
    result = dict(task='south-winter-v1', development_seeds=list(range(5000, 5005)),
                start_sol=env.start_sol, winter_day_in_year=env.start_sol % env.cfg['planet']['year_sols'],
                target_K=env.target, target_offset_K=2, heat_budget_Wm2_sol=300,
                mean_tracking_rmse_K=means, privileged_search_best=best,
                privileged_search_gain_vs_constant=1-means[best]/means['constant'],
                search_scope='10 consecutive three-interval full-budget pulses; selected on development truth',
                rows=rows, wall_seconds=time.perf_counter()-wall,
                cpu_seconds=time.process_time()-cpu, allocated_cpu_cores=1,
                gpu_hours=0, agent_training_admitted=False, provenance=provenance())
    if response:
        eligible = ('zero', 'constant', 'seasonal', 'threshold')
        strongest = min(eligible, key=means.get)
        result.update(task='south-winter-response-v1', data_namespace='control-response-v1',
                      target_K=None, response_target_K=2.0,
                      reference_rule='Matched private no-heat truth; never a policy input',
                      threshold_rule='Public South-band temperature below fixed frost point +2 K',
                      strongest_eligible_controller=strongest,
                      privileged_search_gain_vs_strongest=1-means[best]/means[strongest])
        result['provenance']['command'] = ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m '
                                          'scripts.pilot.control_feasibility --response --out '
                                          'docs/pilot/results/control_response_feasibility.json')
        for row in rows:
            remaining = 300.0
            applied = []
            for requested in row['actions_Wm2']:
                heat = min(requested, remaining/10)
                applied.append(heat)
                remaining -= heat*10
            row['applied_actions_Wm2'] = applied
            assert 300-remaining == row['heat_used_Wm2_sol']
        result['allocated_cpu_core_hours'] = result['wall_seconds']/3600
        result['process_cpu_hours'] = result['cpu_seconds']/3600
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--later-windows', action='store_true')
    mode.add_argument('--response', action='store_true')
    args = parser.parse_args()
    result = screen_later_windows() if args.later_windows else screen(response=args.response)
    if args.later_windows:
        result['provenance']['command'] = ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m '
                                          'scripts.pilot.control_feasibility --later-windows --out '
                                          'docs/pilot/results/control_later_windows.json')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()

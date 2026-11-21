"""Bounded privileged development reference, starting from constant heat."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from physics.control import ResponseHeatingControl
from scripts.pilot.control_feasibility import provenance


def search():
    wall, cpu = time.perf_counter(), time.process_time()
    env = ResponseHeatingControl()
    initial = {}
    for seed in range(5000, 5005):
        env.reset(seed)
        initial[seed] = env.state_dict()
    candidates, seen = [], {}

    def evaluate(actions):
        key = tuple(float(a) for a in actions)
        if key in seen:
            return seen[key]
        assert len(key) == 12 and all(0 <= a <= 10 for a in key)
        rows = []
        for seed, state in initial.items():
            env.load_state_dict(state)
            transitions = [env.step(action) for action in key]
            metrics = env.metrics()
            applied = [t['applied_heat'] for t in transitions]
            assert metrics['heat_used_Wm2_sol'] <= 300
            assert np.isclose(10*sum(applied), metrics['heat_used_Wm2_sol'])
            rows.append(dict(seed=seed, applied_actions_Wm2=applied, **metrics))
        score = float(np.mean([r['tracking_rmse_K'] for r in rows]))
        candidates.append(dict(candidate=len(candidates), requested_actions_Wm2=list(key),
                               mean_tracking_rmse_K=score, rows=rows))
        seen[key] = score
        return score

    result = minimize(evaluate, np.full(12, 2.5), method='SLSQP',
                      bounds=[(0.0, 10.0)]*12,
                      constraints={'type': 'ineq', 'fun': lambda a: 30.0-np.sum(a)},
                      options={'eps': .05, 'maxiter': 10, 'ftol': 1e-6})
    evaluate(result.x)
    # Finite-difference probes can cross the requested integral constraint.
    feasible = [c for c in candidates if sum(c['requested_actions_Wm2']) <= 30+1e-8]
    best = min(feasible, key=lambda c: c['mean_tracking_rmse_K'])
    baseline = candidates[0]['mean_tracking_rmse_K']
    info = provenance()
    source = 'scripts/pilot/control_response_search.py'
    info['source_sha256'][source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info['command'] = ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m '
                       'scripts.pilot.control_response_search --out '
                       'docs/pilot/results/control_response_search.json')
    elapsed_wall, elapsed_cpu = time.perf_counter()-wall, time.process_time()-cpu
    return dict(task='south-winter-response-v1', data_namespace='control-response-v1',
                development_seeds=list(initial), privileged=True,
                search_scope='12 ten-sol controls, SLSQP eps=.05, maxiter=10; initial constant 2.5',
                optimizer_success=bool(result.success), optimizer_message=str(result.message),
                optimizer_iterations=int(result.nit), objective_evaluations=len(candidates),
                constant_tracking_rmse_K=baseline, best_candidate=best['candidate'],
                best_tracking_rmse_K=best['mean_tracking_rmse_K'],
                gain_vs_constant=1-best['mean_tracking_rmse_K']/baseline,
                candidates=candidates, wall_seconds=elapsed_wall, cpu_seconds=elapsed_cpu,
                allocated_cpu_cores=1, allocated_cpu_core_hours=elapsed_wall/3600,
                process_cpu_hours=elapsed_cpu/3600, gpu_hours=0,
                agent_training_admitted=False, provenance=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = search()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'candidates'}, indent=2))


if __name__ == '__main__':
    main()

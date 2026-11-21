"""Check two lower-budget constant baselines on development conditions only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from physics.control import ResponseHeatingControl
from scripts.pilot.control_feasibility import provenance


def screen():
    wall, cpu = time.perf_counter(), time.process_time()
    env = ResponseHeatingControl()
    rows = []
    for seed in range(5000, 5005):
        env.reset(seed)
        initial = env.state_dict()
        for heat in (2.0, 2.25):
            env.load_state_dict(initial)
            transitions = [env.step(heat) for _ in range(12)]
            metrics = env.metrics()
            assert metrics['heat_used_Wm2_sol'] == heat*120
            assert all(t['applied_heat'] == heat for t in transitions)
            rows.append(dict(seed=seed, constant_heat_Wm2=heat,
                             requested_actions_Wm2=[heat]*12,
                             applied_actions_Wm2=[t['applied_heat'] for t in transitions], **metrics))
    info = provenance()
    source = 'scripts/pilot/control_response_constants.py'
    info['source_sha256'][source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    info['command'] = ('OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m '
                       'scripts.pilot.control_response_constants --out '
                       'docs/pilot/results/control_response_constants.json')
    elapsed_wall, elapsed_cpu = time.perf_counter()-wall, time.process_time()-cpu
    return dict(task='south-winter-response-v1', data_namespace='control-response-v1',
                development_seeds=list(range(5000, 5005)), rows=rows,
                mean_tracking_rmse_K={str(h): float(np.mean([r['tracking_rmse_K'] for r in rows
                                                            if r['constant_heat_Wm2'] == h]))
                                      for h in (2.0, 2.25)},
                wall_seconds=elapsed_wall, cpu_seconds=elapsed_cpu, allocated_cpu_cores=1,
                allocated_cpu_core_hours=elapsed_wall/3600, process_cpu_hours=elapsed_cpu/3600,
                gpu_hours=0, agent_training_admitted=False, provenance=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = screen()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()

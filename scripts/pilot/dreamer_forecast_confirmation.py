"""One-use confirmation of heating-response forecasts on conditions 5300–5331."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import tarfile
from types import SimpleNamespace

import numpy as np
import torch

from scripts.pilot import dreamer_control_worker as worker
from scripts.pilot.dreamer_control import digest
from scripts.pilot.small_study import write_json

SOURCE = 'cd7b517b33fe18b26df2ffa4fd1a972540b6316ea3d180491b6dfff2b324c556'
CONFIG_SHA = 'ad2e246782449eebf9e555075f2f66368cabd80c43a7443084e04dd5dd4b6258'
DATA = ['c5899c45029eff57bb7603802bd3e984edead82ade4824c1ed8c449207b65a2a',
        '78c1a3ebd9d9af9bab6f1220fc35c220c934ee16c5e77547596226df31bbba0d']
RECIPE_SHA = digest(dict(learning_rate=.0003, horizon=3))
CHECKPOINTS = [500, 1000, 2000, 5000, 10000]

SEEDS = tuple(range(201, 206))
CONDITIONS = tuple(range(5300, 5332))
COMPARISONS = {'response': 'linear_response', 'temperature': 'persistence_temperature',
               'reward': 'linear_reward'}


def forecast_report(records):
    """Resample training seeds and shared conditions independently, keeping schedules paired."""
    if set(records) != set(SEEDS) or any(set(rows) != set(CONDITIONS) for rows in records.values()):
        raise ValueError('All five training seeds and all 32 forecast conditions are required')
    rng = np.random.default_rng(6100)
    seed_indices = rng.integers(0, len(SEEDS), (10000, len(SEEDS), 1))
    condition_indices = rng.integers(0, len(CONDITIONS), (10000, 1, len(CONDITIONS)))
    result = {'per_seed': {s: {} for s in SEEDS}, 'comparisons': {}}
    for outcome, baseline_name in COMPARISONS.items():
        def mse_matrix(name):
            errors = np.array([[records[s][c][name]['rmse'] for c in CONDITIONS] for s in SEEDS])
            if not np.isfinite(errors).all() or (errors < 0).any():
                raise ValueError('Invalid condition RMSE')
            return errors ** 2

        model, baseline = mse_matrix('model_' + outcome), mse_matrix(baseline_name)
        if not np.array_equal(baseline, np.broadcast_to(baseline[0], baseline.shape)):
            raise ValueError('Shared condition baseline changed between training seeds')

        def scores(m, b):
            model_rmse, baseline_rmse = float(np.sqrt(m.mean())), float(np.sqrt(b.mean()))
            return dict(model_rmse=model_rmse, baseline_rmse=baseline_rmse,
                        gain=1 - model_rmse / baseline_rmse if baseline_rmse else None)

        for i, seed in enumerate(SEEDS):
            result['per_seed'][seed][outcome] = scores(model[i], baseline[i])
        pooled = scores(model, baseline)
        denom = baseline[seed_indices, condition_indices].mean((1, 2))
        if (denom > 0).all():
            gains = 1 - np.sqrt(model[seed_indices, condition_indices].mean((1, 2)) / denom)
            pooled['bootstrap_95'] = np.quantile(gains, [.025, .975]).tolist()
        else:
            pooled['bootstrap_95'] = None
        result['comparisons'][outcome] = pooled
    result['at_sols'] = {}
    for name, metric in records[SEEDS[0]][CONDITIONS[0]].items():
        if 'at_sols' in metric:
            result['at_sols'][name] = {
                sol: float(np.sqrt(np.mean([records[s][c][name]['at_sols'][sol] ** 2
                                            for s in SEEDS for c in CONDITIONS])))
                for sol in metric['at_sols']}
    response = result['comparisons']['response']
    result['passed'] = (response['gain'] is not None and response['model_rmse'] <= .9 * response['baseline_rmse']
                        and response['bootstrap_95'] is not None and response['bootstrap_95'][0] > 0)
    return result


def open_forecast(out, runs, data_hashes, evaluator_hashes):
    if [r['seed'] for r, _ in runs] != list(SEEDS) or data_hashes != DATA:
        raise ValueError('All five fresh seeds and original training and validation data are required')
    frozen = dict(source_sha256=SOURCE, data_sha256=data_hashes, config_sha256=CONFIG_SHA,
                  evaluator_sha256=evaluator_hashes, conditions=list(CONDITIONS),
                  bootstrap_draws=10000, bootstrap_seed=6100, minimum_gain=.1, runs=[])
    for row, path in runs:
        identity = dict(source_sha256=SOURCE, data_sha256=DATA, config_sha256=CONFIG_SHA,
                        recipe_sha256=RECIPE_SHA, mode=row['mode'], variant=row['variant'],
                        seed=row['seed'], rung=24)
        curves = [c for c in row['curves'] if c['phase'] == 'model']
        if ((row['mode'], row['variant']) not in (('model', 'fast'), ('confirm', 'selected'))
                or any(row[k] != v for k, v in identity.items())
                or [c['updates'] for c in curves] != CHECKPOINTS
                or not all(np.isfinite(c['validation']['model_response']['rmse']) for c in curves)):
            raise ValueError('Incomplete model validation curve or wrong recipe identity')
        best = min(curves, key=lambda c: c['validation']['model_response']['rmse'])
        sha = worker.file_hash(path)
        if row['mode'] == 'model' and row['checkpoint_sha256'] != sha:
            raise ValueError('Selected checkpoint checksum mismatch')
        saved = torch.load(path, map_location='cpu', weights_only=False)['pilot']
        if (saved['identity'] != identity or saved['model_updates'] != best['updates']
                or saved['behavior_updates'] != 0 or saved['best_model_metrics'] != best['validation']):
            raise ValueError('Checkpoint does not match the validation-selected model')
        frozen['runs'].append(dict(identity=identity, checkpoint=str(path), checkpoint_sha256=sha,
                                   updates=best['updates'], validation=best['validation']))
    # Creation spends the single evaluation, including an interruption during generation or scoring.
    with (out / 'FROZEN.json').open('x') as file:
        json.dump(frozen, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    return frozen


def archived_config(archive):
    root = Path(worker.__file__).resolve().parents[2]
    if worker.file_hash(archive) != SOURCE:
        raise ValueError('Original training source archive checksum mismatch')
    with tarfile.open(archive, 'r:gz') as bundle:
        for entry in bundle.getmembers():
            if entry.isfile() and (root / entry.name).read_bytes() != bundle.extractfile(entry).read():
                raise ValueError(f'Extracted source differs from original archive: {entry.name}')
    config = json.loads((root / 'data/specification/pilot/dreamer_control.json').read_text())
    config['model_checkpoints'] = CHECKPOINTS
    if digest(config) != CONFIG_SHA:
        raise ValueError('Long-run model configuration checksum mismatch')
    return config


def trained_models(base, out, config):
    runs = []
    for seed in SEEDS:
        print(f'Preparing model seed {seed}', flush=True)
        # Only the exact long recipe with horizon 3 can identify the same model initialization.
        original = base / 'followup-v1' / f'long-confirm-selected-s{seed}'
        if (original / 'result.json').exists():
            record = json.loads((original / 'result.json').read_text())
            row = record['result']
            if (record['status']['status'] == 'ok' and row.get('recipe_sha256') == RECIPE_SHA
                    and row.get('config_sha256') == CONFIG_SHA and row.get('seed') == seed):
                runs.append((row, original / 'model-best.pt'))
                continue
        train_dir = out / f'model-s{seed}'
        train_dir.mkdir(exist_ok=True)
        path = train_dir / 'model.json'
        if not path.exists():
            args = SimpleNamespace(mode='model', seed=seed, rung=24, variant='fast', base=base,
                                   out=train_dir, source=SOURCE)
            write_json(path, worker.train_stage(args, config))
        runs.append((json.loads(path.read_text()), train_dir / 'best.pt'))
    return runs


def run(base, archive, scorer):
    import data.control as data

    config = archived_config(archive)
    out = base / 'forecast-confirmation-v1'
    out.mkdir(exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'FROZEN.json').exists():
            raise FileExistsError('Forecast evaluation already opened. Retain the existing result.')
        hashes, training, _ = worker.datasets(base, 24)
        if hashes != DATA:
            raise ValueError('Original train/validation data changed')
        files = (Path(__file__), Path(scorer.__file__))
        evaluator = {path.name: worker.file_hash(path) for path in files}
        runs = trained_models(base, out, config)
        frozen = open_forecast(out, runs, hashes, evaluator)
        frozen_hash = worker.file_hash(out / 'FROZEN.json')
        print(f'FROZEN_SHA256={frozen_hash}', flush=True)
        # Register this split only inside this worker, after spending the one-use evaluation.
        data.SPLITS['forecast-confirmation'] = CONDITIONS
        dataset = out / 'conditions'
        data.generate_dataset(dataset, split='forecast-confirmation', allocated_cpu_cores=2)
        manifest, episodes = data.load_dataset(dataset, split='forecast-confirmation')
        if tuple(manifest['condition_seeds']) != CONDITIONS:
            raise ValueError('Wrong forecast condition IDs')
        grouped = {c: [] for c in CONDITIONS}
        for row, episode in zip(manifest['episodes'], episodes, strict=True):
            grouped[row['condition_seed']].append(episode)

        env = worker.ClimateVectorEnv()
        band = env.task.band
        results = {}
        for selected in frozen['runs']:
            seed = selected['identity']['seed']
            print(f'Scoring frozen model seed {seed}', flush=True)
            path = Path(selected['checkpoint'])
            if worker.file_hash(path) != selected['checkpoint_sha256']:
                raise ValueError('Frozen checkpoint changed')
            learner, _ = worker.recover(path, selected['identity'])
            results[seed] = {
                c: scorer.forecast_validation(learner, training, grouped[c], band, env.task.grid.w[band],
                                              sols=range(10, 121, 10)) for c in CONDITIONS
            }
            write_json(out / f'forecast-s{seed}.json', results[seed])
        report = forecast_report(results)
        report.update(frozen=frozen, frozen_sha256=frozen_hash,
                      forecast_data_sha256=manifest['data_sha256'])
        write_json(out / 'report.json', report)
        return report


def main():
    import time

    # In production these two evaluator files share one read-only mount outside the archived tree.
    if __package__:
        from . import dreamer_control_forecast as scorer
    else:
        import dreamer_control_forecast as scorer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    result = run(args.base.resolve(), args.archive, scorer)
    status = dict(status='ok', device='cpu', allocated_cpu_cores=2, source_sha256=SOURCE,
                  wall_seconds=time.perf_counter() - start, versions=worker.versions())
    print('RESULT_JSON=' + json.dumps(dict(status=status, result=result), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

"""Concrete CPU stages. All expensive results and recovery files stay on the PVC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
from benchmarks.dreamer_control.climate import ClimateVectorEnv, climate_imagination
from data.control import generate_dataset, load_dataset
from benchmarks.dreamer_control.learner import versions
from scripts.pilot.dreamer_control_forecast import forecast_validation, readiness, fit_response_baseline
from scripts.pilot.dreamer_paired_response import matched_training_pairs, paired_response_loss
from scripts.pilot.dreamer_control import digest, freeze_confirmation, recipe_digest
from scripts.pilot.small_study import write_json


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_learner(config, seed, rate, horizon=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return DreamerLearner(LearnerConfig(**{**config['learner'], 'learning_rate': rate,
                                          'imagination_horizon': horizon}),
                          imagination_env_factory=climate_imagination)


def save_recovery(learner, path, driver):
    """Commit weights, RNG, replay and driver cursor in one atomic checkpoint."""
    temporary = path.with_suffix('.pending')
    learner.save(temporary)
    state = torch.load(temporary, weights_only=False)
    state['pilot'] = driver
    torch.save(state, temporary)
    with temporary.open('rb') as file:
        os.fsync(file.fileno())
    temporary.replace(path)


def recover(path, identity):
    state = torch.load(path, weights_only=False)
    driver = state['pilot']
    if driver['identity'] != identity:
        raise ValueError('Recovery source/config/data/seed identity changed')
    return DreamerLearner.load(path, imagination_env_factory=climate_imagination), driver


def run_baseline(env, name):
    plan = name.get('simple_model_plan') if isinstance(name, dict) else None
    name = name['selected'] if isinstance(name, dict) else name
    reset = env.observation().copy()
    for step in range(12):
        if name == 'no_heat':
            heat = 0.
        elif name.startswith('constant_'):
            heat = float(name.split('_')[1])
        elif name == 'simple_model':
            heat = plan[step]
        elif name == 'seasonal':
            heat = 10. if step >= 9 else 0.
        elif name == 'feedback':
            change = np.average((env.observation()[:64]-reset[:64])[env.task.band]*10,
                                weights=env.task.grid.w[env.task.band])
            heat = 10. if change < 2 else 0.
        else:
            raise ValueError('Unknown eligible baseline')
        env.step(torch.tensor([heat/5-1]))
    return env.metrics()


def simple_model_plan(coefficients, band, weights):
    from scipy.optimize import minimize

    def objective(heats):
        previous, total = np.zeros(64), 0.
        for heat in heats:
            response = coefficients[:, 0]*previous + coefficients[:, 1]*heat
            daily = np.linspace(previous, response, 11)[1:]
            total += np.average((daily[:, band]-2)**2, axis=-1, weights=weights).sum()
            previous = response
        return total

    fit = minimize(objective, np.full(12, 2.5), method='SLSQP', bounds=[(0, 10)]*12,
                   constraints=[{'type': 'ineq', 'fun': lambda heat: 300.-10*heat.sum()}],
                   options={'maxiter': 200, 'ftol': 1e-8})
    if not fit.success or not np.isfinite(fit.x).all() or fit.x.sum()*10 > 300+1e-5:
        raise RuntimeError(f'Simple-model planning failed: {fit.message}')
    return np.clip(fit.x, 0, 10).tolist()


def select_baseline(config, training):
    env = ClimateVectorEnv()
    coefficients = fit_response_baseline(training)
    plan = simple_model_plan(coefficients, env.task.band, env.task.grid.w[env.task.band])
    env, scores = ClimateVectorEnv(), {name: [] for name in config['baseline_candidates']}
    for seed in range(5100, 5108):
        env.reset(seed=seed)
        initial = env.state_dict()
        for name in scores:
            env.load_state_dict(initial)
            scores[name].append(run_baseline(env, {'selected': name, 'simple_model_plan': plan})['tracking_rmse_K'])
    means = {name: float(np.mean(values)) for name, values in scores.items()}
    return dict(selected=min(means, key=means.get), validation_means=means, validation_rows=scores,
                simple_model_plan=plan, simple_model_coefficients=coefficients.tolist(),
                fit_conditions=sorted({5000+i for i in range(len(training)//4)}),
                planning='Training-only linear response recurrence; bounded SLSQP, no future truth calls')


def evaluate(learner, seeds, baseline=None, initial_states=None, trajectory_dir=None):
    env, rows = ClimateVectorEnv(), []
    initial_states = {} if initial_states is None else initial_states
    for seed in seeds:
        if seed not in initial_states:
            env.reset(seed=seed)
            initial_states[seed] = env.state_dict()
        initial = initial_states[seed]
        env.load_state_dict(initial)
        observation, carry, daily, heats = env.observation(), None, [], []
        for _ in range(12):
            action, carry = learner.act(observation, carry, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            carry['previous_action'] = info['dynamics_action']
            heats.append(info['applied_heat'])
            daily.extend(info['daily_T_s'])
        metrics = env.metrics()
        violation = int(any(h < 0 or h > 10 for h in heats) or sum(heats)*10 > 300+1e-5
                        or not terminated or truncated or observation[-2] != 0)
        row = dict(condition_seed=seed, violations=violation, applied_heat_Wm2=heats,
                   whole_temperature_min_K=float(np.min(daily)),
                   whole_temperature_max_K=float(np.max(daily)),
                   whole_temperature_mean_K=float(np.mean(daily)), **metrics)
        if trajectory_dir is not None:
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            with (trajectory_dir/f'{seed}.npz').open('xb') as file:
                np.savez_compressed(file, daily_T_s=np.asarray(daily), applied_heat_Wm2=np.asarray(heats))
        if baseline:
            env.load_state_dict(initial)
            row['baseline_metrics'] = run_baseline(env, baseline)
            row['baseline_rmse_K'] = row['baseline_metrics']['tracking_rmse_K']
        rows.append(row)
    return rows


def datasets(base, rung, *, include_train_manifest=False):
    train_manifest, training = load_dataset(base/f'data-{rung}'/'train', split='train')
    val_manifest, validation = load_dataset(base/'data-12'/'validation', split='validation')
    data = ([train_manifest['data_sha256'], val_manifest['data_sha256']], training, validation)
    return (*data, train_manifest) if include_train_manifest else data


def train_stage(args, config):
    weight = config.get('paired_response_weight', 0.)
    selected = json.loads(os.environ.get('MARSWORLD_RECIPE', '{}'))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('paired_response_weight must be finite and nonnegative')
    if args.mode != 'model' and selected.get('paired_response_weight', weight) != weight:
        raise ValueError('selected recipe and configuration disagree on paired_response_weight')
    hashes, training, validation, train_manifest = datasets(args.base, args.rung, include_train_manifest=True)
    rate = config['model_learning_rates'][args.variant] if args.mode == 'model' else selected['learning_rate']
    horizon = 3 if args.mode == 'model' else int(args.variant[1:]) if args.mode == 'actor' else selected['horizon']
    identity = dict(source_sha256=args.source, config_sha256=digest(config), data_sha256=hashes,
                    seed=args.seed, mode=args.mode, variant=args.variant, rung=args.rung,
                    recipe_sha256=recipe_digest(dict(learning_rate=rate, horizon=horizon,
                                                     paired_response_weight=weight)))
    if weight:
        identity['paired_response_weight'] = weight
    path = args.out/'resume.pt'
    if path.exists():
        learner, driver = recover(path, identity)
    else:
        learner = new_learner(config, args.seed, rate, horizon)
        for episode in training:
            learner.add_episode(**episode['policy'])
        driver = dict(identity=identity, seed_cursor=0, model_updates=0, behavior_updates=0,
                      curves=[], best_score=None, best_metrics=None, versions=versions(),
                      resolved_learner_config=asdict(learner.config))
    env = ClimateVectorEnv()
    band, weights = env.task.band, env.task.grid.w[env.task.band]
    if weight:
        pairs = matched_training_pairs(train_manifest, training, band)
        area = torch.as_tensor(weights, dtype=torch.float32)
    start = time.perf_counter()
    initial_states = {}
    model_checkpoints = config['model_checkpoints']
    # Every confirmation is a fresh initialization, including model and behavior.
    for step in range(driver['model_updates'], model_checkpoints[-1]):
        auxiliary = None
        if weight:
            auxiliary = weight * paired_response_loss(learner, *pairs[step % len(pairs)], band, area)
        loss = learner.update(model_only=True, paired_response_loss=auxiliary)
        driver['model_updates'] += 1
        if driver['model_updates'] in model_checkpoints:
            metrics = forecast_validation(learner, training, validation, band, weights)
            score = metrics['model_response']['rmse']
            driver['curves'].append(dict(phase='model', updates=driver['model_updates'],
                                        validation=metrics, loss=loss, elapsed_seconds=time.perf_counter()-start))
            if driver.get('best_model_score') is None or score < driver['best_model_score']:
                driver['best_model_score'] = score
                driver['best_model_metrics'] = metrics
                save_recovery(learner, args.out/'model-best.pt', driver)
            if args.mode == 'model' and (driver['best_score'] is None or score < driver['best_score']):
                driver.update(best_score=score, best_metrics=metrics)
                save_recovery(learner, args.out/'best.pt', driver)
            save_recovery(learner, path, driver)
            write_json(args.out/'curves.json', driver['curves'])
    if args.mode != 'model':
        if not readiness(driver['best_model_metrics'], config['readiness_response_improvement']):
            return dict(**identity, readiness_failed=True, actor_updates=0, curves=driver['curves'],
                        validation=driver['best_model_metrics'], seed_cursor=driver['seed_cursor'])
        if driver['behavior_updates'] == 0:
            learner, _ = recover(args.out/'model-best.pt', identity)
            save_recovery(learner, path, driver)
        for _ in range(driver['behavior_updates'], config['behavior_checkpoints'][-1]):
            if driver['behavior_updates'] % config['collect_every_behavior_updates'] == 0:
                condition = 5000 + driver['seed_cursor'] % args.rung
                learner.collect_episode(env, seed=condition)
                driver['seed_cursor'] += 1
            loss = learner.update()
            driver['behavior_updates'] += 1
            if driver['behavior_updates'] in config['behavior_checkpoints']:
                rows = evaluate(learner, range(5100, 5108), initial_states=initial_states)
                score = float(np.mean([row['tracking_rmse_K'] for row in rows]))
                driver['curves'].append(dict(phase='behavior', updates=driver['behavior_updates'],
                                            validation_rmse_K=score, rows=rows, loss=loss,
                                            elapsed_seconds=time.perf_counter()-start))
                if driver['best_score'] is None or score < driver['best_score']:
                    driver.update(best_score=score, best_metrics=rows)
                    save_recovery(learner, args.out/'best.pt', driver)
                save_recovery(learner, path, driver)
                write_json(args.out/'curves.json', driver['curves'])
    result = dict(**identity, checkpoint=str((args.out/'best.pt').relative_to(args.base)),
                  checkpoint_sha256=file_hash(args.out/'best.pt'), curves=driver['curves'],
                  validation_rmse_K=driver['best_score'], validation=driver['best_metrics'],
                  versions=versions(), seed_cursor=driver['seed_cursor'],
                  resolved_learner_config=driver['resolved_learner_config'])
    if args.mode == 'model':
        result['ready'] = readiness(driver['best_metrics'], config['readiness_response_improvement'])
    return result


def meter(args, config):
    train_dir, val_dir = args.out/'meter-train', args.out/'meter-validation'
    train = generate_dataset(train_dir, split='train', condition_seeds=[5000], allocated_cpu_cores=2)
    validation = generate_dataset(val_dir, split='validation', condition_seeds=[5100], allocated_cpu_cores=2)
    _, episodes = load_dataset(train_dir, split='train')
    _, validation_episodes = load_dataset(val_dir, split='validation')
    learner = new_learner(config, 101, config['model_learning_rates']['base'])
    for episode in episodes:
        learner.add_episode(**episode['policy'])
    times = {}
    started = time.perf_counter()
    for _ in range(12):
        learner.update(model_only=True)
    times['model_update_seconds'] = (time.perf_counter()-started)/12
    # Compatibility behavior updates use synthetic vectors, never climate episodes.
    synthetic = new_learner(config, 103, config['model_learning_rates']['base'])
    rng = np.random.default_rng(103)
    observations = rng.normal(0, .1, (13, 133)).astype(np.float32)
    observations[:, -2] = np.arange(120, -1, -10)/120
    observations[:, -1] = np.arange(300, -1, -25)/300
    terminated = np.zeros((12, 1), dtype=bool)
    terminated[-1] = True
    synthetic.add_episode(observations=observations, actions=np.full((12, 1), -.5),
                          rewards=-np.ones((12, 1)), terminated=terminated,
                          truncated=np.zeros((12, 1), dtype=bool))
    started = time.perf_counter()
    for _ in range(12):
        synthetic.update()
    times['behavior_update_seconds'] = (time.perf_counter()-started)/12
    forecast_env = ClimateVectorEnv()
    started = time.perf_counter()
    forecast_validation(learner, episodes, validation_episodes, forecast_env.task.band,
                        forecast_env.task.grid.w[forecast_env.task.band])
    times['forecast_validation_seconds'] = 8*(time.perf_counter()-started)
    started = time.perf_counter()
    learner.collect_episode(ClimateVectorEnv(), seed=5000)
    times['collection_seconds'] = time.perf_counter()-started
    identity = {'meter': True, 'source': args.source, 'config': digest(config)}
    save_recovery(synthetic, args.out/'recovery.pt', dict(identity=identity, seed_cursor=1))
    expected = synthetic.update()
    restored, cursor = recover(args.out/'recovery.pt', identity)
    observed = restored.update()
    if expected != observed or cursor['seed_cursor'] != 1 or any(
            not torch.equal(a, b) for a, b in zip(synthetic.loss.state_dict().values(), restored.loss.state_dict().values())):
        raise AssertionError('CPU recovery next update differs')
    return dict(**times, generation_condition_seconds=max(train['cost']['wall_seconds'], validation['cost']['wall_seconds']),
                generator_cost_basis='two-core generation occupancy; Study.account also charges startup and full Job allocation',
                allocated_cpu_cores=2, versions=versions(), recovery_next_update_exact=True, behavior_timing_data='synthetic 133-field vectors; no physical actor training')


def final_evaluation(args, config):
    frozen = json.loads(os.environ['MARSWORLD_FROZEN'])
    if (frozen['source_sha256'] != args.source or frozen['config_sha256'] != digest(config)
            or frozen['recipe'].get('paired_response_weight', 0.) != config.get('paired_response_weight', 0.)
            or frozen['seeds'] != list(range(201, 206))):
        raise ValueError('Final evaluation freeze identity mismatch')
    freeze_confirmation(frozen['runs'], frozen['source_sha256'], frozen['config_sha256'],
                        frozen['data_sha256'], frozen['baseline'], frozen['recipe'])
    hashes, _, _ = datasets(args.base, args.rung)
    if hashes != frozen['data_sha256']:
        raise ValueError('Frozen dataset changed')
    learners = []
    for row in frozen['runs']:
        path = args.base/row['checkpoint']
        if file_hash(path) != row['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        identity = {k: row[k] for k in (
            'source_sha256', 'config_sha256', 'data_sha256', 'seed', 'mode', 'variant', 'rung', 'recipe_sha256')}
        if row.get('paired_response_weight'):
            identity['paired_response_weight'] = row['paired_response_weight']
        learner, _ = recover(path, identity)
        learners.append((row['seed'], learner))
    # A failed or interrupted held-out workload must never silently resample.
    with (args.out/'EVALUATION_STARTED.json').open('x') as file:
        json.dump(frozen, file)
    records, initial_states = [], {}
    for seed, learner in learners:
        records.append(dict(seed=seed, rows=evaluate(learner, range(5200, 5220), frozen['baseline'], initial_states,
                         trajectory_dir=args.out/f'trajectories/seed-{seed}')))
        write_json(args.out/f'seed-{seed}.json', records[-1])
    return dict(records=records, frozen_sha256=digest(frozen))


def execute(args, config):
    if args.mode == 'meter':
        return meter(args, config)
    if args.mode == 'data':
        train = generate_dataset(args.base/f'data-{args.rung}'/'train', split='train', train_conditions=args.rung, allocated_cpu_cores=2)
        if args.rung == 12:
            validation = generate_dataset(args.base/'data-12'/'validation', split='validation', allocated_cpu_cores=2)
        else:
            validation, _ = load_dataset(args.base/'data-12'/'validation', split='validation')
        _, training = load_dataset(args.base/f'data-{args.rung}'/'train', split='train')
        baseline = select_baseline(config, training)
        write_json(args.base/f'data-{args.rung}'/'baseline.json', baseline)
        return dict(data_sha256=[train['data_sha256'], validation['data_sha256']], baseline=baseline,
                    generation_cost=[train['cost'], validation['cost']], allocated_cpu_cores=2,
                    cost_basis='generator records 2-core stage occupancy; Study.account charges full Job occupancy including startup')
    if args.mode in ('model', 'actor', 'confirm'):
        return train_stage(args, config)
    if args.mode == 'final':
        return final_evaluation(args, config)
    raise ValueError('Unknown stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('meter', 'data', 'model', 'actor', 'confirm', 'final'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', default='base')
    parser.add_argument('--paired-response-weight', type=float,
                        help='weight for paired response MSE in K² during model pretraining')
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--rung', type=int, choices=(12, 24), default=12)
    parser.add_argument('--source', default=os.environ.get('MARSWORLD_SOURCE_SHA256', 'local-fixture'))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.paired_response_weight is not None:
        if args.mode in ('meter', 'data'):
            parser.error('--paired-response-weight applies to model, actor, confirm and final')
        config['paired_response_weight'] = args.paired_response_weight
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'result.json').exists():
        raise RuntimeError('Completed worker output already exists; controller must reuse its retained record')
    wall, cpu = time.perf_counter(), time.process_time()
    status = dict(mode=args.mode, variant=args.variant, seed=args.seed, source_sha256=args.source, device='cpu')
    try:
        result = execute(args, config)
        status['status'] = 'ok'
    except Exception as error:
        status.update(status='failed', error=repr(error))
        result = {}
        traceback.print_exc()
    status.update(wall_seconds=time.perf_counter()-wall, process_cpu_seconds=time.process_time()-cpu,
                  allocated_cpu_cores=2, versions=versions())
    record = dict(status=status, result=result)
    write_json(args.out/'result.json', record)
    print('RESULT_JSON='+json.dumps(record, allow_nan=False), flush=True)
    if status['status'] != 'ok':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

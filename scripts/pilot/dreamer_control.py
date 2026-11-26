"""Durable, cost-gated Nautilus driver for the relative-warming Dreamer pilot."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
import tarfile
import time

import numpy as np

from scripts.pilot.nautilus_study import ROOT, Study, kubectl, source_bundle
from scripts.pilot.small_study import write_json

CONFIG = ROOT / 'data/specification/pilot/dreamer_control.json'
FINAL_SEEDS = list(range(201, 206))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def recipe_digest(recipe):
    settings = {k: recipe[k] for k in ('learning_rate', 'horizon')}
    if recipe.get('paired_response_weight'):
        settings['paired_response_weight'] = float(recipe['paired_response_weight'])
    return digest(settings)


def freeze_confirmation(records, source, config_hash, data_hashes, baseline, recipe):
    if len(records) != 5 or sorted(r['seed'] for r in records) != FINAL_SEEDS:
        raise ValueError('All five fresh confirmation seeds must finish before test access')
    for row in records:
        if (row['source_sha256'] != source or row['config_sha256'] != config_hash
                or row['data_sha256'] != data_hashes or len(row['checkpoint_sha256']) != 64
                or not math.isfinite(row['validation_rmse_K'])
                or row['mode'] != 'confirm' or row['variant'] != 'selected'
                or row.get('paired_response_weight', 0.) != recipe.get('paired_response_weight', 0.)
                or row['recipe_sha256'] != recipe_digest(recipe)):
            raise ValueError('Confirmation checkpoint identity mismatch')
    return dict(source_sha256=source, config_sha256=config_hash, data_sha256=data_hashes,
                baseline=baseline, recipe=recipe, seeds=FINAL_SEEDS, runs=sorted(records, key=lambda r: r['seed']))


def control_report(records):
    if len(records) != 5 or sorted(r['seed'] for r in records) != FINAL_SEEDS:
        raise ValueError('Report requires all five frozen training seeds')
    conditions = list(range(5200, 5220))
    for record in records:
        if [r['condition_seed'] for r in record['rows']] != conditions:
            raise ValueError('Report requires all twenty matched conditions in order')
    actors = np.array([[r['tracking_rmse_K'] for r in s['rows']] for s in records])
    baselines = np.array([[r['baseline_rmse_K'] for r in s['rows']] for s in records])
    if not np.isfinite(actors).all() or not np.isfinite(baselines).all() or (actors < 0).any():
        raise ValueError('Nonfinite or negative final RMSE')
    if not np.array_equal(baselines, np.broadcast_to(baselines[0], baselines.shape)):
        raise ValueError('Baseline must be matched across every training seed')
    baseline = baselines[0].mean()
    gain = float(1-actors.mean()/baseline) if baseline > 0 else None
    rng, samples = np.random.default_rng(6100), []
    if baseline > 0:
        for _ in range(10000):
            seeds = rng.integers(0, 5, 5)
            groups = rng.integers(0, 20, 20)
            b = baselines[0, groups].mean()
            if b > 0:
                samples.append(1-actors[np.ix_(seeds, groups)].mean()/b)
    violations = sum(r['violations'] for s in records for r in s['rows'])
    return dict(pooled_improvement=gain, baseline_mean_rmse_K=float(baseline),
                per_seed=[dict(seed=s['seed'], mean_rmse_K=float(a.mean()),
                               improvement=float(1-a.mean()/baseline) if baseline > 0 else None)
                          for s, a in zip(records, actors)],
                bootstrap_95=np.quantile(samples, [.025, .975]).tolist() if samples else None,
                bootstrap=dict(draws=10000, seed=6100, paired_condition_groups=True),
                violations=violations, passed=gain is not None and gain >= .10 and violations == 0,
                rows=records)


class DreamerControlStudy(Study):
    def __init__(self, state_dir, study_id, previous_state, config_path=CONFIG):
        super().__init__(Path(state_dir), study_id)
        self.config_path = Path(config_path).resolve()
        config = json.loads(self.config_path.read_text())
        if (config['development_seeds'] != [101, 102] or config['confirmation_seeds'] != FINAL_SEEDS
                or config['horizons'] != [3, 6] or config['final_conditions'] != [5200, 5219]
                or config['cpu_hours'] != 48 or config['gpu_hours'] != 12):
            raise ValueError('This driver requires the declared bounded pilot and sealed five-seed gate')
        previous_state = str(Path(previous_state).resolve())
        if 'control_config' in self.state:
            if self.state['control_config'] != config or self.state['previous_state'] != previous_state:
                raise ValueError('Resolved configuration or previous ledger path changed')
        else:
            self.state.update(control_config=config, config_sha256=digest(config),
                              previous_state=previous_state, phase='waiting-for-previous')
            self.save()
        self.spec = config

    def import_previous(self):
        """Wait outside cluster workers; import the complete cumulative ledger once."""
        raw = Path(self.state['previous_state']).read_bytes()
        previous = json.loads(raw)
        if previous['phase'] != 'complete':
            return False
        previous_hash = hashlib.sha256(raw).hexdigest()
        if 'previous_sha256' in self.state:
            if self.state['previous_sha256'] != previous_hash:
                raise ValueError('Imported cumulative cost ledger changed')
            return True
        jobs = list(previous['jobs'].values())
        if any('charged_seconds' not in j or j.get('accounting_unresolved') for j in jobs):
            raise ValueError('Previous cumulative allocation charges remain unresolved')
        for device in ('cpu', 'cuda'):
            self.state['jobs']['previous-'+device] = dict(device=device, reserved_seconds=0,
                charged_seconds=sum(j['charged_seconds'] for j in jobs if j['device'] == device))
        self.state['jobs']['local-feasibility'] = dict(device='cpu', reserved_seconds=0,
            charged_seconds=self.spec['local_cpu_core_hours']*3600/2,
            allocation_evidence='Conservative local feasibility and metering allowance, CPU core-hours / 2')
        self.state.update(previous_sha256=previous_hash, images=previous['images'], phase='meter')
        (self.state_dir/'previous-state.json').write_bytes(raw)
        self.save()
        return True

    def prepare(self):
        if 'source_sha256' not in self.state:
            original = source_bundle()
            target = io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(original), mode='r:gz') as old, \
                    tarfile.open(fileobj=target, mode='w:gz') as archive:
                for member in old.getmembers():
                    archive.addfile(member, old.extractfile(member))
                for name in ('docs/pilot/HEATING_CONTROL_TASK.md', 'docs/pilot/HEATING_RESPONSE_TASK.md'):
                    archive.add(ROOT/name, arcname=name, recursive=False)
            content = target.getvalue()
            sha = hashlib.sha256(content).hexdigest()
            (self.state_dir/'source.tar.gz').write_bytes(content)
            self.state.update(source_sha256=sha, configmap=f'mars-control-source-{sha[:12]}')
            self.save()
        super().prepare()

    def manifest(self, task):
        mode, variant, seed, rung, device = task
        if device != 'cpu':
            raise ValueError('Climate learner is checked only on CPU')
        name, manifest = super().manifest(task)
        container = manifest['spec']['template']['spec']['containers'][0]
        base = f'/results/{self.study_id}'
        output = f'{base}/{mode}-{variant}-s{seed}-r{rung}'
        env = {item['name']: item['value'] for item in container['env']}
        env.update(MARSWORLD_TORCH_VERSION='2.10.0',
                   MARSWORLD_EXTRA_REQUIREMENTS='k8s/nautilus/requirements-torchrl.txt',
                   MARSWORLD_TASK_DIR=output, MARSWORLD_RUN_DIR=output,
                   TMPDIR='/runtime/tmp', XDG_CACHE_HOME='/runtime/cache',
                   TORCH_HOME='/runtime/cache/torch', PIP_CACHE_DIR='/runtime/cache/pip',
                   MARSWORLD_RECIPE=json.dumps(self.state.get('recipe', {})),
                   MARSWORLD_FROZEN=json.dumps(self.state.get('decision', {})))
        container['env'] = [{'name': k, 'value': v} for k, v in env.items()]
        script = container['args'][0]
        script = script[:script.index('exec python -u -m scripts.pilot.small_study')]
        script = 'mkdir -p /runtime/tmp /runtime/cache "'+output+'"\n'+script
        script += (f'exec python -u -m scripts.pilot.dreamer_control_worker {mode} '
                   f'--config /dev/stdin --base {base} '
                   f"--out {output} --variant {variant} --seed {seed} --rung {rung} <<'PILOT_CONFIG'\n"
                   f'{json.dumps(self.spec, allow_nan=False)}\nPILOT_CONFIG\n')
        container['args'][0] = script
        return name, manifest

    def estimated_seconds(self, task):
        mode, _, _, rung, _ = task
        meter = self.state.get('meter', {})
        generation = meter.get('generation_condition_seconds', 20.)
        model = meter.get('model_update_seconds', .5)
        behavior = meter.get('behavior_update_seconds', 1.)
        collection = meter.get('collection_seconds', 20.)
        model_validation = meter.get('forecast_validation_seconds', 10.)
        if mode == 'meter':
            return 600.
        if mode == 'data':
            # Includes baseline selection: seven schedules per validation condition.
            return generation*(rung + (8 if rung == 12 else 0)) + collection*8*7
        if mode in ('model', 'actor', 'confirm'):
            total = model*self.spec['model_checkpoints'][-1] + model_validation*len(self.spec['model_checkpoints'])
            if mode != 'model':
                count = self.spec['behavior_checkpoints'][-1]
                total += behavior*count + collection*(math.ceil(count/self.spec['collect_every_behavior_updates'])
                                                        + 8*len(self.spec['behavior_checkpoints']))
            return total
        if mode == 'final':
            return 5*20*2*collection
        raise ValueError('Unknown stage for cost admission')

    def admit(self, tasks):
        seconds = 0.
        for task in tasks:
            name, _ = self.manifest(task)
            if name not in self.state['jobs']:
                seconds += self.estimated_seconds(task)*1.5 + max(300., max(
                    (j.get('startup_seconds', 0.) for j in self.state['jobs'].values()), default=0.))
        cpu, _ = self.remaining()
        if seconds > 0 and seconds*2/3600 > cpu:
            raise RuntimeError(f'Stage not admitted: projected {seconds*2/3600:.3f} CPU core-hours exceed remaining {cpu:.3f}')
        if any(task[0] in ('actor', 'confirm', 'final') for task in tasks):
            if not self.state.get('readiness_passed') or 'previous_sha256' not in self.state:
                raise ValueError('Actor training requires completed previous comparison and model readiness')
        if any(task[0] == 'final' for task in tasks) and 'decision' not in self.state:
            raise ValueError('Held-out access requires five frozen checkpoints and baseline')

    def batch(self, tasks):
        """Persist submission intent; recover an interrupted apply without rerunning work."""
        results = []
        admission_error = None
        for start in range(0, len(tasks), 2):
            if admission_error is None:
                try:
                    self.admit(tasks[start:])
                except RuntimeError as error:
                    # Collect existing work before reporting blocked new submissions.
                    admission_error = error
            names = []
            for task in tasks[start:start+2]:
                name, manifest = self.manifest(task)
                job = self.state['jobs'].get(name)
                if job and job.get('record', {}).get('status', {}).get('status') == 'ok':
                    results.append(job['record'])
                    continue
                if job and ('error' in job or 'record' in job):
                    raise RuntimeError(f'Previous task {name} failed; no automatic workload retry')
                if not job:
                    if admission_error is not None:
                        continue
                    active = json.loads(kubectl('get', 'jobs', '-l', 'app=marsworld-small-study', '-o', 'json').stdout)
                    running = sum(not any(c['type'] in ('Complete', 'Failed') and c['status'] == 'True'
                                  for c in j.get('status', {}).get('conditions', [])) for j in active['items'])
                    if running >= 2:
                        raise RuntimeError('Two study workers are active; no additional work admitted')
                    job = dict(created=time.time(), device='cpu', submitted=False,
                               reserved_seconds=self.estimated_seconds(task)*1.5+300.)
                    self.state['jobs'][name] = job
                    write_json(self.state_dir/f'{name}.manifest.json', manifest)
                    self.save()
                if not job.get('submitted'):
                    remote = kubectl('get', 'job', name, '-o', 'json', check=False)
                    if remote.returncode:
                        if not any(text in (remote.stdout+remote.stderr).lower() for text in ('notfound', 'not found')):
                            raise RuntimeError(remote.stderr or remote.stdout)
                        kubectl('apply', '-f', '-', value=manifest)
                    job['submitted'] = True
                    self.save()
                    print(f'START {name}', flush=True)
                names.append(name)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(self.collect, name) for name in names]
                failures = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as error:
                        failures.append(error)
                if failures:
                    raise failures[0]
                if admission_error is None and any(row['result'].get('readiness_failed') for row in results):
                    return results
        if admission_error is not None:
            raise admission_error
        return results

    def stage(self, tasks, phase):
        self.state['phase'] = phase
        self.save()
        return [record['result'] for record in self.batch(tasks)]

    def finish(self, report):
        report.update(source_sha256=self.state['source_sha256'], config_sha256=self.state['config_sha256'],
                      previous_sha256=self.state['previous_sha256'], remaining_cpu_gpu_hours=self.remaining(),
                      local_protocol_deviation=self.spec['local_protocol_deviation'],
                      local_cost_basis=self.spec['local_cost_basis'],
                      costs=self.state['jobs'], candidates=self.state.get('model_candidates', []),
                      behavior_candidates=self.state.get('behavior_candidates', []))
        if report.get('passed'):
            meter = self.state['meter']
            proposal = self.spec['larger_proposal']
            seconds = (96*meter['generation_condition_seconds'] + 5*(
                proposal['model_updates']*meter['model_update_seconds'] +
                proposal['behavior_updates']*meter['behavior_update_seconds'] +
                math.ceil(proposal['behavior_updates']/20)*meter['collection_seconds']))
            report['larger_proposal'] = dict(**proposal, estimated_cpu_core_hours=seconds*2*1.5/3600,
                startup_allowance_core_hours=6*300*2/3600,
                basis='Measured two-core CPU stages, 1.5 multiplier; startup allowance separate. Requires user approval.')
        write_json(self.state_dir/'report.json', report)
        lines = ['# Dreamer response-control pilot', '', report['outcome'], '',
                 'Final condition data are opened only after five checkpoints and the baseline are frozen.',
                 '', f"Source: `{report['source_sha256']}`", f"Configuration: `{report['config_sha256']}`",
                 f"Remaining CPU/GPU admission hours: {report['remaining_cpu_gpu_hours']}"]
        if 'per_seed' in report:
            lines += ['', f"Pooled improvement: {report['pooled_improvement']}",
                      f"Paired hierarchical bootstrap 95% interval: {report['bootstrap_95']}",
                      f"Constraint violations: {report['violations']}", '',
                      '| Training seed | Mean episode RMSE (K) | Improvement |', '|---|---:|---:|']
            lines += [f"| {row['seed']} | {row['mean_rmse_K']} | {row['improvement']} |" for row in report['per_seed']]
        lines += ['', 'All candidate failures, learning curves, data hashes, allocation evidence and complete results are in report.json and state.json.',
                  'This small pilot does not establish climate-control skill beyond its declared simulator conditions.']
        (self.state_dir/'report.md').write_text('\n'.join(lines)+'\n')
        self.state.update(phase='complete', outcome=report['outcome'])
        self.save()

    def run(self):
        if self.state['phase'] == 'complete':
            return
        while not self.import_previous():
            print('Waiting for the existing frozen forecaster comparison to complete; no new jobs started', flush=True)
            time.sleep(30)
        self.prepare()
        self.state['meter'] = self.stage([('meter', 'base', 101, 12, 'cpu')], 'meter')[0]
        self.save()
        rung = self.state.get('rung', 12)
        while True:
            data = self.stage([('data', 'base', 101, rung, 'cpu')], 'data')[0]
            self.state.update(data_sha256=data['data_sha256'], baseline=data['baseline'], rung=rung)
            self.save()
            tasks = [('model', variant, seed, rung, 'cpu') for variant in self.spec['model_learning_rates']
                     for seed in self.spec['development_seeds']]
            models = self.stage(tasks, 'world-model-validation')
            self.state.setdefault('model_candidates', {})[str(rung)] = models
            self.save()
            groups = {v: [r for r in models if r['variant'] == v] for v in self.spec['model_learning_rates']}
            eligible = {v: rows for v, rows in groups.items() if len(rows) == 2 and all(r['ready'] for r in rows)}
            rank = lambda rows: np.mean([r['validation']['model_response']['rmse'] for r in rows])
            if eligible:
                selected = min(eligible, key=lambda v: rank(eligible[v]))
                self.state.update(readiness_passed=True, recipe=dict(
                    learning_rate=self.spec['model_learning_rates'][selected], model_variant=selected, rung=rung))
                if self.spec.get('paired_response_weight'):
                    self.state['recipe']['paired_response_weight'] = self.spec['paired_response_weight']
                self.save()
                break
            best = min(groups, key=lambda v: rank(groups[v]))
            improving = all(next(c['validation']['model_response']['rmse'] for c in r['curves'] if c['updates'] == 2000)
                            <= .9*next(c['validation']['model_response']['rmse'] for c in r['curves'] if c['updates'] == 1000)
                            for r in groups[best])
            if rung == 12 and improving:
                # Admit the complete extra rung before spending on its data.
                self.admit([('data', 'base', 101, 24, 'cpu')] +
                           [('model', v, s, 24, 'cpu') for v in groups for s in (101, 102)])
                rung = 24
                continue
            self.finish(dict(passed=False, outcome='World-model readiness failed. No actor or final-test jobs were admitted.',
                             final_test_opened=False, larger_proposal=None))
            return
        actors = self.stage([('actor', f'h{h}', seed, rung, 'cpu')
                             for h in (3, 6) for seed in (101, 102)], 'behavior-validation')
        self.state['behavior_candidates'] = actors
        if any(row.get('readiness_failed') for row in actors):
            self.finish(dict(passed=False, final_test_opened=False,
                             outcome='A development initialization failed its own model readiness. No resampling or confirmation admitted.'))
            return
        horizon = min((3, 6), key=lambda h: np.mean([r['validation_rmse_K'] for r in actors if r['variant'] == f'h{h}']))
        self.state['recipe']['horizon'] = horizon
        self.save()
        confirmed = self.stage([('confirm', 'selected', seed, rung, 'cpu') for seed in FINAL_SEEDS], 'confirmation')
        self.state['confirmation_candidates'] = confirmed
        if any(row.get('readiness_failed') for row in confirmed):
            self.finish(dict(passed=False, final_test_opened=False,
                             outcome='A fresh confirmation model failed readiness. No held-out access or resampling admitted.'))
            return
        decision = freeze_confirmation(confirmed, self.state['source_sha256'], self.state['config_sha256'],
                                       self.state['data_sha256'], self.state['baseline'], self.state['recipe'])
        if 'decision' in self.state and self.state['decision'] != decision:
            raise ValueError('Frozen decision changed')
        self.state['decision'] = decision
        write_json(self.state_dir/'FROZEN.json', decision)
        self.save()
        final = self.stage([('final', 'frozen', 0, rung, 'cpu')], 'final-once')[0]
        if final['frozen_sha256'] != digest(decision):
            raise ValueError('Final result does not match the frozen decision')
        report = control_report(final['records'])
        report.update(final_test_opened=True, outcome=('Pooled five-seed gate passed.' if report['passed'] else
                      'Pooled five-seed gate failed. This is a valid negative result; no retuning is admitted.'))
        self.finish(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--previous-state', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--study-id', default='mars-dreamer-ctrl-v1')
    args = parser.parse_args()
    if (not args.study_id.replace('-', '').isalnum() or not args.study_id.islower() or len(args.study_id) > 22):
        parser.error('Study ID must be at most 22 lowercase letters, digits, and hyphens')
    if args.config.resolve() != CONFIG.resolve():
        parser.error('Production workers require the checked-in frozen configuration path')
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir/'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = DreamerControlStudy(args.state_dir, args.study_id, args.previous_state, args.config)
        try:
            study.run()
        except Exception as error:
            study.state['blocker'] = repr(error)
            study.save()
            (args.state_dir/'blocker.md').write_text(f'Pilot stopped admission: {error}\nExisting Jobs and artifacts are retained.\n')
            raise


if __name__ == '__main__':
    main()

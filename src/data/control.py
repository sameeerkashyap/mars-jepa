"""Versioned response-control episodes. Final test conditions remain sealed.

Generate only after cost admission; this module never fits a model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from importlib.metadata import version

import numpy as np

from benchmarks.dreamer_control.climate import ClimateVectorEnv

NAMESPACE = "control-response-v1"
TASK = "south-winter-response-v1"
SPLITS = {"train": range(5000, 5024), "validation": range(5100, 5108)}
SCHEDULES = ("zero", "constant_2p5", "random_pulse", "random_feasible")
POLICY_KEYS = ("observations", "actions", "requested_actions", "rewards", "terminated", "truncated")
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("src/physics/reference.py", "src/physics/control.py", "src/physics/surface.py",
                "benchmarks/dreamer_control/climate.py", "src/data/control.py")
TASK_FILES = ("docs/pilot/HEATING_CONTROL_TASK.md", "docs/pilot/HEATING_RESPONSE_TASK.md")


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _split(split, seeds):
    if split not in SPLITS:
        raise ValueError("only train/validation splits are open; final test is sealed")
    if not seeds or len(seeds) != len(set(seeds)) or any(
        type(seed) is not int or seed not in SPLITS[split] for seed in seeds
    ):
        raise ValueError("condition seeds must be unique and belong to the declared split")


def _schedule(seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
    if index == 0:
        return np.zeros(12)
    if index == 1:
        return np.full(12, 2.5)
    if index == 2:
        actions = np.zeros(12)
        actions[rng.choice(12, 3, replace=False)] = 10.
        return actions
    actions, remaining = [], 300.
    for _ in range(12):
        heat = rng.uniform(0, min(10, remaining / 10))
        actions.append(heat)
        remaining = max(0., remaining - 10 * heat)
    return np.array(actions)


def generate_dataset(destination, *, split, condition_seeds=None, train_conditions=12, allocated_cpu_cores=1):
    """Write four matched schedules per condition, refusing any existing directory.

    Default training groups are nested 12/24; explicit subsets permit cost checks.
    All schedules in one condition share the same initial paired truth checkpoint.
    """
    if type(allocated_cpu_cores) is not int or allocated_cpu_cores < 1:
        raise ValueError("allocated_cpu_cores must be a positive integer")
    if train_conditions not in (12, 24):
        raise ValueError("train_conditions must be 12 or 24")
    if split not in SPLITS:
        raise ValueError("only train/validation splits are open; final test is sealed")
    seeds = list(condition_seeds) if condition_seeds is not None else list(SPLITS[split])
    if condition_seeds is None and split == "train":
        seeds = seeds[:train_conditions]
    _split(split, seeds)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    wall, cpu = time.perf_counter(), time.process_time()
    env = ClimateVectorEnv()
    manifest = dict(format_version=1, data_namespace=NAMESPACE, task=TASK, split=split,
                    condition_seeds=seeds, schedules=list(SCHEDULES),
                    proposed_training_groups=[list(range(5000, 5012)), list(range(5000, 5024))],
                    sealed_test_conditions=[5200, 5219], reserved_conditions=[9000, 9999],
                    source_sha256={p: _hash_file(ROOT / p) for p in SOURCE_FILES},
                    task_sha256={p: _hash_file(ROOT / p) for p in TASK_FILES},
                    config=env.task.cfg, config_sha256=_hash_json(env.task.cfg),
                    versions=dict(python=sys.version, numpy=version("numpy"), scipy=version("scipy")),
                    schedule_rng="numpy SeedSequence([condition_seed, schedule_index, 1])",
                    episodes=[])
    for seed in seeds:
        env.reset(seed=seed)
        initial = env.state_dict()
        for index, schedule in enumerate(SCHEDULES):
            episode_wall, episode_cpu = time.perf_counter(), time.process_time()
            env.load_state_dict(initial)
            observations = [env.observation()]
            rows = {key: [] for key in (*POLICY_KEYS[1:], "requested_heat", "applied_heat",
                                        "daily_T_s", "daily_S", "daily_t_sol")}
            for heat in _schedule(seed, index):
                raw = np.array([heat / 5 - 1], dtype=np.float32)
                observation, reward, terminated, truncated, info = env.step(raw)
                observations.append(observation)
                for key, value in dict(actions=info["dynamics_action"].numpy(), requested_actions=raw,
                                       rewards=[reward], terminated=[terminated], truncated=[truncated],
                                       requested_heat=[info["requested_heat"]], applied_heat=[info["applied_heat"]],
                                       daily_T_s=info["daily_T_s"], daily_S=info["daily_S"],
                                       daily_t_sol=info["daily_t_sol"]).items():
                    rows[key].append(value)
            arrays = {key: np.asarray(value) for key, value in rows.items()}
            arrays["observations"] = np.stack(observations)
            if index == 0:
                no_heat_daily = arrays["daily_T_s"].copy()
            arrays["oracle_daily_response_T_s"] = arrays["daily_T_s"] - no_heat_daily
            for key in ("actions", "requested_actions", "rewards"):
                arrays[key] = arrays[key].astype(np.float32)
            filename = f"{seed}-{schedule}.npz"
            with (destination / filename).open("xb") as file:
                np.savez_compressed(file, **arrays)
            manifest["episodes"].append(dict(condition_seed=seed, schedule=schedule, schedule_index=index,
                                               file=filename, sha256=_hash_file(destination / filename),
                                               oracle_diagnostics=env.metrics(),
                                               wall_seconds=time.perf_counter()-episode_wall,
                                               cpu_seconds=time.process_time()-episode_cpu))
    manifest["cost"] = dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu,
                             allocated_cpu_cores=allocated_cpu_cores, gpu_hours=0,
                             spinup_included=True, matched_schedule_spinup_reused=True)
    manifest["cost"]["allocated_cpu_core_hours"] = allocated_cpu_cores * manifest["cost"]["wall_seconds"] / 3600
    manifest["cost"]["process_cpu_hours"] = manifest["cost"]["cpu_seconds"] / 3600
    manifest["data_sha256"] = _hash_json(manifest)
    with (destination / "manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2, allow_nan=False)
        file.write("\n")
    return manifest


def _validate_episode(data, record, env):
    shapes = dict(observations=(13, 133), actions=(12, 1), requested_actions=(12, 1),
                  rewards=(12, 1), terminated=(12, 1), truncated=(12, 1),
                  requested_heat=(12, 1), applied_heat=(12, 1), daily_T_s=(12, 10, 64),
                  daily_S=(12, 10, 64), daily_t_sol=(12, 10),
                  oracle_daily_response_T_s=(12, 10, 64))
    if set(data) != set(shapes):
        raise ValueError("episode schema has missing or private fields")
    for key, shape in shapes.items():
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid episode schema: {key}")
        dtype = np.bool_ if key in ("terminated", "truncated") else np.floating
        if not np.issubdtype(data[key].dtype, dtype):
            raise ValueError(f"invalid episode dtype: {key}")
    expected_done = np.zeros((12, 1), dtype=bool)
    expected_done[-1] = True
    if not np.array_equal(data["terminated"], expected_done) or data["truncated"].any():
        raise ValueError("episode must end exactly once at decision twelve")
    obs, heat = data["observations"], data["applied_heat"]
    checks = [np.allclose(obs[:, -2], np.arange(120, -1, -10) / 120),
              np.allclose(obs[:, -1], np.r_[300., 300.-np.cumsum(10*heat)] / 300, atol=1e-6),
              (heat >= 0).all(), (heat <= 10).all(), heat.sum() * 10 <= 300 + 1e-5,
              np.allclose(data["actions"], heat / 5 - 1),
              np.allclose(obs[1:, 128], heat[:, 0] / 10), obs[0, 128] == 0,
              np.allclose(data["requested_heat"], 5*(np.clip(data["requested_actions"].astype(np.float64), -1, 1)+1)),
              (data["rewards"] <= 0).all(),
              np.allclose(obs[1:, :64], (data["daily_T_s"][:, -1] - env.frost) / 10, atol=1e-6),
              np.allclose(obs[1:, 64:128], data["daily_S"][:, -1] / 589, atol=1e-6),
              np.allclose(np.diff(data["daily_t_sol"].reshape(-1)), 1.),
              np.isclose(-data["rewards"].sum() / 120,
                         record["oracle_diagnostics"]["tracking_rmse_K"]**2, rtol=1e-5)]
    response_error = data["oracle_daily_response_T_s"][:, :, env.task.band] - 2
    interval_error = np.average(response_error**2, axis=-1,
                                weights=env.task.grid.w[env.task.band]).sum(-1)
    checks.append(np.allclose(data["rewards"][:, 0], -interval_error, rtol=1e-5))
    remaining = 300.
    for requested, applied in zip(data["requested_heat"].flat, heat.flat):
        checks.append(np.isclose(applied, min(requested, remaining / 10)))
        remaining = max(0., remaining - 10*applied)
    if not all(checks):
        raise ValueError("episode violates transition alignment or heat/time contract")


def load_dataset(destination, *, split):
    """Verify provenance, hashes, grouped split identity and schema before use.

    Return (manifest, episodes). Only episode['policy'] is passed to add_episode;
    daily observations and private oracle metrics stay in separate namespaces.
    """
    if split not in SPLITS:
        raise ValueError("only train/validation splits are open; final test is sealed")
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    digest = manifest.pop("data_sha256", None)
    if digest != _hash_json(manifest):
        raise ValueError("dataset manifest hash mismatch")
    manifest["data_sha256"] = digest
    if (manifest["format_version"] != 1 or manifest["data_namespace"] != NAMESPACE
            or manifest["task"] != TASK or manifest["split"] != split):
        raise ValueError("unsupported dataset version/task/split identity")
    _split(split, manifest["condition_seeds"])
    for key, paths in (("source_sha256", SOURCE_FILES), ("task_sha256", TASK_FILES)):
        if manifest[key] != {p: _hash_file(ROOT / p) for p in paths}:
            raise ValueError("source/task hash mismatch")
    env = ClimateVectorEnv()
    if manifest["config_sha256"] != _hash_json(manifest["config"]) or manifest["config"] != env.task.cfg:
        raise ValueError("configuration hash mismatch")
    expected = {(seed, schedule) for seed in manifest["condition_seeds"] for schedule in SCHEDULES}
    records = manifest["episodes"]
    if len(records) != len(expected) or {(r["condition_seed"], r["schedule"]) for r in records} != expected:
        raise ValueError("missing, repeated, or cross-split condition/schedule")
    episodes = []
    for record in records:
        filename = f"{record['condition_seed']}-{record['schedule']}.npz"
        if record["file"] != filename or record["schedule_index"] != SCHEDULES.index(record["schedule"]):
            raise ValueError("episode filename/seed stream identity mismatch")
        path = destination / filename
        if _hash_file(path) != record["sha256"]:
            raise ValueError("episode file hash mismatch")
        with np.load(path, allow_pickle=False) as file:
            data = {key: file[key].copy() for key in file.files}
        _validate_episode(data, record, env)
        episodes.append(dict(policy={key: data[key] for key in POLICY_KEYS},
                             daily={key: data[f"daily_{key}"] for key in ("T_s", "S", "t_sol")},
                             oracle_diagnostics={**record["oracle_diagnostics"],
                                                 "daily_response_T_s": data["oracle_daily_response_T_s"]}))
    return manifest, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(SPLITS), required=True)
    parser.add_argument("--train-conditions", type=int, choices=(12, 24), default=12)
    args = parser.parse_args()
    manifest = generate_dataset(args.out, split=args.split, train_conditions=args.train_conditions)
    print(json.dumps(manifest["cost"], indent=2))


if __name__ == "__main__":
    main()

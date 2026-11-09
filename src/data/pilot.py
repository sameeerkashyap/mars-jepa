"""
THE data API for the Checkpoint-B datasets (single module; `dio.py` was merged
into this file and moved to legacy/).

Public API
----------
Observables (model input; safe for train.py / models/):
    load_trajectory(path)                                  -> dict (one file)
    load_split(data_root, family, split, *, calibration_mode=False)
                                                           -> [dict, ...]
    subsample_train(trajs, fraction, seed)                 -> [dict, ...]
    stratified_fraction_subsets(entries, fractions, *, seed, n_strata)
                                                           -> {fraction: [ids]}
Evaluation labels (control_T_s, response_T_s; eval.py ONLY):
    load_eval_labels(path, *, calibration_mode=False)      -> dict (one file)
    load_split_eval_labels(data_root, family, split, *, calibration_mode=False)
                                                           -> [dict, ...] aligned
                                                              with load_split
Hidden diagnostics (never model input):
    load_hidden(path)                                      -> dict

Guarantees:
  * The DEFAULT loader returns ONLY observable inputs (T_s, S, F_a, grid/time
    coords). It never returns hidden diagnostics ('hidden__') nor evaluation
    labels ('eval__', i.e. control_T_s / response_T_s).
  * Evaluation labels are reachable only via `load_eval_labels` /
    `load_split_eval_labels`, which RAISE if called with `calibration_mode=True`.
  * Hidden diagnostics are reachable only via `load_hidden`.
  * CALIBRATION code paths (calibration_mode=True) cannot load Tier-1 / Tier-2
    test data.
  * Every load of a test split (observables OR eval labels) is appended, with a
    UTC timestamp, to <data_root>/test_access.log.
  * A stratified-by-magnitude data-fraction subsampler produces NESTED,
    deterministic subsets.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import numpy as np

HIDDEN = "hidden__"
EVAL = "eval__"
TEST_SPLITS = {"test_tier1", "test_tier2"}


def _manifest(data_root):
    with open(os.path.join(data_root, "manifest.json")) as fh:
        return json.load(fh)


# ----------------------------------------------------------------- loaders
def load_trajectory(path):
    """Load one trajectory's OBSERVABLE inputs only.

    Hidden ('hidden__') and eval ('eval__') arrays are never returned.
    """
    npz = np.load(path)
    obs = {k: npz[k] for k in npz.files
           if not k.startswith(HIDDEN) and not k.startswith(EVAL)}
    # hard guarantees against leaks:
    assert not any(k.startswith((HIDDEN, EVAL)) for k in obs), "leak!"
    for forbidden in ("m", "T_g", "tau", "control_T_s", "response_T_s"):
        obs.pop(forbidden, None)            # legacy un-prefixed files
    return obs


def load_eval_labels(path, *, calibration_mode=False):
    """Load evaluation labels (control_T_s, response_T_s) for a trajectory.

    RAISES if `calibration_mode=True` -- calibration code paths must not touch
    evaluation labels.
    """
    if calibration_mode:
        raise RuntimeError("load_eval_labels called in calibration_mode -- "
                           "evaluation labels are forbidden on calibration paths.")
    npz = np.load(path)
    return {k[len(EVAL):]: npz[k] for k in npz.files if k.startswith(EVAL)}


def load_hidden(path):
    """Explicitly load the HIDDEN diagnostics group (NOT for model input)."""
    npz = np.load(path)
    return {k[len(HIDDEN):]: npz[k] for k in npz.files if k.startswith(HIDDEN)}


# ----------------------------------------------------------------- splits
def _log_test_access(data_root, family, split):
    line = (f"{datetime.now(timezone.utc).isoformat()}\tTEST-LOAD\t"
            f"{family}/{split}\n")
    with open(os.path.join(data_root, "test_access.log"), "a") as fh:
        fh.write(line)


def split_entries(data_root, family, split):
    """Manifest trajectory entries (id, seed, magnitude, file, ...) for a split."""
    man = _manifest(data_root)
    key = f"{family}/{split}"
    if key not in man["splits"]:
        raise KeyError(f"no such split: {key}")
    return [t for t in man["trajectories"]
            if t["family"] == family and t["split"] == split]


def list_split_files(data_root, family, split):
    return [os.path.join(data_root, e["file"])
            for e in split_entries(data_root, family, split)]


def load_split(data_root, family, split, *, calibration_mode=False):
    """Return a list of observable trajectory dicts for a split.

    Guards: calibration_mode + test split -> RuntimeError; calibration family
    has no test splits; any test-split load is timestamp-logged.
    """
    if split in TEST_SPLITS:
        if calibration_mode:
            raise RuntimeError(
                "calibration code path attempted to load a test split "
                f"('{family}/{split}') -- forbidden.")
        if family == "calibration":
            raise KeyError("calibration family has no test splits.")
        _log_test_access(data_root, family, split)
    return [load_trajectory(f) for f in list_split_files(data_root, family, split)]


def load_split_eval_labels(data_root, family, split, *, calibration_mode=False):
    """Eval labels for every trajectory of a split, aligned with load_split.

    RAISES in calibration_mode (before touching any file); test-split access is
    timestamp-logged like load_split."""
    if calibration_mode:
        raise RuntimeError("load_split_eval_labels called in calibration_mode -- "
                           "evaluation labels are forbidden on calibration paths.")
    files = list_split_files(data_root, family, split)
    if split in TEST_SPLITS:
        _log_test_access(data_root, family, split)
    return [load_eval_labels(f) for f in files]


# ----------------------------------------------------- data-fraction subsampler
def stratified_fraction_subsets(entries, fractions=(0.1, 0.3, 1.0), *,
                                seed=0, n_strata=5):
    """Stratified-by-magnitude, NESTED, deterministic data-fraction subsets.

    Parameters
    ----------
    entries : list of dicts each having 'id' and 'magnitude'
    fractions : iterable of fractions in (0, 1]
    seed : int, controls the deterministic within-stratum ordering
    n_strata : number of magnitude strata

    Returns
    -------
    dict {fraction: [ids]} where, for f1 < f2, subset(f1) is a PREFIX-nested
    subset of subset(f2). The ordering interleaves strata (round-robin) so that
    every prefix is balanced across magnitude.
    """
    entries = list(entries)
    n = len(entries)
    if n == 0:
        return {float(f): [] for f in fractions}

    mags = np.array([e["magnitude"] for e in entries], dtype=float)
    # assign strata by magnitude rank (equal-count strata, stable)
    order_by_mag = np.argsort(mags, kind="stable")
    stratum = np.empty(n, dtype=int)
    k = max(1, min(n_strata, n))
    for pos, idx in enumerate(order_by_mag):
        stratum[idx] = min(k - 1, pos * k // n)

    rng = np.random.default_rng(seed)
    prio = rng.permutation(n)                      # deterministic tie-break
    buckets = [[] for _ in range(k)]
    for i in range(n):
        buckets[stratum[i]].append(i)
    for b in buckets:
        b.sort(key=lambda i: prio[i])

    # round-robin interleave -> global order; prefixes are stratified & nested
    global_order = []
    j = 0
    while len(global_order) < n:
        for b in buckets:
            if j < len(b):
                global_order.append(b[j])
        j += 1

    ids = [entries[i]["id"] for i in global_order]
    out = {}
    for f in fractions:
        count = int(math.ceil(float(f) * n))
        count = max(1, min(n, count))
        out[float(f)] = ids[:count]
    return out


def subsample_train(trajs, fraction, seed):
    """Stratified-by-forcing-magnitude, NESTED, DETERMINISTIC subsample of a
    list of loaded trajectories (magnitude = max |F_a|).  Uses the same ordering
    as `stratified_fraction_subsets`; returns trajectories in original order."""
    entries = [{"id": i, "magnitude": float(np.max(np.abs(np.asarray(t["F_a"]))))}
               for i, t in enumerate(trajs)]
    ids = stratified_fraction_subsets(entries, (fraction,), seed=seed)[float(fraction)]
    return [trajs[i] for i in sorted(ids)]

"""Structure-only check of diag_linearity.py (does NOT run the simulator)."""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_linearity as dl
import gen_data
import metric_defs


def test_defaults_and_seeds_outside_dataset_ranges():
    assert dl.TRAIN_MAGS == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert dl.TEST_MAGS == [7.5, 8.0, 9.0, 10.0]
    assert dl.DIAG_SEEDS == [7000, 7001, 7002, 7003, 7004]
    assert dl.REC_YEARS == 3
    ranges = list(gen_data.SEED_RANGES.values())
    assert not any(lo <= s <= hi for s in dl.DIAG_SEEDS for lo, hi in ranges)


def test_structure_spawn_safe_and_frozen_asserts():
    assert dl._seed_job.__module__ == "diag_linearity"
    assert dl._seed_job.__qualname__ == "_seed_job"          # top-level
    src = inspect.getsource(dl)
    assert "ProcessPoolExecutor" in src and 'get_context("spawn")' in src
    assert 'cfg["time"]["dt_sol"] == 0.05' in src and 'cfg["physics"]["D"] == 0.10' in src
    assert 'cfg["grid"]["n"] == 64' in src
    assert dl.polar_bands is metric_defs.polar_bands
    assert dl.winter_steps is metric_defs.winter_steps
    for flag in ("--workers", "--seeds", "--out"):
        assert flag in src


def _fake_seed(seed, slope, curve=0.0, pinned_missing=False):
    resp = {}
    for A in dl.TRAIN_MAGS + dl.TEST_MAGS:
        r = slope * A + curve * A ** 2
        resp[str(A)] = {h: {"band": r, "pinned": None if pinned_missing else 2 * r,
                            "rest": r / 2} for h in ("N", "S")}
    return {"seed": seed, "response_K": resp}


def test_fit_and_summary_exact_line():
    per = [_fake_seed(7000, 0.8), _fake_seed(7001, 1.2)]
    tab = dl.summarize(per, dl.TRAIN_MAGS, dl.TEST_MAGS)
    for h in ("N", "S"):
        for g in dl.GROUPS:
            row = tab[h][g]
            assert row["n_seeds"] == 2 and row["seeds_missing"] == []
            for A in dl.TEST_MAGS:
                v = row["by_A"][str(A)]
                assert np.isclose(v["err_K"], 0.0) and np.isclose(v["rel_err"], 0.0)


def test_curvature_gives_underprediction_and_missing_seed_reported():
    per = [_fake_seed(7000, 1.0, curve=0.05),
           _fake_seed(7001, 1.0, curve=0.05, pinned_missing=True)]
    tab = dl.summarize(per, dl.TRAIN_MAGS, dl.TEST_MAGS)
    assert tab["N"]["band"]["by_A"]["10.0"]["err_K"] < 0
    pin = tab["S"]["pinned"]
    assert pin["n_seeds"] == 1 and pin["n_seeds_total"] == 2
    assert pin["seeds_missing"] == [7001]
    dl.print_table(tab, dl.TEST_MAGS)                         # no crash

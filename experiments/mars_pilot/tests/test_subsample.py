"""Tests for the stratified data-fraction subsampler: nested + deterministic."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataio import stratified_fraction_subsets


def _entries(n, seed=0):
    rng = np.random.default_rng(seed)
    mags = rng.uniform(0, 5, size=n)
    return [{"id": f"t{i:03d}", "magnitude": float(mags[i])} for i in range(n)]


def test_nested_and_counts():
    entries = _entries(100)
    subs = stratified_fraction_subsets(entries, (0.1, 0.3, 1.0), seed=7)
    s10, s30, s100 = subs[0.1], subs[0.3], subs[1.0]
    assert len(s10) == 10 and len(s30) == 30 and len(s100) == 100
    # nested: 10% subset of 30% subset of 100%
    assert set(s10) <= set(s30) <= set(s100)
    # prefix-nested (ordering preserved)
    assert s30[:10] == s10
    assert s100[:30] == s30


def test_deterministic():
    entries = _entries(60)
    a = stratified_fraction_subsets(entries, (0.1, 0.3, 1.0), seed=3)
    b = stratified_fraction_subsets(entries, (0.1, 0.3, 1.0), seed=3)
    assert a == b
    c = stratified_fraction_subsets(entries, (0.1, 0.3, 1.0), seed=4)
    # different seed generally reorders (not asserting inequality of full set at
    # 100%, but the 10% membership should differ for this data)
    assert a[0.1] != c[0.1]


def test_stratified_balance():
    # the 10% subset should span low and high magnitudes, not cluster
    entries = _entries(100, seed=1)
    subs = stratified_fraction_subsets(entries, (0.1, 1.0), seed=2, n_strata=5)
    mags = {e["id"]: e["magnitude"] for e in entries}
    chosen = np.array([mags[i] for i in subs[0.1]])
    assert chosen.min() < 1.5 and chosen.max() > 3.5

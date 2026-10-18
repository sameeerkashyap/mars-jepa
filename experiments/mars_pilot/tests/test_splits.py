"""Seed ranges are disjoint across all splits and families, and within the
declared per-family ranges; the held-out range is never used."""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_data import SPLITS, SEED_RANGES


def _all_seeds():
    out = []
    for family, fam_splits in SPLITS.items():
        for split, spec in fam_splits.items():
            for s in spec["seeds"]:
                out.append((family, split, s))
    return out


def test_no_duplicate_seeds_across_splits():
    seeds = [s for _, _, s in _all_seeds()]
    dupes = [s for s, c in Counter(seeds).items() if c > 1]
    assert not dupes, f"duplicate seeds across splits: {dupes}"


def test_seeds_within_declared_family_ranges():
    for family, split, s in _all_seeds():
        lo, hi = SEED_RANGES[family]
        assert lo <= s <= hi, f"{family}/{split} seed {s} outside {lo}-{hi}"


def test_family_ranges_disjoint():
    ranges = list(SEED_RANGES.values())
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            (a0, a1), (b0, b1) = ranges[i], ranges[j]
            assert a1 < b0 or b1 < a0, f"overlapping families: {ranges[i]} {ranges[j]}"


def test_held_out_range_unused():
    lo, hi = SEED_RANGES["held_out_reserved"]
    for _, _, s in _all_seeds():
        assert not (lo <= s <= hi), f"seed {s} intrudes on held-out range"


def test_tier2_constant_only_and_out_of_range():
    import gen_data
    spec = SPLITS["main"]["test_tier2"]
    assert spec["shapes"] == ["constant"]
    assert spec["mag"] == gen_data.TIER2_MAG
    assert min(spec["mag"]) > SPLITS["main"]["train"]["mag"][1]
    tasks = gen_data.build_tasks(None, "/nonexistent", 1,
                                 splits={"main": {"test_tier2": spec}})
    assert {t["shape"] for t in tasks} == {"constant"}
    lo, hi = spec["mag"]
    assert all(lo <= t["magnitude"] <= hi for t in tasks)

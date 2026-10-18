"""Structure-only check of diag_tier2.py on a tiny run (short spin/record);
the real diagnostic (record_years=3) is run by the reviewer."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diag_tier2
import gen_data


def test_diag_tier2_defaults():
    assert diag_tier2.MAGS == [7.5, 8.0, 9.0, 10.0]
    assert diag_tier2.DIAG_SEEDS == [7000, 7001, 7002, 7003, 7004]
    assert diag_tier2.REC_YEARS == 3
    ranges = list(gen_data.SEED_RANGES.values())
    assert not any(lo <= s <= hi for s in diag_tier2.DIAG_SEEDS for lo, hi in ranges)


def test_diag_tier2_tiny(monkeypatch):
    monkeypatch.setattr(gen_data, "SPIN_YEARS", 0.05)
    per = [diag_tier2._seed_job((s, [8.0], 1.0, 0.1, 0.5)) for s in (7000, 7001)]
    for p in per:
        assert set(p["by_A"]["8.0"]) == {"N", "S"}
    tab = diag_tier2.summarize(per, [8.0])
    assert set(tab["8.0"]) == {"N", "S"}
    for v in tab["8.0"].values():
        assert v is None or {"signal_K", "seed_std_K", "SN", "n_seeds"} <= set(v)

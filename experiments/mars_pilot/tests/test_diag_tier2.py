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


def test_diag_tier2_counts_seeds_without_pinned_cells():
    """frac_unpinned must not silently drop seeds with no control-pinned cell."""
    def p(fr):
        r = {"winter_mean_response_K": 1.0, "winter_peak_response_K": 2.0,
             "frac_ctrl_pinned_unpinned_forced": fr}
        return {"by_A": {"8.0": {"N": r, "S": r}}}
    tab = diag_tier2.summarize([p(1.0), p(0.5), p(float("nan"))], [8.0])
    v = tab["8.0"]["N"]
    assert v["n_seeds"] == 3 and v["frac_unpinned_n_seeds"] == 2
    assert v["n_seeds_no_ctrl_pinned"] == 1
    assert v["frac_unpinned"] == 0.75 and v["frac_unpinned_min"] == 0.5


def test_diag_tier2_uses_metric_defs():
    import metric_defs
    assert diag_tier2.polar_bands is metric_defs.polar_bands
    assert diag_tier2.winter_steps is metric_defs.winter_steps

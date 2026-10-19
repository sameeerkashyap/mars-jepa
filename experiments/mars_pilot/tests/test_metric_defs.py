"""metric_defs: observable-only band and polar-winter masks."""

import inspect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metric_defs as md


def _lat():
    return np.radians(np.array([-80.0, -65.0, -60.0, 0.0, 60.0, 61.0, 85.0]))


def test_constants():
    assert md.BAND_DEG == 60.0 and md.WINTER_S_FRAC == 0.10


def test_polar_bands_strict():
    b = md.polar_bands(_lat())
    assert b["N"].tolist() == [False, False, False, False, False, True, True]
    assert b["S"].tolist() == [True, True, False, False, False, False, False]


def test_seasonal_ref_and_winter_steps():
    t = np.arange(100)
    S = np.zeros((100, 7))
    S[:, 5:] = (100.0 * (1 + np.sin(2 * np.pi * t / 100)))[:, None]   # 0..200
    band = md.polar_bands(_lat())["N"]
    assert md.seasonal_ref(S, band) == np.max(S[:, 5])
    w = md.winter_steps(S, band)
    assert np.array_equal(w, S[:, 5] <= 0.1 * S[:, 5].max())
    assert w.any() and not w.all()


def test_window_uses_full_trajectory_reference():
    """A low-S window has its own small max; s_ref from the full trajectory
    must be used to avoid calling the whole window 'winter' or 'summer'."""
    S = np.zeros((10, 7))
    S[:, 5:] = np.array([200, 150, 100, 50, 15, 10, 5, 10, 15, 50.0])[:, None]
    band = md.polar_bands(_lat())["N"]
    win = S[4:9]                                   # 15, 10, 5, 10, 15
    ref = md.seasonal_ref(S, band)
    assert md.winter_steps(win, band, s_ref=ref).all()             # all <= 20
    assert not md.winter_steps(win, band).all()                    # self-ref


def test_observable_only_signature():
    for f in (md.polar_bands, md.seasonal_ref, md.winter_steps):
        params = set(inspect.signature(f).parameters)
        assert not params & {"T_s", "T", "m", "T_g", "tau", "pred", "control_T_s"}

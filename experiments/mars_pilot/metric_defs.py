"""
metric_defs.py -- single source of truth for the polar-winter metric masks.

OBSERVABLE-ONLY: every mask here depends on insolation S and latitude only,
never on model output, T_s, or hidden state (m, T_g, tau).  Used by eval.py,
diag_tier2.py and diag_linearity.py so all three apply the same definition
(PREREG §6: band |lat| > 60 deg, per hemisphere; polar winter = steps where the
band-mean insolation is <= WINTER_S_FRAC of its seasonal max).
"""

from __future__ import annotations

import numpy as np

BAND_DEG = 60.0
WINTER_S_FRAC = 0.10


def polar_bands(lat_rad):
    """{"N": lat > 60 deg, "S": lat < -60 deg} boolean (ncell,) masks."""
    latd = np.degrees(np.asarray(lat_rad, dtype=float))
    return {"N": latd > BAND_DEG, "S": latd < -BAND_DEG}


def _band_mean_S(S, band):
    S = np.asarray(S, dtype=float)
    band = np.asarray(band, dtype=bool)
    if not band.any():
        raise ValueError("empty latitude band")
    return S[:, band].mean(axis=1)


def seasonal_ref(S_full, band):
    """Max over time of the band-mean insolation, from a FULL trajectory."""
    return float(_band_mean_S(S_full, band).max())


def winter_steps(S, band, frac=WINTER_S_FRAC, s_ref=None):
    """Boolean (time,) mask: band-mean S <= frac * s_ref.

    s_ref defaults to the max within S itself; pass seasonal_ref(S_full, band)
    when S is a sub-window of a longer trajectory."""
    s_band = _band_mean_S(S, band)
    if s_ref is None:
        s_ref = float(s_band.max())
    return s_band <= frac * s_ref

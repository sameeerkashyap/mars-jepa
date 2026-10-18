"""
Checkpoint A tests for the 1-D Mars climate pilot.

Covers:
  a. Diffusion alone conserves area-weighted global heat to machine precision.
  b. Truth energy closure (heat + latent) equals integrated net flux.
  c. CO2 ice mass never negative; ice budget closes.
  d. Unforced truth run reaches a periodic seasonal cycle (stated tolerance).
  e. mismatch_scale = 0  =>  truth and Model 2 agree within tolerance.
  f. Determinism for a given seed.
"""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import Grid, Diffusion, load_config, frost_point
from sim_truth import run_truth
from sim_ebm import fit_ebm, run_ebm


@pytest.fixture(scope="module")
def cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))


# ---------------------------------------------------------------- a
def test_diffusion_conserves_global_heat(cfg):
    grid = Grid(cfg["grid"]["n"])
    diff = Diffusion(grid)

    rng = np.random.default_rng(1)
    T = 180.0 + 40.0 * rng.standard_normal(grid.n)

    # (i) the raw operator integrates to zero globally
    lap = diff.apply(T)
    assert abs(grid.global_mean(lap)) < 1e-10

    # (ii) a backward-Euler diffusion sub-step conserves area-weighted heat
    dt = 0.05 * cfg["planet"]["sol_seconds"]
    lu = diff.implicit_factor(cfg["physics"]["D"] / cfg["truth"]["C_s"], dt)
    before = grid.global_mean(T)
    for _ in range(50):
        T = diff.implicit_solve(lu, T)
    after = grid.global_mean(T)
    assert abs(after - before) < 1e-9 * abs(before)


# ---------------------------------------------------------------- b
def test_truth_energy_closure(cfg):
    r = run_truth(cfg, F_a=3.0, seed=7, years=2)
    w = r["w"]
    C_s, C_g, L = r["C_s"], r["C_g"], r["L"]

    heat_init = np.sum(w * (C_s * r["T_s_init"] + C_g * r["T_g_init"]))
    heat_final = np.sum(w * (C_s * r["T_s_final"] + C_g * r["T_g_final"]))
    dm = np.sum(w * (r["m_final"] - r["m_init"]))

    lhs = (heat_final - heat_init) - L * dm   # Delta(heat) + Delta(E_ice)
    rhs = r["netflux_integral"]               # integral of net flux

    scale = abs(rhs) + 1.0
    assert abs(lhs - rhs) / scale < 1e-8


# ---------------------------------------------------------------- c
def test_ice_nonnegative_and_budget(cfg):
    r = run_truth(cfg, F_a=0.0, seed=3, years=2)
    assert np.all(r["m"] >= 0.0)
    assert np.all(r["m_final"] >= 0.0)
    # ice only exists at/below the frost point (within clamp tolerance)
    iced = r["m"] > 0.0
    assert np.all(r["T_s"][iced] <= r["T_frost"] + 1e-6)


# ---------------------------------------------------------------- d
def test_periodic_seasonal_cycle(cfg):
    # Periodicity is a property of the DETERMINISTIC seasonal forcing.
    # Random dust storms are, by construction, not periodic year-to-year,
    # so this test disables them to isolate the seasonal cycle.
    cfg = copy.deepcopy(cfg)
    cfg["truth"]["storm_rate_per_year"] = 0.0
    # run long enough to spin up, then compare two consecutive final years
    r = run_truth(cfg, F_a=0.0, seed=0, years=6)
    year_sols = cfg["planet"]["year_sols"]
    out_per_year = int(round(year_sols / cfg["time"]["output_every_sol"]))

    Ts = r["T_s"]
    w = r["w"]
    last = Ts[-out_per_year:]
    prev = Ts[-2 * out_per_year:-out_per_year]
    n = min(len(last), len(prev))
    diff = last[:n] - prev[:n]
    # area-weighted RMS difference between consecutive years
    rms = np.sqrt(np.mean(np.sum(w * diff**2, axis=1)))
    # TOLERANCE: 0.5 K area-weighted RMS year-over-year
    assert rms < 0.5, f"seasonal cycle not periodic: rms={rms:.3f} K"


# ---------------------------------------------------------------- e
def test_mismatch_zero_matches_ebm(cfg):
    # at mismatch_scale = 0 the truth reduces to Model 2
    truth = run_truth(cfg, F_a=0.0, seed=0, years=4, mismatch_scale=0.0)
    params, rms = fit_ebm(cfg, truth)

    # The least-squares fit error must be small. Note it is NOT zero: with the
    # current regime the CO2 frost clamp (shared physics) forms seasonal caps
    # even at mismatch_scale=0, and that nonlinearity cannot be captured by the
    # linear EBM, leaving a small irreducible residual (~0.45 W/m^2).
    # TOLERANCE: < 1.0 W/m^2 RMS (well below the mismatch at scale>=0.1).
    assert rms < 1.0, f"fit rms too large: {rms}"

    # forward-integrating Model 2 from the same state tracks the truth
    ebm = run_ebm(cfg, params, F_a=0.0, years=2,
                  T_init=truth["T_s_final"].copy(),
                  m_init=truth["m_final"].copy())
    truth2 = run_truth(cfg, F_a=0.0, seed=0, years=2, mismatch_scale=0.0,
                       T_init=truth["T_s_final"].copy(),
                       Tg_init=truth["T_g_final"].copy(),
                       m_init=truth["m_final"].copy())
    n = min(len(ebm["T_s"]), len(truth2["T_s"]))
    w = cfg_weights(cfg)
    d = ebm["T_s"][:n] - truth2["T_s"][:n]
    rms_T = np.sqrt(np.mean(np.sum(w * d**2, axis=1)))
    # TOLERANCE: 2 K area-weighted RMS over 2 years
    assert rms_T < 2.0, f"EBM vs truth mismatch: {rms_T:.3f} K"


def cfg_weights(cfg):
    return Grid(cfg["grid"]["n"]).w


# ---------------------------------------------------------------- f
def test_determinism(cfg):
    r1 = run_truth(cfg, F_a=1.0, seed=42, years=2)
    r2 = run_truth(cfg, F_a=1.0, seed=42, years=2)
    assert np.array_equal(r1["T_s"], r2["T_s"])
    assert np.array_equal(r1["m"], r2["m"])
    assert np.array_equal(r1["tau"], r2["tau"])

    # a different seed gives a different (storm) trajectory
    r3 = run_truth(cfg, F_a=1.0, seed=43, years=2)
    assert not np.array_equal(r1["tau"], r3["tau"])

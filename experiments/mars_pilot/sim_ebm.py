"""
MODEL 2 -- SIMPLIFIED ENERGY-BALANCE MODEL (EBM).

Used ONLY to supply the physics residual that quantifies how incomplete the
simplified physics is relative to the truth simulator.

Governing equation (single temperature, no subsurface, no dust):
    C_eff dT/dt = S(1-A) - eps*sigma*T^4 + D d/dx[(1-x^2) dT/dx] + F_a(t)

The free parameters C_eff, D, eps are fit by LEAST SQUARES to the UNFORCED
truth run. Because the fit is linear in (C_eff, D, eps) the solution is a
single linear-least-squares solve.  The CO2 frost clamp is applied here too
(both models share it); frost-pinned cells are excluded from the fit and from
the residual by a mask.
"""

from __future__ import annotations

import numpy as np

from common import (SIGMA, Grid, Diffusion, insolation, frost_point,
                    frost_clamp, load_config)


def _frost_mask(T, T_frost, tol=2.0):
    """True where a cell is NOT frost-pinned (usable for fit / residual).

    tol (K) excludes not only frost-pinned cells but also cells within a thin
    band above the frost point that are actively transitioning on/off the cap,
    whose finite-difference dT/dt would otherwise contaminate the fit.
    """
    return T > (T_frost + tol)


def fit_ebm(cfg, truth):
    """Fit (C_eff, D, eps) to an UNFORCED truth run by linear least squares.

    The truth run must have been produced with F_a = 0.  Returns a params dict
    and the RMS fit residual (W/m^2) over the unmasked cells.

    Linear model (per cell, per stored time):
        C_eff*dT/dt - D*diff + eps*sigma*T^4 = S(1-A)
    with design columns [dT/dt, -diff, sigma*T^4] and target S(1-A).
    """
    grid = Grid(cfg["grid"]["n"])
    diff = Diffusion(grid)
    phys = cfg["physics"]
    A_bare = phys["albedo_bare"]

    T = truth["T_s"]                     # (ntime, ncell)
    S = truth["S"]
    T_frost = truth["T_frost"]
    sol_s = cfg["planet"]["sol_seconds"]
    dt_out = cfg["time"]["output_every_sol"] * sol_s

    # dT/dt via central differences in time
    dTdt = np.gradient(T, dt_out, axis=0)
    # diffusion operator (per unit D) applied to each stored field
    diff_term = np.stack([diff.apply(T[k]) for k in range(T.shape[0])], axis=0)

    # mask out frost-pinned cells
    mask = _frost_mask(T, T_frost)
    # also drop the first/last time row (one-sided gradients) for cleanliness
    mask[0] = False
    mask[-1] = False

    rows = mask.ravel()
    col_dTdt = dTdt.ravel()[rows]
    col_diff = -diff_term.ravel()[rows]
    col_T4 = (SIGMA * T**4).ravel()[rows]
    target = ((1.0 - A_bare) * S).ravel()[rows]

    Amat = np.column_stack([col_dTdt, col_diff, col_T4])
    coef, *_ = np.linalg.lstsq(Amat, target, rcond=None)
    C_eff, D_fit, eps_fit = coef

    pred = Amat @ coef
    rms = float(np.sqrt(np.mean((pred - target) ** 2)))

    params = {"C_eff": float(C_eff), "D": float(D_fit), "eps": float(eps_fit),
              "A_bare": A_bare, "A_ice": phys["albedo_ice"],
              "L": truth["L"], "T_frost": T_frost}
    return params, rms


def residual(T, S, F_a, dt, params, cfg, mask=None):
    """Model-2 residual for an arbitrary temperature trajectory.

    Returns  C_eff*dT/dt - [S(1-A) - eps*sigma*T^4 + D*diff + F_a]
    for every (time, cell), with frost-pinned cells masked out (set to NaN).

    Parameters
    ----------
    T : (ntime, ncell)   temperature trajectory
    S : (ntime, ncell)   insolation
    F_a : scalar or (ntime,) or (ntime, ncell)  forcing in W/m^2
    dt : float           time spacing between rows (seconds)
    params : dict        fitted EBM parameters
    mask : optional boolean array, True where usable (else derived from frost)
    """
    grid = Grid(cfg["grid"]["n"])
    diff = Diffusion(grid)
    C_eff = params["C_eff"]
    D = params["D"]
    eps = params["eps"]
    A_bare = params["A_bare"]
    T_frost = params["T_frost"]

    dTdt = np.gradient(T, dt, axis=0)
    diff_term = np.stack([diff.apply(T[k]) for k in range(T.shape[0])], axis=0)

    Fa = np.asarray(F_a, dtype=float)
    if Fa.ndim == 0:
        Fa = np.full_like(T, float(Fa))
    elif Fa.ndim == 1:
        Fa = Fa[:, None] * np.ones((1, T.shape[1]))

    res = C_eff * dTdt - ((1.0 - A_bare) * S - eps * SIGMA * T**4
                          + D * diff_term + Fa)

    if mask is None:
        mask = _frost_mask(T, T_frost)
        mask[0] = False
        mask[-1] = False
    out = np.where(mask, res, np.nan)
    return out


def run_ebm(cfg, params, F_a=0.0, years=1, T_init=None, m_init=None):
    """Forward-integrate Model 2 with fitted parameters (for plotting).

    Same numerics as the truth model: explicit source + implicit diffusion
    + CO2 frost clamp. No subsurface, no dust.
    """
    grid = Grid(cfg["grid"]["n"])
    diff = Diffusion(grid)
    planet = cfg["planet"]
    co2 = cfg["co2"]

    C_eff = params["C_eff"]
    D = params["D"]
    eps = params["eps"]
    A_bare = params["A_bare"]
    A_ice = params["A_ice"]
    L = params["L"]
    T_frost = params["T_frost"]

    sol_s = planet["sol_seconds"]
    dt_sol = cfg["time"]["dt_sol"]
    dt = dt_sol * sol_s
    total_sol = years * planet["year_sols"]
    nsteps = int(round(total_sol / dt_sol))
    out_every = max(1, int(round(cfg["time"]["output_every_sol"] / dt_sol)))

    Fa_func = F_a if callable(F_a) else (lambda t, v=float(F_a): v)

    T = (np.full(grid.n, cfg["init"]["T_init"]) if T_init is None
         else T_init.copy())
    m = np.zeros(grid.n) if m_init is None else m_init.copy()

    lu = diff.implicit_factor(D / C_eff, dt)

    n_out = nsteps // out_every + 1
    out = {k: np.zeros((n_out, grid.n)) for k in ["T_s", "m", "S"]}
    out["t_sol"] = np.zeros(n_out)
    oi = 0

    for step in range(nsteps + 1):
        t_sol = step * dt_sol
        S = insolation(grid.x, t_sol, planet)
        Fa = Fa_func(t_sol)
        A = np.where(m > 0.0, A_ice, A_bare)
        absorbed = S * (1.0 - A)
        emitted = eps * SIGMA * T**4

        if step % out_every == 0 and oi < n_out:
            out["t_sol"][oi] = t_sol
            out["T_s"][oi] = T
            out["m"][oi] = m
            out["S"][oi] = S
            oi += 1
        if step == nsteps:
            break

        T = T + (dt / C_eff) * (absorbed - emitted + Fa)
        T = diff.implicit_solve(lu, T)
        T, m, _ = frost_clamp(T, m, C_eff, T_frost, L)

    for k in list(out.keys()):
        out[k] = out[k][:oi]
    out["x"] = grid.x
    out["lat"] = grid.lat
    out["w"] = grid.w
    out["T_s_final"] = T
    out["m_final"] = m
    return out


if __name__ == "__main__":
    from sim_truth import run_truth
    cfg = load_config()
    truth = run_truth(cfg, F_a=0.0, seed=0, years=3, mismatch_scale=0.0)
    params, rms = fit_ebm(cfg, truth)
    print("fitted params:", params)
    print("fit RMS (W/m^2):", rms)

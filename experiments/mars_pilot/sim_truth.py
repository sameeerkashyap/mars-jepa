"""
MODEL 1 -- TRUTH SIMULATOR for the 1-D Mars climate pilot.

Richer physics; this is what generates ALL data in later phases.

State per cell:
    T_s : surface temperature          (K)
    T_g : subsurface reservoir temp    (K)
    m   : CO2 ice mass                  (kg/m^2, >= 0)

Governing equations (continuous form):
    C_s dT_s/dt = S(1-A) exp(-tau) - eps*sigma*T_s^4
                  + D d/dx[(1-x^2) dT_s/dx] - G (T_s - T_g) + F_a(t)   (+ latent)
    C_g dT_g/dt = G (T_s - T_g)

  * tau(x,t) is a HIDDEN seasonal + storm dust optical depth.
  * A = albedo, higher where CO2 ice is present (m > 0).
  * F_a(t) is one scalar added radiative forcing, applied UNIFORMLY in
    latitude (assumption -- see README.md).
  * mismatch_scale scales BOTH tau and G; at 0 the truth reduces to Model 2.

Numerics: operator splitting per step dt:
    (1) explicit forward-Euler update of the source terms,
    (2) backward-Euler (implicit) conservative diffusion sub-step,
    (3) CO2 frost clamp with exact latent-heat bookkeeping.
"""

from __future__ import annotations

import numpy as np

from common import (SIGMA, Grid, Diffusion, insolation, frost_point,
                    frost_clamp, load_config)


def _as_forcing_func(F_a):
    """Normalise F_a into a function of t_sol returning W/m^2 (scalar)."""
    if callable(F_a):
        return F_a
    val = float(F_a)
    return lambda t: val


def dust_tau(grid, t_sol, cfg, storms, mismatch_scale):
    """Hidden dust optical depth tau(x, t), scaled by mismatch_scale.

    Seasonal baseline (peaks near perihelion) plus Gaussian-in-time storm
    pulses with a latitudinal (in x) shape. Returns array over cells.
    """
    from common import solar_longitude

    planet = cfg["planet"]
    t = cfg["truth"]
    Ls, _ = solar_longitude(t_sol, planet)
    Ls_peak = np.radians(t["dust_peak_Ls_deg"])
    seasonal = t["dust_base"] + t["dust_seasonal_amp"] * 0.5 * (
        1.0 + np.cos(Ls - Ls_peak))
    tau = np.full(grid.n, seasonal)

    for (t0, amp, dur, xc, xw) in storms:
        time_shape = np.exp(-((t_sol - t0) / dur) ** 2)
        lat_shape = np.exp(-((grid.x - xc) / xw) ** 2)
        tau = tau + amp * time_shape * lat_shape

    return mismatch_scale * np.maximum(tau, 0.0)


def _draw_storms(cfg, seed, total_sol):
    """Draw random dust-storm pulses deterministically from the seed."""
    rng = np.random.default_rng(seed)
    t = cfg["truth"]
    year = cfg["planet"]["year_sols"]
    n_years = total_sol / year
    expected = t["storm_rate_per_year"] * n_years
    n_storms = rng.poisson(expected)
    storms = []
    for _ in range(n_storms):
        t0 = rng.uniform(0.0, total_sol)
        amp = max(0.0, rng.normal(t["storm_amp_mean"], t["storm_amp_std"]))
        dur = t["storm_duration_sol"]
        xc = t["storm_lat_center"]
        xw = t["storm_lat_width"]
        storms.append((t0, amp, dur, xc, xw))
    return storms


def run_truth(cfg, F_a=0.0, seed=0, years=None, mismatch_scale=None,
              T_init=None, Tg_init=None, m_init=None, store=True):
    """Integrate the truth model.

    Returns a dict with time series (stored every output_every_sol):
        t_sol, Ls, x, lat, w
        T_s, T_g, m            (ntime, ncell)
        tau, S, F_a            (hidden dust, insolation, forcing)
        flux terms (absorbed, emitted, diffusion, coupling, forcing)
        energy-closure accumulators
    plus final state (T_s, T_g, m) for chaining runs.
    """
    grid = Grid(cfg["grid"]["n"])
    diff = Diffusion(grid)

    planet = cfg["planet"]
    phys = cfg["physics"]
    tr = cfg["truth"]
    co2 = cfg["co2"]

    if mismatch_scale is None:
        mismatch_scale = tr["mismatch_scale"]
    if years is None:
        years = cfg["init"]["spinup_years"]

    sol_s = planet["sol_seconds"]
    dt_sol = cfg["time"]["dt_sol"]
    dt = dt_sol * sol_s
    total_sol = years * planet["year_sols"]
    nsteps = int(round(total_sol / dt_sol))
    out_every = max(1, int(round(cfg["time"]["output_every_sol"] / dt_sol)))

    C_s = tr["C_s"]
    C_g = tr["C_g"]
    G = tr["G"] * mismatch_scale          # scaled coupling
    eps = phys["emissivity"]
    D = phys["D"]
    A_bare = phys["albedo_bare"]
    A_ice = phys["albedo_ice"]
    L = co2["latent_heat"]

    T_frost = frost_point(co2["surface_pressure_pa"], co2)

    # state
    T_s = (np.full(grid.n, cfg["init"]["T_init"]) if T_init is None
           else T_init.copy())
    T_g = (np.full(grid.n, cfg["init"]["T_init"]) if Tg_init is None
           else Tg_init.copy())
    m = np.zeros(grid.n) if m_init is None else m_init.copy()

    # snapshot of the initial state (for energy-closure accounting)
    T_s_init = T_s.copy()
    T_g_init = T_g.copy()
    m_init_snap = m.copy()

    storms = _draw_storms(cfg, seed, total_sol)
    Fa_func = _as_forcing_func(F_a)

    # implicit diffusion factor for the surface (constant D, C_s)
    lu = diff.implicit_factor(D / C_s, dt)

    # energy-closure accumulators (area-weighted, global)
    netflux_integral = 0.0   # integral of NetFlux dt (radiation + forcing)

    # output buffers
    n_out = nsteps // out_every + 1
    if store:
        out = {k: np.zeros((n_out, grid.n)) for k in
               ["T_s", "T_g", "m", "tau", "S",
                "absorbed", "emitted", "diffusion", "coupling"]}
        out["t_sol"] = np.zeros(n_out)
        out["Ls"] = np.zeros(n_out)
        out["F_a"] = np.zeros(n_out)
    oi = 0

    from common import solar_longitude

    for step in range(nsteps + 1):
        t_sol = step * dt_sol
        S = insolation(grid.x, t_sol, planet)
        tau = dust_tau(grid, t_sol, cfg, storms, mismatch_scale)
        Fa = Fa_func(t_sol)
        A = np.where(m > 0.0, A_ice, A_bare)

        absorbed = S * (1.0 - A) * np.exp(-tau)
        emitted = eps * SIGMA * T_s**4
        coupling = G * (T_s - T_g)
        # NetFlux for closure (diffusion integrates to zero globally; omit)
        netflux = absorbed - emitted + Fa

        if store and (step % out_every == 0) and oi < n_out:
            Ls, _ = solar_longitude(t_sol, planet)
            out["t_sol"][oi] = t_sol
            out["Ls"][oi] = np.degrees(Ls)
            out["F_a"][oi] = Fa
            out["T_s"][oi] = T_s
            out["T_g"][oi] = T_g
            out["m"][oi] = m
            out["tau"][oi] = tau
            out["S"][oi] = S
            out["absorbed"][oi] = absorbed
            out["emitted"][oi] = emitted
            out["diffusion"][oi] = D * diff.apply(T_s)
            out["coupling"][oi] = coupling
            oi += 1

        if step == nsteps:
            break

        # accumulate global net flux over this step (before the update)
        netflux_integral += dt * grid.global_mean(netflux)

        # (1) explicit source step
        T_s = T_s + (dt / C_s) * (absorbed - emitted - coupling + Fa)
        T_g = T_g + (dt / C_g) * coupling

        # (2) implicit conservative diffusion sub-step (surface only)
        T_s = diff.implicit_solve(lu, T_s)

        # (3) CO2 frost clamp (latent bookkeeping)
        T_s, m, _dm = frost_clamp(T_s, m, C_s, T_frost, L)

    result = {
        "x": grid.x, "lat": grid.lat, "w": grid.w,
        "T_frost": T_frost,
        "netflux_integral": netflux_integral,
        "T_s_final": T_s, "T_g_final": T_g, "m_final": m,
        "T_s_init": T_s_init, "T_g_init": T_g_init, "m_init": m_init_snap,
        "C_s": C_s, "C_g": C_g, "G": G, "D": D, "eps": eps, "L": L,
        "mismatch_scale": mismatch_scale,
    }
    if store:
        for k in list(out.keys()):
            out[k] = out[k][:oi]
        result.update(out)
    return result


if __name__ == "__main__":
    cfg = load_config()
    r = run_truth(cfg, F_a=0.0, seed=0, years=1)
    print("frost point:", r["T_frost"], "K")
    print("final global-mean T_s:", r["w"] @ r["T_s_final"], "K")
    print("max ice:", r["m_final"].max(), "kg/m^2")

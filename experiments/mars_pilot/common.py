"""
Shared physics for the 1-D (latitude-only) Mars climate pilot.

Everything that is common to the TRUTH simulator (sim_truth.py) and the
SIMPLIFIED EBM (sim_ebm.py) lives here:

  * the sin(latitude) grid and area weights,
  * daily-mean insolation S(x, t) with obliquity + eccentricity + polar night,
  * the conservative finite-volume diffusion operator,
  * the CO2 frost point and the frost-clamp / latent-heat bookkeeping,
  * config loading.

Sign / bookkeeping conventions used throughout (see README.md):

  Surface energy budget (per cell):
      C_s dT_s/dt = NetFlux + L dm/dt
  where NetFlux collects radiation, diffusion, coupling and forcing, and
  L dm/dt is the latent heat released when CO2 frost deposits (dm/dt > 0).
  Integrating:
      d/dt(C_s T_s) - L dm/dt = NetFlux
  so the "latent energy stored in ice" is E_ice = -L * m (depositing gas to
  ice releases heat, hence the ice carries negative energy relative to gas).
  The global closure checked in the tests is therefore:
      Delta(heat) + Delta(E_ice) = integral(NetFlux)
      i.e. Delta(C_s T_s + C_g T_g) - L * Delta(m) = integral(NetFlux).
"""

from __future__ import annotations

import numpy as np
import yaml
import scipy.sparse as sp
from scipy.sparse.linalg import splu

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W m^-2 K^-4 (physical constant)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# Grid in x = sin(latitude)
# --------------------------------------------------------------------------
class Grid:
    """Uniform grid in x = sin(latitude), N cell centres on [-1, 1].

    Area weight of a cell is proportional to its width in x (because the
    fraction of the sphere's surface between sin(lat) and sin(lat)+dx is
    exactly dx/2). Weights are normalised to sum to 1.
    """

    def __init__(self, n=64):
        self.n = n
        self.edges = np.linspace(-1.0, 1.0, n + 1)
        self.x = 0.5 * (self.edges[:-1] + self.edges[1:])   # cell centres
        self.dx = 2.0 / n
        self.lat = np.arcsin(self.x)                         # radians
        # area weights proportional to cell width (uniform here), normalised
        self.w = np.full(n, self.dx) / 2.0                  # sum(w) = 1
        # (1 - x^2) evaluated at interior interfaces; zero at the two poles
        self.e_iface = 1.0 - self.edges**2                  # length n+1
        self.e_iface[0] = 0.0
        self.e_iface[-1] = 0.0

    def global_mean(self, field):
        """Area-weighted global mean of a field defined at cell centres."""
        return np.sum(self.w * field, axis=-1)


# --------------------------------------------------------------------------
# Conservative finite-volume diffusion operator
#     D d/dx[(1 - x^2) dT/dx]  with zero flux at x = +/- 1
# --------------------------------------------------------------------------
class Diffusion:
    """Builds the FV Laplacian-like operator L such that (L @ T)_i approximates
    d/dx[(1 - x^2) dT/dx] on the grid, with zero flux at the poles.

    The operator rows satisfy  sum_i w_i (L @ T)_i = 0  for any T, so the
    diffusion term integrates to zero globally (heat conserving).
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        n = grid.n
        dx = grid.dx
        e = grid.e_iface
        L = np.zeros((n, n))
        for i in range(n):
            e_lo = e[i]       # interface i-1/2
            e_hi = e[i + 1]   # interface i+1/2
            L[i, i] = -(e_lo + e_hi) / dx**2
            if i > 0:
                L[i, i - 1] = e_lo / dx**2
            if i < n - 1:
                L[i, i + 1] = e_hi / dx**2
        self.L = L

    def apply(self, T):
        """Return d/dx[(1-x^2) dT/dx] (without the D coefficient)."""
        return self.L @ T

    def implicit_factor(self, coeff, dt):
        """LU factorisation of (I - dt * coeff * L) for a backward-Euler
        diffusion sub-step. `coeff` = D / C (units 1/s * x^2).

        L is tridiagonal, so we factorise the SPARSE matrix with SuperLU
        (O(n) factor + solve) instead of a dense LU. The factored matrix is
        identical to the dense one, so results match to round-off.
        """
        n = self.grid.n
        A = sp.identity(n, format="csc") - dt * coeff * sp.csc_matrix(self.L)
        return splu(A.tocsc())

    @staticmethod
    def implicit_solve(lu, rhs):
        return lu.solve(rhs)


# --------------------------------------------------------------------------
# Insolation:  daily-mean S(x, t) for Mars
# --------------------------------------------------------------------------
def solar_longitude(t_sol, cfg_planet):
    """Return (Ls, true_anomaly) in radians for time t_sol (sols) since
    perihelion (t=0 is perihelion). Solves Kepler's equation.

    Ls = solar longitude measured from the northern spring equinox.
    """
    P = cfg_planet["year_sols"]
    e = cfg_planet["eccentricity"]
    Ls_peri = np.radians(cfg_planet["Ls_perihelion_deg"])

    M = 2.0 * np.pi * (np.asarray(t_sol, dtype=float) % P) / P  # mean anomaly
    # Newton solve for eccentric anomaly E: M = E - e sin E
    E = M + e * np.sin(M)
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < 1e-13):
            break
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0),
                          np.sqrt(1 - e) * np.cos(E / 2.0))
    Ls = (nu + Ls_peri) % (2.0 * np.pi)
    return Ls, nu


def insolation(x, t_sol, cfg_planet):
    """Daily-mean insolation S(x, t) in W/m^2.

    Handles polar day / polar night exactly (S = 0 during polar night).
    `x` may be an array of cell centres; `t_sol` is a scalar time in sols.
    """
    S0 = cfg_planet["solar_constant"]
    e = cfg_planet["eccentricity"]
    obliq = np.radians(cfg_planet["obliquity_deg"])

    Ls, nu = solar_longitude(t_sol, cfg_planet)
    # distance factor (a/r)^2 = ((1 + e cos nu)/(1 - e^2))^2
    dist_factor = ((1.0 + e * np.cos(nu)) / (1.0 - e**2)) ** 2

    decl = np.arcsin(np.sin(obliq) * np.sin(Ls))  # solar declination
    phi = np.arcsin(np.clip(x, -1.0, 1.0))        # latitude

    # hour angle at sunset: cos H0 = -tan(phi) tan(decl)
    cosH0 = -np.tan(phi) * np.tan(decl)
    cosH0 = np.clip(cosH0, -1.0, 1.0)             # polar night / polar day
    H0 = np.arccos(cosH0)

    S = (S0 / np.pi) * dist_factor * (
        H0 * np.sin(phi) * np.sin(decl)
        + np.cos(phi) * np.cos(decl) * np.sin(H0)
    )
    return np.maximum(S, 0.0)


# --------------------------------------------------------------------------
# CO2 frost physics
# --------------------------------------------------------------------------
def frost_point(pressure_pa, cfg_co2):
    """CO2 frost (condensation) temperature for a given CO2 partial pressure.

    Uses the vapor-pressure fit  p = A * exp(-B / T)  inverted:
        T_frost = B / (ln A - ln p)
    with A, B taken from config (documented, flagged UNCERTAIN). With the
    default coefficients this gives ~148 K at 610 Pa, consistent with Mars
    polar CO2 frost temperatures.
    """
    A = cfg_co2["vapor_A"]
    B = cfg_co2["vapor_B"]
    return B / (np.log(A) - np.log(pressure_pa))


def frost_clamp(T_s, m, C_s, T_frost, L):
    """Apply the CO2 frost clamp to a tentative surface temperature.

    Parameters
    ----------
    T_s : array    tentative surface temperature after the flux + diffusion step
    m   : array    current CO2 ice mass (kg/m^2), m >= 0
    C_s : float    surface heat capacity (J/m^2/K)
    T_frost : float frost-point temperature (K)
    L   : float    latent heat of CO2 (J/kg)

    Returns
    -------
    T_new, m_new, dm   (dm = m_new - m_old, per cell)

    Energy bookkeeping (exact, per cell):
        C_s * (T_new - T_s_tentative) = L * dm
    Freezing (dm > 0) releases latent heat that holds T at the frost point;
    sublimation (dm < 0) absorbs it.
    """
    T_new = T_s.copy()
    m_new = m.copy()

    # --- freezing: tentative T below frost point ---
    freeze = T_s < T_frost
    dm_freeze = np.where(freeze, C_s * (T_frost - T_s) / L, 0.0)
    m_new = m_new + dm_freeze
    T_new = np.where(freeze, T_frost, T_new)

    # --- sublimation: tentative T above frost point but ice present ---
    melt = (~freeze) & (m_new > 0.0)
    # energy available to sublimate (J/m^2)
    e_avail = np.where(melt, C_s * (T_s - T_frost), 0.0)
    dm_possible = e_avail / L                     # mass that could sublimate
    dm_sub = np.minimum(dm_possible, m_new)       # limited by available ice
    dm_sub = np.where(melt, dm_sub, 0.0)
    m_new = m_new - dm_sub
    # if all ice sublimated, remaining energy warms the surface above frost
    T_after_sub = T_frost + (e_avail - dm_sub * L) / C_s
    T_new = np.where(melt, T_after_sub, T_new)

    dm = m_new - m
    return T_new, m_new, dm

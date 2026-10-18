"""
Checkpoint A plots.  Writes PNGs into figures/.

  1. truth_T_vs_lat_season.png : truth T_s vs latitude and season (unforced)
  2. polar_ice.png            : polar CO2 ice mass over one Mars year
  3. truth_vs_ebm_forced.png  : truth vs fitted Model 2 under constant F_a
                                in {2, 5, 10, 20} W/m^2
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Grid, load_config
from sim_truth import run_truth
from sim_ebm import fit_ebm, run_ebm

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def spin_up(cfg, years, mismatch_scale=None, seed=0, F_a=0.0):
    """Run to a spun-up state and return the final state."""
    r = run_truth(cfg, F_a=F_a, seed=seed, years=years,
                  mismatch_scale=mismatch_scale)
    return r


def plot_T_vs_lat_season(cfg):
    year = cfg["planet"]["year_sols"]
    spin = spin_up(cfg, years=cfg["init"]["spinup_years"])
    # one more year from the spun-up state
    r = run_truth(cfg, F_a=0.0, seed=0, years=1,
                  T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                  m_init=spin["m_final"])
    Ls = r["Ls"]
    lat = np.degrees(r["lat"])
    # sort by Ls for a clean seasonal axis
    order = np.argsort(Ls)
    fig, ax = plt.subplots(figsize=(8, 5))
    pcm = ax.pcolormesh(Ls[order], lat, r["T_s"][order].T, shading="auto",
                        cmap="inferno")
    fig.colorbar(pcm, ax=ax, label="surface temperature T_s (K)")
    ax.set_xlabel("solar longitude Ls (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title("Truth surface temperature vs latitude and season (unforced)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "truth_T_vs_lat_season.png"), dpi=120)
    plt.close(fig)


def plot_polar_ice(cfg):
    spin = spin_up(cfg, years=cfg["init"]["spinup_years"])
    r = run_truth(cfg, F_a=0.0, seed=0, years=1,
                  T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                  m_init=spin["m_final"])
    lat = np.degrees(r["lat"])
    npole = np.argmax(lat)   # closest to +90
    spole = np.argmin(lat)   # closest to -90
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r["Ls"], r["m"][:, npole], label=f"N pole ({lat[npole]:.0f} deg)")
    ax.plot(r["Ls"], r["m"][:, spole], label=f"S pole ({lat[spole]:.0f} deg)")
    ax.set_xlabel("solar longitude Ls (deg)")
    ax.set_ylabel("CO2 ice mass (kg/m^2)")
    ax.set_title("Polar CO2 ice mass over one Mars year (unforced truth)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "polar_ice.png"), dpi=120)
    plt.close(fig)


def plot_truth_vs_ebm_forced(cfg):
    grid = Grid(cfg["grid"]["n"])
    w = grid.w
    # fit EBM on scale-0 unforced run
    base = run_truth(cfg, F_a=0.0, seed=0, years=3, mismatch_scale=0.0)
    params, _ = fit_ebm(cfg, base)

    # common spun-up start (unforced, default mismatch)
    spin = spin_up(cfg, years=cfg["init"]["spinup_years"])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, 4))
    for Fa, c in zip([2.0, 5.0, 10.0, 20.0], colors):
        tr = run_truth(cfg, F_a=Fa, seed=0, years=2,
                       T_init=spin["T_s_final"], Tg_init=spin["T_g_final"],
                       m_init=spin["m_final"])
        eb = run_ebm(cfg, params, F_a=Fa, years=2,
                     T_init=spin["T_s_final"].copy(),
                     m_init=spin["m_final"].copy())
        n = min(len(tr["T_s"]), len(eb["T_s"]))
        tmean = np.sum(w * tr["T_s"][:n], axis=1)
        emean = np.sum(w * eb["T_s"][:n], axis=1)
        t = tr["t_sol"][:n]
        ax.plot(t, tmean, color=c, label=f"truth  F_a={Fa:g}")
        ax.plot(t, emean, color=c, ls="--", label=f"Model 2 F_a={Fa:g}")
    ax.set_xlabel("time (sols)")
    ax.set_ylabel("global-mean surface temperature (K)")
    ax.set_title("Truth (solid) vs fitted Model 2 (dashed) under constant F_a")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "truth_vs_ebm_forced.png"), dpi=120)
    plt.close(fig)


def main():
    plot_T_vs_lat_season(cfg=load_config())
    plot_polar_ice(cfg=load_config())
    plot_truth_vs_ebm_forced(cfg=load_config())
    print("wrote figures to", OUT + "/")


if __name__ == "__main__":
    main()

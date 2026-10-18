"""
Solver regression check: capture a reference trajectory signature with
`--save`, then after the solver change run `--check` to confirm outputs match
the pre-change run within tolerance.

Covers truth (with subsurface, dust, frost) and the EBM forward integrator,
across a couple of configs, so any change to the diffusion solve shows up.
"""

import sys

import numpy as np

from common import load_config
from sim_truth import run_truth
from sim_ebm import fit_ebm, run_ebm

REF = "regression_ref.npz"


def signature():
    cfg = load_config()
    sig = {}
    # truth: forced, dusty, subsurface active
    r1 = run_truth(cfg, F_a=3.0, seed=1, years=2, mismatch_scale=0.25)
    sig["truth_Ts"] = r1["T_s"]
    sig["truth_Tg"] = r1["T_g"]
    sig["truth_m"] = r1["m"]
    # truth: unforced, scale 0
    r0 = run_truth(cfg, F_a=0.0, seed=0, years=3, mismatch_scale=0.0)
    sig["truth0_Ts"] = r0["T_s"]
    # ebm fit + forward run
    params, rms = fit_ebm(cfg, r0)
    sig["ebm_fit"] = np.array([params["C_eff"], params["D"], params["eps"], rms])
    eb = run_ebm(cfg, params, F_a=5.0, years=2)
    sig["ebm_Ts"] = eb["T_s"]
    sig["ebm_m"] = eb["m"]
    return sig


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    sig = signature()
    if mode == "--save":
        np.savez(REF, **sig)
        print(f"saved reference signature to {REF}")
        for k, v in sig.items():
            print(f"  {k}: shape {np.shape(v)}")
        return
    ref = np.load(REF)
    worst = 0.0
    print(f"{'field':>12} {'max_abs_diff':>14} {'ref_scale':>12}")
    for k in sig:
        d = np.max(np.abs(sig[k] - ref[k]))
        scale = np.max(np.abs(ref[k])) + 1e-30
        worst = max(worst, d)
        print(f"{k:>12} {d:>14.3e} {scale:>12.3e}")
    # TOLERANCE: solver change (dense LU -> sparse/banded LU of the same matrix)
    # must reproduce trajectories to ~1e-6 absolute (temperatures ~1e2 K, ice
    # ~1e3 kg/m2), i.e. far below any physical relevance.
    tol = 1e-6
    print(f"\nworst max_abs_diff = {worst:.3e}  (tolerance {tol:.0e})")
    assert worst < tol, "solver change altered outputs beyond tolerance!"
    print("PASS: outputs match pre-change run within tolerance.")


if __name__ == "__main__":
    main()

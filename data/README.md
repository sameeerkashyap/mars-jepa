# MarsWorld data

- `synthetic/pilot/`: heating trajectories and their manifest.
- `synthetic/pi_jepa/`: spatial diffusion arrays.
- `arco_mars/`: future historical climate data.
- `specification/`: settings, split rules, thresholds, and hashes.
- `diagnostics/pilot/`: saved simulator checks and logs.

Large arrays are ignored by Git. Specifications and the pilot manifest are
tracked. Reader code lives in `src/marsworld/data/`.

Read [the data plan](../docs/DATA_AND_MODELING.md).

# Data & Solver

Stage A of the data strategy: a reproducible synthetic solver produces
trajectories with a known source of truth, and a torch dataset exposes
`(context, future-rollout, forcing)` samples with train/val/test splitting and
data-efficiency subsampling.

## Solver

::: marsworld.data.solver

## Dataset

::: marsworld.data.dataset

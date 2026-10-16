# Training

The composite loss combines JEPA alignment, forecast reconstruction and the
physics terms. The `Trainer` owns the optimisation loop, multi-step latent
rollout, EMA target updates and checkpointing, and runs unchanged on CPU or GPU.

## Composite loss

::: marsworld.training.losses

## Trainer

::: marsworld.training.trainer

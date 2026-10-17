# Architecture

MarsWorld implements a **JEPA** augmented with a physical-state decoder so that
physics constraints can be evaluated in interpretable field space.

## Data flow

1. **Context encoder** maps the current field $x_t$ to a latent $z_t$.
2. **Latent predictor** advances $z_t$ to $\hat z_{t+1}$ conditioned on the
   environmental **forcing** $a_t$ (wind, source strength).
3. **Target encoder** (an EMA / stop-gradient copy of the context encoder)
   produces the training target $\tilde z_{t+1}$ from the *true* next field.
4. **Physical decoder** reconstructs $\hat x_{t+1}$ from $\hat z_{t+1}$.

## Objective

$$
\mathcal{L} =
\lambda_{\text{JEPA}}\,\mathcal{L}_{\text{JEPA}}
+ \lambda_{\text{forecast}}\,\mathcal{L}_{\text{forecast}}
+ \lambda_{\text{PDE}}\,\mathcal{L}_{\text{PDE}}
+ \lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}}
$$

- $\mathcal{L}_{\text{JEPA}}$ — alignment between $\hat z_{t+1}$ and the
  stop-gradient target $\tilde z_{t+1}$.
- $\mathcal{L}_{\text{forecast}}$ — reconstruction of $\hat x_{t+1}$ vs truth.
- $\mathcal{L}_{\text{PDE}}$ — residual of the discretised diffusion equation,
  computed on decoded fields.
- $\mathcal{L}_{\text{cons}}$ — domain-integral conservation.

## Ablation ladder

The same code yields the research plan's three models by changing weights:

| Model | `jepa_weight` | `physics.enabled` | Description |
| --- | --- | --- | --- |
| A — simple field predictor | 0 | false | forecast only |
| B — JEPA latent predictor | >0 | false | JEPA + forecast |
| C — physics-informed JEPA | >0 | true | full objective |

## Why a decoder?

The physics loss must be evaluated on physically meaningful quantities. Decoding
latents back into fields lets us compute PDE residuals and conservation
diagnostics directly, rather than imposing constraints on arbitrary latent
coordinates.

## Why residual prediction (and why it matters)

The decoder predicts the field **increment**, not the absolute field:

$$ \hat x_{t+1} = x_t + \text{decode}(\hat z_{t+1}). $$

This is controlled by `model.predict_residual` (default `True`). The reason is
empirical and important: for small time steps the frame-to-frame change is tiny,
so the **persistence baseline** $x_{t+1} := x_t$ is extremely strong. If the
model predicts the absolute field, its reconstruction-error floor exceeds the
actual change and it *loses to doing nothing*. In residual mode, a zero decoder
output is exactly persistence, so the model can only improve on it.

## Task difficulty: `data.substeps`

The solver takes `substeps` internal steps per **stored** frame. Larger values
enlarge the per-frame change, weakening the persistence baseline and giving the
model real dynamics to learn. The physics residual automatically uses the
effective spacing `frame_dt = dt * substeps`.

## Validation & audit

The core was audited empirically (see repo history). Headline results:

| Check | Result |
| --- | --- |
| Gradient flow / target-encoder freeze | all params receive grad; target frozen |
| Representation collapse | none (latent std stable, 100% dims active) |
| Autoencoder capacity | normalized recon MSE ~0.001 on diffusion fields |
| Beats persistence baseline | **yes** — skill grows with difficulty |

Persistence skill $1 - \text{MSE}_{\text{model}}/\text{MSE}_{\text{persistence}}$
(2000-step controlled run):

| `substeps` | skill |
| --- | --- |
| 1 (near-degenerate) | +0.04 |
| 8 (default) | +0.44 |
| 16 (harder) | +0.62 |

That skill **increases** with task difficulty is the key signal: the model has
learned dynamics, not merely learned to echo its input. `persistence_mse` is
logged every epoch so this can be tracked during training.

!!! warning "Constraints enforced by the code"
    - `grid_size` must be divisible by 8 (encoder downsamples ×2 three times;
      the decoder now raises instead of silently resizing).
    - Any `hidden_channels` width works (GroupNorm groups = `gcd(channels, 8)`).
    - The global-pool bottleneck is adequate for *smooth* diffusion fields but
      would lose high-frequency detail (e.g. sharp dust fronts); a spatial /
      U-Net latent is the recommended upgrade for that regime.

## Extending to new parameters

The pipeline is parameter-agnostic. To add **dust transport**:

1. Add a `DustAdvectionSolver` in `marsworld/data/solver.py` with the same
   `generate_dataset` signature.
2. Point `data.parameter: dust` in a config; bump `model.in_channels` if the
   field is multi-channel.
3. Add the matching advection-diffusion residual in `marsworld/physics/`.

No changes to the model core, trainer, or deployment are required.

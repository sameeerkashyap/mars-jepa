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

## Extending to new parameters

The pipeline is parameter-agnostic. To add **dust transport**:

1. Add a `DustAdvectionSolver` in `marsworld/data/solver.py` with the same
   `generate_dataset` signature.
2. Point `data.parameter: dust` in a config; bump `model.in_channels` if the
   field is multi-channel.
3. Add the matching advection-diffusion residual in `marsworld/physics/`.

No changes to the model core, trainer, or deployment are required.

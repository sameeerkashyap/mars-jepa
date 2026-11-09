# What we model and which data we use

We predict **how surface temperature and CO₂ frost respond when we add heat over
time**. We test whether memory and physical equations improve these predictions.

## Synthetic data first, then ARCO-Mars

| Source | What it contains | Purpose | Status |
|---|---|---|---|
| Synthetic heating pilot | Climate sequences made by our simulator, with heating and matching no-heating runs | Train models and test stronger heating or new schedules | Generator and specification exist; full arrays must be recovered or regenerated |
| ARCO-Mars | Historical Mars records combining observations with atmospheric simulations | Check present-day climate and run a separate forecasting experiment | Integration is planned |

```text
Synthetic heating data              ARCO-Mars historical data
          ↓                                    ↓
Learn responses to actions           Learn and check present-day climate
          ↓                                    ↓
Test new heating schedules           Test held-out Mars years
```

ARCO-Mars contains historical climate records. Controlled simulations supply
the examples of engineered heating. We report the experiments separately.

## Our reference simulator

The simulator divides Mars into **64 latitude bands**, from south to north.
It tracks surface temperature, underground temperature, and CO₂ ice amount.
It calculates sunlight, dust blocking sunlight, heat lost to space, heat exchange,
and the added heat we choose.

```text
Sunlight + dust + added heat
             ↓
     Surface temperature
        ↕            ↕
 Underground heat   CO₂ frost
```

Added heat can turn frost into gas while temperature stays near the frost point.
Once the ice is gone, the surface can warm further. Predicting when this happens
tests how a model handles hidden state and physical processes.

The reference code is `src/marsworld/physics/reference.py`. Shared heat and frost
calculations live in `src/marsworld/physics/surface.py`.

## What trains the models

A trajectory is a sequence of climate states over time. All models in the heating
comparison receive the same kinds of observations.

| Quantity | Role |
|---|---|
| Past surface temperatures across 64 bands | Observed context |
| Sunlight, including the forecast period | Known input |
| Heating schedule, including the forecast period | Action input |
| Future surface temperatures | Training target |
| Ice amount, underground temperature, dust | Hidden simulator values for diagnostics and labeled special comparisons |

The current protocol observes **40 sols** and predicts **120 sols**. A sol is one
Mars day. Future temperatures remain hidden during prediction.

```text
Past temperatures + sunlight + heating schedule
                       ↓
                Prediction model
                       ↓
             Future temperatures
                       ↓
      Compare with the reference simulation
```

Each heated trajectory has a no-heating control with the same starting state and
storm seed. Their temperature difference measures the response to heating. The
normal training loader hides both these control answers and the hidden states.
Evaluation reads the control answers separately.

## Training, validation, and tests

- **Training:** constant and ramp heating within 0–5 W/m². W/m² means heat
  supplied per square metre.
- **Validation:** separate trajectories used to choose model settings.
- **First test:** unseen step and pulse schedules within 0–5 W/m².
- **Second test:** constant heating within 8–10 W/m², as specified in the
  committed manifest and later design notes.

Whole trajectories use different seeds across sets. The generator's historical
default for the second test is 7.5–10 W/m². Supply `--tier2-mag 8 10` to generate
the manifest's design. Record a new manifest and hashes for any new dataset.

The current training fractions are 10%, 30%, and 100%. They let us measure how
much data each method needs. Protocol changes need a recorded update.

## Methods

| Method | What it uses | Location |
|---|---|---|
| Latent JEPA | A compact internal state that changes under heating | `src/marsworld/models/latent_jepa/` |
| GRU | Memory learned from observations | `benchmarks/gru/` |
| RSSM | Recurrent memory and a probability distribution over hidden state | `benchmarks/rssm/` |
| Transformer | Relationships within the observed sequence | `benchmarks/transformer/` |
| Hybrid | Heat equations plus a learned correction | `benchmarks/hybrid/` |
| Linear | A simple relation between recent temperatures and inputs | `benchmarks/linear/` |

The physics-only comparison uses the hybrid equations without a learned
correction. The no-heating-input GRU measures how much the action input helps.
The benchmark registry gives these methods the same rollout interface.

Current learned outputs are temperatures. Explicit ice estimates and changing
pressure predictions need additional model work. Physics penalties are available
for selected architectures; the registered sweep does not enable them for every arm.

## The separate spatial prototype

`src/marsworld/models/pi_jepa/` uses a two-dimensional diffusion generator in
`src/marsworld/data/solver.py`. It has a two-value transport forcing vector.
Its arrays live under `data/synthetic/pi_jepa/`.

The heating pilot uses `data/synthetic/pilot/`. Each track has its own loader,
configuration, and trainer. Within a track, its models share the same dataset.

## ARCO-Mars next

Check surface temperature, CO₂ ice, pressure, and dust variable meanings and
units. Average using the right time and area weights. Train on earlier whole
Mars years and test on held-out years. Compare against persistence and seasonal
averages. Separately compare seasonal temperature and ice with our simulator.

See the [ARCO-Mars paper](https://arxiv.org/html/2606.21701v1) and
[detailed revision plan](pilot/PAPER_REVISION_PLAN.md).

Pressure stays at 610 Pa in the current heating simulator. Water, oxygen, and
changing atmospheric composition require additional physics. Real-Mars accuracy
still needs the climate checks above.

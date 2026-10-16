# Mars PI-JEPA: Physics-Informed Predictive World Model for Martian Dust Dynamics

**Status:** Research proposal and implementation plan  
**Target:** Independent, experimentally validated component of the broader MarsWorld research agenda  
**Initial scope:** One Martian physical process, controlled forcing experiments, and data-efficiency evaluation

## 1. Project summary

Build a compact **physics-informed Joint Embedding Predictive Architecture (PI-JEPA)** that predicts the evolution of one physical aspect of the Martian atmosphere. The recommended first target is **dust transport**, with a simpler fallback of **surface temperature dynamics**.

Rather than attempting an end-to-end agent that simulates centuries of terraforming, isolate one testable scientific machine learning question:

> Can physically constrained latent predictive models improve data efficiency, long-horizon forecast consistency, and generalization to unseen forcing conditions in Martian atmospheric dynamics?

The result should be an independently evaluable research component. It can later serve as a validated module or empirical motivation in **MarsWorld**, the larger proposed intervention-conditioned planetary world model.

**Important limitation:** Better performance on dust transport does *not* establish the feasibility of multi-century climate prediction or terraforming control. The scope of any claim must match the evaluation performed.

## 2. Recommended problem choice

| Physical process | Prediction target | Key physics | Feasibility | Research advantage |
| --- | --- | --- | --- | --- |
| **Dust transport (recommended)** | Future dust concentration/optical-depth fields | Advection, diffusion, lifting, deposition | Moderate | Spatial prediction, controlled forcing, conservation diagnostics |
| Seasonal CO₂ exchange | Atmospheric and surface CO₂ reservoirs | Phase changes, mass and energy budgets | Moderate–hard | Strong connection to pressure and terraforming |
| Surface temperature | Temperature fields or time series | Radiation and surface heat storage | Easier | Fastest proof of concept |

**Decision:** Start with dust transport if suitably time-resolved wind and dust fields can be assembled. Otherwise use surface temperature for the first controlled study.

## 3. Research hypotheses

- **H1 — Forecasting:** A JEPA-style latent predictor can learn short- and medium-range evolution of a Martian physical field.
- **H2 — Physical consistency:** Adding a physically evaluated constraint reduces numerical/physical violations relative to a latent-only model.
- **H3 — Data efficiency:** Physics-informed prediction requires fewer labeled trajectories to reach comparable forecast error.
- **H4 — Forcing generalization:** Conditioning on forcing or source controls improves predictions for unseen forcing strengths or combinations.

Treat all four as hypotheses; do not assert their outcome before running experiments.

## 4. Minimal architecture

```text
 Current physical field x_t                 Future field x_(t+Δt)
            |                                        |
     Context encoder                            Target encoder
            |                               (EMA / stop-gradient)
            v                                        v
       Latent z_t                             Target z_(t+Δt)
            |
            +-- forcing / wind / source inputs a_t, c_t
            |
     Latent dynamics predictor
            |
            v
   Predicted latent z_hat_(t+Δt)  ----- JEPA alignment loss ----- target
            |
    Physical-state decoder
            |
            v
     Predicted field x_hat_(t+Δt)
            |
       Forecast loss + PDE residual / boundary diagnostics
```

### Components

1. **Context encoder:** Compact CNN or ViT encoder for gridded Martian physical states.
2. **Target encoder:** Momentum-updated or stop-gradient encoder that creates training targets in representation space.
3. **Predictor:** Small recurrent block or MLP conditioned on current latent state, time step, and environmental forcing. Add recurrence if multi-step history matters.
4. **Physical decoder:** Predicts interpretable fields, enabling direct computation of PDE and conservation losses.
5. **Physics loss:** Evaluated on decoded fields using discretized governing equations and boundary conditions.

**Pretraining vs fine-tuning:** A JEPA-inspired model trained from scratch on Mars-specific fields is *not* fine-tuning. If you intend to fine-tune a pretrained PI-JEPA checkpoint, first verify that its inputs, discretization, model weights, and licensing support the transfer. A model trained on unrelated image/video data may not yield a useful initialization for numerical atmospheric fields.

## 5. Physics formulation: dust transport

A simplified transported dust quantity \(q\) can obey

\[
\frac{\partial q}{\partial t}
+ \nabla\cdot(q\mathbf{u})
= \nabla\cdot(K\nabla q) + S - D,
\]

where:

- \(q\): dust tracer concentration (or a modeled proxy; optical depth is not automatically identical to concentration);
- \(\mathbf u\): prescribed or modeled wind field;
- \(K\): transport/diffusion coefficient;
- \(S\): lifting/source term;
- \(D\): deposition/sink term.

A candidate objective is

\[
\mathcal L =
\lambda_{\mathrm{JEPA}}\mathcal L_{\mathrm{JEPA}}
+\lambda_{\mathrm{forecast}}\mathcal L_{\mathrm{forecast}}
+\lambda_{\mathrm{PDE}}\mathcal L_{\mathrm{PDE}}.
\]

Compute \(\mathcal L_{\mathrm{PDE}}\) from predicted physical fields, not arbitrary latent coordinates without a justified mapping. Specify a stable numerical discretization, correct handling of winds, and appropriate boundaries. For a closed domain with zero source and sink, dust mass conservation can be tested; with lifting and deposition, the budget must explicitly include those terms.

## 6. Data strategy

### Stage A: Controlled synthetic trajectories

Start with a simplified dust advection–diffusion solver. Vary:

- Initial dust concentration/distribution;
- Prescribed wind fields;
- Diffusivity;
- Dust lifting intensity and spatial location;
- Deposition strength;
- Seasonal or time-dependent forcing, if represented.

Create trajectories that have a known, reproducible source of truth. Retain no-intervention trajectories for counterfactual comparison.

### Stage B: Mars-specific fields

Investigate these data sources for realistic boundary conditions, variability, and external evaluation:

- **EMARS:** Mars atmospheric reanalysis;
- **OpenMARS:** Independent Mars reanalysis;
- **Mars Climate Database (MCD):** Climate/statistical environmental fields, useful for contextual sampling.

**Caution:** Reanalysis/climatology data are *not* automatically counterfactual trajectories under engineered interventions. Use a solver with explicitly varied forcing to construct such examples, and separately report whether a model trained on synthetic data transfers to Mars-like fields.

## 7. Baselines and ablations

Minimum comparison:

| Model | JEPA alignment | Action/forcing conditioning | Physics regularization |
| --- | --- | --- | --- |
| A. Simple field predictor | No | Optional | No |
| B. JEPA latent predictor | Yes | Yes | No |
| C. Physics-informed JEPA | Yes | Yes | Yes |

Useful additional baselines if time permits: action-conditioned FNO, a reduced-order physical surrogate, or a recurrent encoder–decoder. For a decisive architecture comparison, use matched data splits and report approximate parameter counts and training compute.

Ablate: loss weights, training data fraction, rollout length, and forcing extrapolation. Do not claim JEPA is categorically better than transformers or neural operators unless the comparison supports that statement.

## 8. Experimental protocol

### Experiment 1: One-step and multi-step prediction

Train each baseline on the same trajectories. Evaluate one-step error and free-running rollout error over increasing horizons.

### Experiment 2: Sample efficiency

Train with **10%, 25%, 50%, and 100%** of training trajectories. Plot test forecast error and constraint violations against data fraction. Repeat across several random seeds if compute allows.

### Experiment 3: Out-of-distribution forcing

Hold out selected lifting strengths, wind regimes, diffusivity ranges, or combinations during training. Evaluate extrapolation to those settings. Keep the held-out protocol explicit, rather than calling all held-out data “OOD.”

### Experiment 4: Intervention response fidelity

For an action or forcing \(a\), compare the predicted change relative to no intervention with the solver's actual change:

\[
E_{\mathrm{response}}
=
\left\|
\left(\hat{x}_T(a)-\hat{x}_T(0)\right)
-
\left(x_T(a)-x_T(0)\right)
\right\|.
\]

This is more directly relevant to a future control-oriented world model than unconditional forecast accuracy alone.

### Report these metrics

- Normalized RMSE or MAE by physical variable;
- Rollout error against time horizon;
- PDE residual and source/sink-adjusted mass-budget error;
- Intervention response error;
- Runtime/inference speed and training cost;
- Optional: calibration or ensemble spread if probabilistic dynamics are implemented.

## 9. Expected deliverables

- [ ] Reproducible synthetic data generator and defined forcing controls;
- [ ] Dataset documentation, train/validation/test and OOD splits;
- [ ] Lightweight JEPA implementation with physical-field decoder;
- [ ] Physics-informed objective and diagnostics;
- [ ] Two or more relevant baselines and at least one physics-loss ablation;
- [ ] Data-efficiency and rollout-error plots;
- [ ] Intervention-response evaluation;
- [ ] Limitations section addressing synthetic-to-real transfer;
- [ ] Workshop-ready short paper and public experiment repository (subject to venue policies).

## 10. Proposed 4–6 week pilot timeline

| Week | Focus | Exit criterion |
| --- | --- | --- |
| 1 | Choose dust variables, build solver and split protocol | Reproducible trajectories + physical budget test |
| 2 | Train basic forecasting baseline and JEPA predictor | Stable one-step and multi-step training |
| 3 | Integrate physical decoder and PDE loss | Verified numerical loss and constraint diagnostics |
| 4 | Run data-efficiency and OOD forcing comparisons | Completed plots and result tables |
| 5 | Optional Mars-specific data validation and stronger baseline | Clear synthetic-vs-Mars evidence boundary |
| 6 | Paper, ablations, code cleanup | Submission-ready concise manuscript |

If data integration or solver validation is harder than expected, prioritize correct physical experiments and baselines over more architecture components.

## 11. How this strengthens MarsWorld

The original *World Models Are the Right Foundation for Simulating Mars Terraforming* manuscript proposes a physics-regularized latent dynamics model and a benchmark agenda, but does not yet demonstrate the proposed system's advantages. This project would provide a smaller, defensible empirical result:

**Potential supported contribution:** A physics-informed JEPA-like component has been evaluated for one Martian atmospheric process under controlled forcing, with measured strengths and failures in forecast accuracy, physical consistency, and data efficiency.

**Not yet supported:** That a complete recurrent world model is optimal for Mars, that the module can predict century-scale planetary evolution, or that an RL planner can reliably optimize terraforming interventions.

A possible independent paper title is:

> **Physics-Informed Joint Embedding Predictive Models for Data-Efficient Martian Dust Dynamics**

The independent paper should foreground a general scientific ML result, such as the effects of physical regularization on data efficiency and intervention extrapolation, rather than rely only on the novelty of applying JEPA to Mars.

## 12. Immediate decisions

1. **Choose task:** Dust transport versus faster surface-temperature proof of concept.
2. **Choose initialization:** JEPA from scratch versus a genuinely compatible pretrained PI-JEPA checkpoint.
3. **Choose ground truth:** Simplified PDE solver first; Mars-specific reanalysis for context or follow-up evaluation.
4. **Choose primary claim:** Data efficiency, physical fidelity, or OOD intervention response. Measure all three, but lead with the result the evidence actually supports.

---

*This is a proposed research and experimental plan, not a report of completed experiments. Venue fit and submission rules should be checked against the selected workshop's current call for papers.*

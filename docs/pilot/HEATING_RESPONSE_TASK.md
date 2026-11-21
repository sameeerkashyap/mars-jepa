# South-winter response task v1

This is the separate, user-approved follow-up to [issue 5](https://github.com/sameeerkashyap/mars-jepa/issues/5#issuecomment-7069855353).
The [original frost-point task](HEATING_CONTROL_TASK.md), its results, and its
`control-v1` identity remain unchanged. This task uses identity
`south-winter-response-v1` and data namespace `control-response-v1`.

## Contract

Use `physics.control.ResponseHeatingControl` with the same public methods as
`HeatingControl`. Keep the original physics, initial conditions, astronomical
onset, fixed disturbance draw, and seed rules. Start at 2223.8 sols (year day
218), after three unheated years. Use 120 sols, twelve ten-sol decisions,
requested heat in [0,10] W/m², and 300 W/m²·sol total heat allowance.
The same projection debits actual applied heat. Invalid actions do not advance
either world. This follow-up uses only development conditions 5000–5004.
For future data, keep conditions 5000–5023 in training, 5100–5107 in validation,
and 5200–5219 sealed, within this separate namespace. Conditions 9000–9999 stay
reserved. No data generation or held-out access is authorized by this screen.

At reset, copy the complete spun-up simulator into a private no-heating world.
Both worlds then advance on the same absolute clock, with identical fixed
storms. Apply the chosen heat only to the heated world; the reference always
receives zero. Do not redraw disturbances or reset the clock between actions.
The trusted private checkpoint contains both worlds, heat bookkeeping, and
metric accumulators. Loading it restores exact continuation.

For each daily post-transition sample and each fixed South-band cell, define
response as heated surface temperature minus matched no-heat temperature.
The target response is 2 K. Compute daily squared error with the same cell-area
weights as the original task. Interval reward is minus the sum of ten daily
errors. Episode RMSE is the square root of the mean of all 120 daily errors.
Report the mean of episode RMSEs across conditions. Maximum deviation and the
area/time fraction within 0.5 K now measure response error. No heat produces
exactly zero response and exactly 2 K RMSE, regardless of the reference climate.

The observation keys, units, fixed scaling, and 133-component vector order
remain as defined in the original contract. Daily transition fields still
contain only heated-world surface temperature, sunlight, and absolute time.
Neither reference temperature nor response is an observation. Do not expose
reference state, hidden physical state, future storms, or private checkpoints
to the world model or actor as inputs. Reward labels come from the private
matched simulator oracle. A learned reward predictor must estimate this reward
from permitted observations and action history. This synthetic oracle cannot
be obtained directly from one physical deployment. Success here would not prove
that a real controller can measure or attain the same response.

## Development evidence

Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.pilot.control_feasibility --response --out docs/pilot/results/control_response_feasibility.json
```

The [result JSON](results/control_response_feasibility.json) records all 70
condition/schedule combinations, requested and projected applied actions,
actual heat use, metrics, source/configuration hashes, versions, and costs.
Every schedule restores the same paired initial state for each condition.
The threshold controller uses only the public area-weighted South-band
temperature and a fixed frost-point-plus-2 K threshold. It does not inspect
the matched reference or response. Seasonal and pulse schedules use public
decision time. The ten-pulse search chooses its placement from development
truth outcomes, so it is privileged and is not an eligible policy baseline.

| Controller | Mean RMSE, K | Mean actual heat, W/m²·sol | Area/time within 0.5 K |
| --- | ---: | ---: | ---: |
| No heat | 2.000000 | 0 | 0% |
| Constant 2.5 W/m² | 0.579762 | 300 | 79.292% |
| Fixed seasonal pulse | 3.141474 | 300 | 2.750% |
| Public frost threshold | 2.003251 | 20 | 0.083% |
| Best privileged pulse, intervals 0–2 | 2.777994 | 300 | 2.500% |

Constant heat reduces RMSE by 71.01% versus no heat. It is the strongest
eligible controller in this initial screen. Its average maximum response error is
2 K. This demonstrates substantial response tracking with this heat allowance,
but does not establish full target attainment. The best searched pulse is
379.16% worse than constant heat. This pulse family shows no useful scheduling
advantage. It does not rule out a better smooth schedule.

### Bounded continuous reference

The follow-up searches twelve heat controls with SciPy SLSQP, starting at
constant 2.5 W/m². Bounds are [0,10] per control, with sum at most 30 (300
W/m²·sol). Finite-difference step is 0.05 and the iteration limit is ten.
Each objective evaluation restores the same five paired snapshots. All
requested and actual actions, per-condition metrics, and optimizer status are
in [the search JSON](results/control_response_search.json). Finite-difference
probes may exceed the requested integral constraint; the environment still
projects actual heat to its hard allowance. Best-candidate selection excludes
such probes. This is a privileged feasibility reference selected from truth
outcomes, not an eligible frozen baseline or a learned controller.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.pilot.control_response_search --out docs/pilot/results/control_response_search.json
```

The search used 130 objective candidates and reached its ten-iteration limit.
It did not converge. Best feasible candidate 117 has mean RMSE **0.491604 K**,
15.206% below constant heat. It uses the full 300 W/m²·sol and puts 86.958% of
area/time within 0.5 K. Its maximum error is still 2 K on every condition.
Its twelve requested heat values, in W/m², are approximately:

```text
3.188 2.321 2.312 2.241 2.180 2.136 2.087 1.930 2.323 3.264 3.492 2.526
```

Use the full-precision JSON values to reproduce; rounding changes the budget.
Per-condition RMSEs for seeds 5000–5004 are 0.468992, 0.577946, 0.474730,
0.470102, and 0.466251 K. This demonstrates useful scheduling headroom and
substantial target attainment on these development conditions. It does not
prove a global optimum, robustness on new conditions, or success by an agent.

Because the heat allowance is a cap, two lower constant baselines were also
checked on the same five conditions. Constant 2.0 W/m² uses 240 W/m²·sol and
scores 0.759216 K; constant 2.25 uses 270 W/m²·sol and scores 0.651079 K.
Constant 2.5 remains strongest among all measured eligible baselines, so the
15.206% development gap remains. This finite guard is not a proof of the best
constant rate. The final pilot must choose its baseline on validation only.
The [constant-guard JSON](results/control_response_constants.json) includes
every condition and actual applied heat. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m scripts.pilot.control_response_constants --out docs/pilot/results/control_response_constants.json
```

The revised contract and local feasibility question are now concrete. Keep
policy training unadmitted until data generation, forecast validation, and
leakage checks pass. Do not tune season, budget, target, or controller families
against sealed conditions. Any later final comparison
must freeze its eligible baseline and model recipe using validation, and use
the original pooled 10% improvement gate across five complete training seeds,
with all constraint, reproducibility, and leakage checks.

The screen used 49.6403 wall seconds with one allocated CPU core: 0.013789
core-hours of occupancy. Process CPU use was 48.9594 seconds, or 0.013600
CPU-hours. The bounded continuous search adds 322.7978 wall seconds at one
core (0.089666 core-hours) and 308.0765 process CPU seconds (0.085577
CPU-hours). The lower-constant guard adds 24.9551 wall seconds at one core
(0.006932 core-hours) and 24.4766 process CPU seconds (0.006799 CPU-hours).
GPU use was zero; no Nautilus jobs ran. All response experiments use 0.110387
core-hours of occupancy and 0.105976 CPU-hours of process time. Including the
two original winter screens, simulator-screen occupancy is 0.14951 core-hours.
Test costs
are separate. The focused original and response suite passed 12 tests in
61.86 seconds. The first response test failed on missing implementation, then
passed in 4.36 seconds. The tests use independent full no-heat truth labels
to check daily reward and clock continuity, plus public observation isolation,
actual budget projection, terminal/invalid actions, reset, and paired recovery.
No forecast test data, policy training, or new dataset was used.

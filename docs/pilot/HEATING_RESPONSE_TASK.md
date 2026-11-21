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
eligible controller in this screen. Its average maximum response error is
2 K. This demonstrates substantial response tracking with this heat allowance,
but does not establish full target attainment. The best searched pulse is
379.16% worse than constant heat. No useful scheduling advantage over the
strongest simple controller is demonstrated. The small pulse search does not
prove an optimum or rule out a better smooth schedule.

Keep policy training gated. The next decision must address the missing
scheduling gap before any learner run. Do not tune season, budget, target,
or controller families against sealed conditions. Any later final comparison
must freeze its eligible baseline and model recipe using validation, and use
the original pooled 10% improvement gate across five complete training seeds,
with all constraint, reproducibility, and leakage checks.

The screen used 49.6403 wall seconds with one allocated CPU core: 0.013789
core-hours of occupancy. Process CPU use was 48.9594 seconds, or 0.013600
CPU-hours. GPU use was zero; no Nautilus jobs ran. Including the two original
winter screens, simulator-screen occupancy is 0.05291 core-hours. Test costs
are separate. The focused original and response suite passed 12 tests in
61.86 seconds. The first response test failed on missing implementation, then
passed in 4.36 seconds. The tests use independent full no-heat truth labels
to check daily reward and clock continuity, plus public observation isolation,
actual budget projection, terminal/invalid actions, reset, and paired recovery.
No forecast test data, policy training, or new dataset was used.

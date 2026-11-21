# South-winter heating task v1

This contract implements [issue 5](https://github.com/sameeerkashyap/mars-jepa/issues/5).
The target and heat allowance remain the user-approved values. Agent training
is gated on the feasibility evidence below and forecast validation.

## Physics and season

Use the existing 64-cell truth simulator with dt=0.05 sol, daily output,
D=0.10 and mismatch_scale=0.10. Use the checked-in physics configuration.
There is no new physical calibration. Solar longitude zero is northern spring;
time zero in this simulator is perihelion (Ls=251 degrees). It is southern
summer, so starting a winter episode at time zero would be wrong.

Compute the annual, area-weighted sunlight of the fixed cells at latitude
less than -60 degrees. On an integer-sol astronomical calendar, find the first
descending crossing below 10% of its annual maximum. Start after three full
unheated years plus that day. Require all 120 episode sols to lie below the
same annual threshold. Do not select episode times using temperatures or ice.
The annual reference uses one full year of supplied astronomy, not a fitted
climate outcome. The precise onset and absolute time are in the result JSON.

Draw one storm realization over spin-up plus the complete episode, then keep
it fixed. Its seed identifies the complete initial-condition/disturbance group.
Matched schedules and no-heating controls restore the same pre-action state.
The season clock never resets after spin-up or between actions.

## Public interface

`physics.control.HeatingControl` supplies `reset(seed)`, `step(heat)`,
`observation()`, `metrics()`, `state_dict()` and `load_state_dict(state)`.
`physics.reference.TruthSimulation` supplies a persistent lower-level
`advance(sols, heat, store=True)` boundary. Its fixed total horizon includes
spin-up. New optional `run_truth` arguments `start_step` and `storms` preserve
the old default behavior and all existing callers.

Reset returns the initial pre-action observation. Step accepts a scalar
requested heat in [0,10] W/m². It applies that uniform flux for the next 10
sols. Reject nonfinite or out-of-range requests. Project valid requests down
to the remaining allowance: `applied=min(requested, remaining_budget/10)`.
Debit `10*applied` W/m²·sol. This constant flux also satisfies the cap at each
0.05-sol substep. Use the same projection in imagination; preserve both the
requested and applied heat. The total allowance is 300 W/m²·sol.

| Observation field | Shape and units | Fixed policy scaling |
| --- | --- | --- |
| T_s | 64 current surface temperatures, K | (T_s - frost point)/10 |
| S | 64 current astronomical sunlight values, W/m² | S/589 |
| previous_heat | Last applied heat, W/m²; zero at reset | divide by 10 |
| Ls | Current solar longitude, radians | sin(Ls), cos(Ls) |
| remaining_sols | 120 down to zero | divide by 120 |
| remaining_budget | 300 down to zero, W/m²·sol | divide by 300 |

The vector order is the table order, with sine then cosine. It has 133
components. No normalizer is fitted on evaluation data. Actor values in
[-1,1] map to requested heat as `5*(a+1)` before budget projection.
The public time and allowance are controller bookkeeping. Hidden ground
temperature, ice, dust, flux diagnostics, and future storm realization are
never policy inputs. The physical task remains partially observed; recurrent
belief is needed for hidden thermal and ice memory. Public time and allowance
remove avoidable nonstationarity, not the hidden physical state.

Step returns `observation`, `reward`, `terminated`, `truncated`,
`requested_heat`, `applied_heat`, and ten daily post-transition samples in
`daily_T_s`, `daily_S`, `daily_t_sol`. Daily fields contain only observations.
Terminate after decision 12. This is a finite task, so `terminated=True` and
`truncated=False`; terminal value bootstrap is zero. Reject further steps
until reset. No extra transition is made at the endpoint.

State dictionaries are trusted private checkpoints. They include hidden state,
clock, fixed storms, heat allowance and metric accumulators. They restore exact
continuation and are separate from replay policy observations. They are not an
untrusted file format or a policy input.

## Reward and score

The fixed target is CO₂ frost point plus 2 K. For each daily post-transition
sample, compute `e_d = sum_i w_i*(T_s[d,i]-target)^2 / sum_i w_i`, using all
fixed South-band cells. Weights are the simulator cell areas. No future ice
mask, control-pinned label, or outcome-dependent filtering is used.
The reward of an action interval is `-sum_d e_d` for its ten days. The primary
episode score in K is `sqrt(sum_d e_d /120)`. Thus reward depends only on
observed temperatures and public target, without revealing hidden state.

Report the unweighted mean of episode RMSEs across condition groups, rather
than taking a root after pooling all episode squared errors. Report actual heat
use, maximum absolute South-band error, and area/time fraction within 0.5 K
as diagnostics. The 0.5 K band describes attainment; it is not a change to the
target or the final improvement gate. Report whole-field temperatures for side
effects in later evaluation. No failure or difficult episode may be dropped.

For final confirmation, freeze one eligible simple baseline and one model
recipe using validation only. For each of five training seeds calculate A_s,
the mean matched final episode RMSE. Let B be the baseline mean over the same
episodes. The user gate is pooled `1-mean_s(A_s)/B >= 0.10`, with all five
seeds complete, no constraint violations, and reproducibility/leakage checks
passing. Report every seed and a paired hierarchical bootstrap 95% interval
(10,000 draws, RNG seed 6100, resample training seeds and condition groups).
The gate is not strengthened to require a 10% gain for each individual seed.
If B is zero, the relative gate is undefined. A wide interval limits any scale
claim even when the point gate passes.

## New data boundary

Use a separate `control-v1` namespace. Allocate training conditions 5000–5023,
validation 5100–5107, and a sealed new control test 5200–5219. The local code
and configuration search found no prior allocations in these ranges. The
feasibility screen uses only training conditions 5000–5004. Keep all schedules,
spin-up states and matched controls from one condition in its split. Seeds
9000–9999 remain reserved and the environment rejects them. Existing forecast
test data, metrics, manifests and generators are unchanged and unread.

Generation must store daily observations and ten-sol transition boundaries,
requested/applied actions, heat allowance, terminal flags, reward, seed streams,
task/config/source hashes, and split identity. Keep hidden diagnostics under a
separate hidden namespace. The world-model decision step is ten sols; retain
daily target observations to check its integrated reward. No new dataset or
learner is generated in this ticket.

## Development feasibility screen

`python -m scripts.pilot.control_feasibility --out <path>` evaluates no heat,
constant 2.5 W/m², a fixed midwinter three-interval pulse, an observable
temperature-threshold controller, and ten consecutive three-interval pulse
placements. Each schedule uses the same initial state and disturbances per
condition. All candidates share the cap. Search selects its placement using
development truth outcomes. It is explicitly privileged and is not an eligible
equal-information learned-policy baseline or a proof of global optimality.

See `results/control_feasibility.json` for every condition, schedule and actual
cost. The screen cannot admit training automatically. Resolve lack of target
attainment or lack of a useful scheduling gap with the user before agent runs;
do not silently change the accepted target, allowance, season, or metric.

Measured on five development conditions, onset is day 218 of the year and
absolute start is 2223.8 sols. The target is 149.739944 K.

| Controller | Mean episode RMSE, K | Mean actual heat, W/m²·sol |
| --- | ---: | ---: |
| No heat | 10.839924 | 0 |
| Constant 2.5 | 12.180202 | 300 |
| Fixed seasonal pulse | 12.539672 | 300 |
| Observable threshold | 10.851056 | 20 |
| Best privileged pulse, intervals 9–11 (zero-based) | 11.423109 | 300 |

The best full-budget pulse improves on constant heating by 6.216%, but does
worse than no heating. No heat spends 5.208% of area-time within 0.5 K of the
target; constant heating spends 4.917%. The average maximum deviation is
28.04 K without heat. The early winter surface remains too warm for added heat
to help tracking. These candidates do not demonstrate target attainment or a
useful gain over the strongest simple controller. They do not prove that no
feasible schedule can improve it. Keep agent training gated and request a
user decision on the task before changing season, target or heat allowance.

The local screen used 69.29 wall seconds with one allocated CPU core
(0.01925 core-hours), 51.57 process CPU seconds (0.01432 CPU-hours), zero
GPU-hours, and no Nautilus jobs. These are distinct occupancy and utilization
charges; use the occupancy figure for conservative budget admission. Test
execution is separate from this experiment charge.

An explicitly exploratory follow-up shifts the same conditions by 20 and 40
sols without changing v1. Both full windows remain below the 10% astronomical
winter threshold. The fixed original storm draw is kept exactly, including its
Gaussian tails. No new storm pulses are drawn in the extended integration
horizon. This pairing isolates the start-time effect, but it does not establish
robustness to extra late storms. See `results/control_later_windows.json`.

| Start offset, sols | No heat | Constant | Seasonal | Threshold | Best privileged pulse |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 10.839924 | 12.180202 | 12.539672 | 10.851056 | 11.423109 |
| 20 | 7.386842 | 8.635097 | 9.045555 | 7.484818 | 7.874968 |
| 40 | 4.814655 | 5.912982 | 6.364519 | 5.085986 | 5.274285 |

Entries are mean episode RMSE in K over the same five training conditions.
All ten prescribed pulse placements were checked at each offset; this adds no
new search family. Positive heat still fails to beat no heat. Recommend keeping
the astronomical onset rule fixed for the v1 record; selecting a favorable
start from these outcomes would need a new declared task. Training remains
gated. The user must decide whether to retain this negative result or change
the task. This does not prove a global optimum or task infeasibility.

The follow-up used 71.53 wall seconds at one core (0.01987 core-hours),
62.92 process CPU seconds (0.01748 CPU-hours), and zero GPU-hours. The total
simulator-screen occupancy is 0.03912 core-hours. Reproduce with the same
command plus `--later-windows` and a separate output path.

## Verification boundary

The focused tests compare full and stepped integration bit for bit for hidden
and observed state. They cover clock/disturbance continuity, zero, max,
alternating and constant schedules, exhausted allowance, observable fields,
daily reward/metric consistency, terminal handling, reset isolation, and exact
checkpoint continuation. Existing synthetic physics tests check energy closure,
nonnegative ice, seasonal cycle, zero-mismatch behavior and seeded determinism.
No Nautilus jobs, full learner or old forecast test access are part of this work.

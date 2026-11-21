# Mars heating control

This project studies temperature prediction and heating control in a synthetic Mars climate.

## Language

**Heating action**:
A uniform added heat flux, measured in watts per square metre, that a controller selects for the next action interval.
_Avoid_: Sunlight action

**Heat budget**:
The maximum heating action integrated over an episode. It is distinct from the CPU and GPU allowance used to run experiments.
_Avoid_: Compute budget

**Control episode**:
A fixed period in which a controller observes climate measurements and selects heating actions to track a temperature target.

**Matched no-heating control**:
A simulator trajectory with the same initial conditions and external disturbances as a heated trajectory, with added heating set to zero.

**Intervention response**:
The temperature difference between a heated trajectory and its matched no-heating control.

**Frost point**:
The temperature at which the simulator exchanges energy with its carbon-dioxide ice reservoir.

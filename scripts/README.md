# Scripts and utilities

Run these commands from the repository root after installing the project.

| Command | Purpose |
|---|---|
| `python -m scripts.generate_data` | Generate spatial diffusion fields |
| `python -m scripts.train` | Train spatial PI-JEPA |
| `python -m scripts.pilot.gen_data` | Generate heating trajectories and controls |
| `python -m scripts.pilot.train` | Train one heating model |
| `python -m scripts.pilot.eval` | Evaluate saved pilot runs |
| `python -m scripts.pilot.calibrate` | Check validation curves and uncertainty |
| `python -m scripts.pilot.sweep` | Run the comparison grid |
| `python -m scripts.pilot.analyze` | Summarize results |
| `python -m scripts.pilot.regression_check` | Check the saved physics fixture |

`pilot/` also holds plotting and simulator diagnostics. `utils.py` supplies shared
logging and random-seed helpers. New default pilot reports go to `outputs/pilot/`.

See [the pilot guide](../docs/pilot/RUNME.md) for generation and evaluation.

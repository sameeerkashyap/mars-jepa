# Configuration

All experiment knobs live in typed dataclasses in `marsworld.config` and are
populated from YAML (`configs/*.yaml`) via `load_config`. Only overrides need to
appear in a YAML file; everything else falls back to the dataclass defaults.

## API

::: marsworld.config

"""Budgeted South-pole heating task. All policy-facing fields are observable."""

from copy import deepcopy

import numpy as np

from physics.reference import TruthSimulation
from physics.surface import Grid, frost_point, insolation, load_config, solar_longitude


def project_heat(requested, remaining_budget, interval_sols=10.0):
    """Scalar truth/imagination constraint, in W/m² and W/m²·sol."""
    return min(max(requested, 0.0), 10.0, max(remaining_budget, 0.0) / interval_sols)


def south_winter_start(cfg):
    """First daily descending crossing of 10% annual South-band sunlight."""
    grid = Grid(cfg['grid']['n'])
    band = np.degrees(grid.lat) < -60
    days = np.arange(int(np.ceil(cfg['planet']['year_sols'])))
    sunlight = np.array([np.average(insolation(grid.x, t, cfg['planet'])[band],
                                    weights=grid.w[band]) for t in days])
    winter = sunlight <= 0.10 * sunlight.max()
    crossings = np.flatnonzero(winter & ~np.roll(winter, 1))
    if len(crossings) != 1:
        raise ValueError('expected one South-winter onset per year')
    start = int(crossings[0])
    # The complete task must lie within the fixed winter, without outcome masks.
    for day in range(start, start + 121):
        s = np.average(insolation(grid.x, day, cfg['planet'])[band], weights=grid.w[band])
        if s > 0.10 * sunlight.max():
            raise ValueError('120-sol episode does not fit in South winter')
    return start


class HeatingControl:
    """12 ten-sol decisions, frost+2 K target, 300 W/m²·sol hard allowance.

    The hidden physical state remains partially observed. Public season, time
    left and heat allowance prevent avoidable bookkeeping ambiguity.
    """

    episode_sols = 120
    interval_sols = 10
    heat_budget = 300.0

    def __init__(self, cfg=None):
        self.cfg = deepcopy(load_config() if cfg is None else cfg)
        if self.cfg['grid']['n'] != 64 or self.cfg['time']['output_every_sol'] != 1:
            raise ValueError('task requires 64 cells and daily output')
        dt = self.cfg['time']['dt_sol']
        if dt <= 0 or not np.isclose(1 / dt, round(1 / dt)):
            raise ValueError('simulator timestep must divide one sol')
        self.grid = Grid(64)
        self.band = np.degrees(self.grid.lat) < -60
        co2 = self.cfg['co2']
        self.target = frost_point(co2['surface_pressure_pa'], co2) + 2
        self.start_sol = 3 * self.cfg['planet']['year_sols'] + south_winter_start(self.cfg)
        self.sim = None

    def reset(self, seed):
        if 9000 <= seed <= 9999:
            raise ValueError('seeds 9000–9999 are reserved')
        self.sim = TruthSimulation(self.cfg, seed, self.start_sol + self.episode_sols,
                                   mismatch_scale=0.10)
        self.sim.advance(self.start_sol, store=False)
        self.elapsed = 0
        self.remaining_budget = self.heat_budget
        self.previous_heat = 0.0
        self.squared_error = 0.0
        self.target_samples = 0.0
        self.max_deviation = 0.0
        return self.observation()

    def observation(self):
        if self.sim is None:
            raise RuntimeError('reset the episode first')
        ls, _ = solar_longitude(self.sim.t_sol, self.cfg['planet'])
        return dict(T_s=self.sim.initial['T_init'].copy(),
                    S=insolation(self.grid.x, self.sim.t_sol, self.cfg['planet']),
                    previous_heat=self.previous_heat, Ls=float(ls),
                    remaining_sols=self.episode_sols - self.elapsed,
                    remaining_budget=self.remaining_budget)

    def step(self, requested_heat):
        if self.sim is None or self.elapsed >= self.episode_sols:
            raise RuntimeError('reset before stepping a new or finished episode')
        requested_heat = float(requested_heat)
        if not np.isfinite(requested_heat) or not 0 <= requested_heat <= 10:
            raise ValueError('heat must be finite and in [0, 10] W/m²')
        applied = project_heat(requested_heat, self.remaining_budget, self.interval_sols)
        result = self.sim.advance(self.interval_sols, applied)
        daily = result['T_s'][1:]
        error = self._tracking_error(daily)
        daily_error = np.average(error**2, axis=1, weights=self.grid.w[self.band])
        reward = -float(daily_error.sum())
        self.squared_error -= reward
        self.target_samples += float(np.average(np.abs(error) <= 0.5, axis=1,
                                                 weights=self.grid.w[self.band]).sum())
        self.max_deviation = max(self.max_deviation, float(np.abs(error).max()))
        self.elapsed += self.interval_sols
        self.remaining_budget = max(0.0, self.remaining_budget - applied * self.interval_sols)
        self.previous_heat = applied
        return dict(observation=self.observation(), reward=reward,
                    terminated=self.elapsed == self.episode_sols, truncated=False,
                    requested_heat=requested_heat, applied_heat=applied,
                    daily_T_s=daily.copy(), daily_S=result['S'][1:].copy(),
                    daily_t_sol=result['t_sol'][1:].copy())

    def _tracking_error(self, daily):
        return daily[:, self.band] - self.target

    def metrics(self):
        if self.sim is None or self.elapsed == 0:
            raise RuntimeError('metrics require at least one transition')
        return dict(tracking_rmse_K=float(np.sqrt(self.squared_error / self.elapsed)),
                    heat_used_Wm2_sol=self.heat_budget - self.remaining_budget,
                    within_half_K_fraction=self.target_samples / self.elapsed,
                    max_deviation_K=self.max_deviation)

    def state_dict(self):
        """Trusted private checkpoint, including storms; exclude from replay inputs."""
        return deepcopy(self.__dict__)

    def load_state_dict(self, state):
        self.__dict__ = deepcopy(state)


class ResponseHeatingControl(HeatingControl):
    """Separate +2 K response task; reference truth is a private reward oracle.

    The policy observes only the heated world. The trusted checkpoint includes
    both worlds. This synthetic reward does not establish physical deployability.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.target = 2.0

    def reset(self, seed):
        observation = super().reset(seed)
        self._reference = deepcopy(self.sim)
        return observation

    def _tracking_error(self, daily):
        reference = self._reference.advance(self.interval_sols, 0.0)['T_s'][1:]
        return (daily - reference)[:, self.band] - self.target

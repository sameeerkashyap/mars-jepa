"""Observable climate vectors and exact public constraints for Dreamer imagination."""

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv

from physics.control import ResponseHeatingControl
from physics.surface import frost_point


class ClimateVectorEnv:
    """133 public fields; private paired truth is used only for reward labels."""

    def __init__(self, cfg=None):
        self.task = ResponseHeatingControl(cfg)
        co2 = self.task.cfg["co2"]
        self.frost = frost_point(co2["surface_pressure_pa"], co2)

    def _vector(self, observation):
        return np.concatenate(((observation["T_s"] - self.frost) / 10,
                               observation["S"] / 589,
                               [observation["previous_heat"] / 10,
                                np.sin(observation["Ls"]), np.cos(observation["Ls"]),
                                observation["remaining_sols"] / 120,
                                observation["remaining_budget"] / 300])).astype(np.float32)

    def reset(self, *, seed):
        return self._vector(self.task.reset(seed))

    def observation(self):
        return self._vector(self.task.observation())

    def step(self, raw_action):
        action = torch.as_tensor(raw_action).detach().cpu()
        if action.shape != (1,) or not torch.isfinite(action).all():
            raise ValueError("action must be one finite scalar vector")
        raw = action.item()
        result = self.task.step(5 * (float(np.clip(raw, -1, 1)) + 1))
        info = {key: result[key] for key in ("requested_heat", "applied_heat", "daily_T_s",
                                            "daily_S", "daily_t_sol")}
        info.update(requested_action=raw,
                    dynamics_action=torch.tensor([result["applied_heat"] / 5 - 1], dtype=torch.float32))
        return (self._vector(result["observation"]), result["reward"], result["terminated"],
                result["truncated"], info)

    def state_dict(self):
        """Trusted private simulator checkpoint, never a policy input."""
        return self.task.state_dict()

    def load_state_dict(self, state):
        self.task.load_state_dict(state)
        co2 = self.task.cfg["co2"]
        self.frost = frost_point(co2["surface_pressure_pa"], co2)

    def metrics(self):
        return self.task.metrics()


class ClimateDreamerEnv(DreamerEnv):
    """Learn latent dynamics, but advance public time and heat exactly."""

    def __init__(self, base):
        super().__init__(base.world_model, base.prior_shape, base.belief_shape, device="cpu")
        self.action_spec = base.action_spec.clone()
        self.reward_spec = base.reward_spec.clone()
        self.state_spec = base.state_spec.clone()
        for key in ("remaining_sols", "remaining_budget", "applied_heat"):
            self.state_spec[key] = Unbounded(1)
        self.observation_spec = self.state_spec.clone()
        self.register_buffer("task_constants", torch.tensor([120., 300., 10., 10.]))
        self._start_context = None

    def prepare_imagination(self, sample):
        observation = sample["next", "observation"]
        if observation.shape[-1] != 133 or not torch.isfinite(observation).all():
            raise ValueError("climate imagination needs finite 133-field observations")
        time = observation[..., -2:] * self.task_constants[:2]
        if (time < -1e-4).any() or (time > self.task_constants[:2] + 1e-4).any():
            raise ValueError("public time or heat outside task bounds")
        sols = time[..., :1]
        if not torch.allclose(sols / 10, (sols / 10).round(), atol=1e-5):
            raise ValueError("remaining time must fall on ten-sol boundaries")
        # Match DreamerV3Loss's posterior state.reshape(-1, features).
        self._start_context = (sols.round().clamp(0, 120).reshape(-1, 1).detach().clone(),
                               time[..., 1:].clamp(0, 300).reshape(-1, 1).detach().clone())

    def _reset(self, tensordict=None, **kwargs):
        td = super()._reset(tensordict, **kwargs)
        context, self._start_context = self._start_context, None
        if tensordict is not None:
            if context is None:
                raise ValueError("prepare_imagination must seed posterior starts before reset")
            sols, budget = context
            if sols.shape[:-1] != td.batch_size:
                raise ValueError("posterior start batch does not match prepared context")
        else:
            sols = torch.full((*td.batch_size, 1), 120.)
            budget = torch.full_like(sols, 300.)
        td.set("remaining_sols", sols.clone())
        td.set("remaining_budget", budget.clone())
        td.set("applied_heat", torch.zeros_like(sols))
        return td

    def _step(self, tensordict):
        sols, budget = tensordict["remaining_sols"], tensordict["remaining_budget"]
        active = sols > 0
        request = 5 * (tensordict["action"].clamp(-1, 1) + 1)
        applied = torch.where(active, torch.minimum(request, budget.clamp_min(0) / 10), 0.)
        # The root action remains the raw Normal sample scored by REINFORCE.
        inputs = tensordict.clone(recurse=False)
        inputs.set("action", applied / 5 - 1)
        out = super()._step(inputs)
        for key in ("state", "belief"):
            out.set(key, torch.where(active, out[key], tensordict[key]))
        out.set("remaining_sols", (sols - 10).clamp_min(0))
        out.set("remaining_budget", (budget - 10 * applied).clamp_min(0))
        out.set("applied_heat", applied)
        out.set("reward", torch.where(active, out["reward"], 0.))
        out.set("terminated", out["remaining_sols"] <= 0)
        out.set("done", out["terminated"])
        return out


def climate_imagination(base, continuation):
    """Factory for DreamerLearner construction and trusted checkpoint recovery."""
    env = ClimateDreamerEnv(base)
    masked = TensorDictSequential(
        continuation,
        TensorDictModule(lambda prediction, sols: prediction * (sols > 0),
                         in_keys=["continuation", "remaining_sols"], out_keys=["continuation"]),
    )
    return env, masked

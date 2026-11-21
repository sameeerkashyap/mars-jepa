"""Small synchronous CPU learner using the pinned TorchRL DreamerV3 recipe."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import tensordict
import torch
import torchrl
from tensordict import TensorDict
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.objectives import (DreamerV3ActorLoss, DreamerV3Loss, DreamerV3ModelLoss,
                                DreamerV3ValueLoss)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators
from torchrl.trainers.algorithms import DreamerV3Optimizer

from .networks import build

UPSTREAM_COMMIT = "cc3fe0f02fc599ab367acbd75d1195139c6f7cef"


def versions():
    return dict(python=sys.version.split()[0], torch=str(torch.__version__),
                torchrl=torchrl.__version__, tensordict=tensordict.__version__,
                numpy=np.__version__, device="cpu", precision="float32", compile=False,
                threads=torch.get_num_threads(), deterministic=torch.are_deterministic_algorithms_enabled())


@dataclass(frozen=True)
class LearnerConfig:
    obs_dim: int
    hidden_dim: int = 64
    belief_dim: int = 64
    categoricals: int = 8
    classes: int = 8
    bins: int = 255
    imagination_horizon: int = 15
    sequence_length: int = 8
    batch_size: int = 8
    replay_episodes: int = 1000
    learning_rate: float = 4e-5
    warmup_steps: int = 1000
    continuation_horizon: float = 333.
    lmbda: float = 0.95
    entropy_bonus: float = 3e-4
    free_bits: float = 1.
    slow_critic_tau: float = 0.02

    def __post_init__(self):
        for name in ("obs_dim", "hidden_dim", "belief_dim", "categoricals", "classes",
                     "bins", "imagination_horizon", "sequence_length", "batch_size", "replay_episodes"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least two for shifted replay targets")
        if self.bins < 2 or self.classes < 2:
            raise ValueError("bins and classes must be at least two")
        if not (self.continuation_horizon > 1 and 0 <= self.lmbda <= 1
                and 0 < self.slow_critic_tau <= 1 and self.learning_rate > 0
                and self.warmup_steps >= 0 and self.free_bits >= 0 and self.entropy_bonus >= 0):
            raise ValueError("invalid learner loss or optimizer setting")

    @property
    def state_dim(self):
        return self.categoricals * self.classes


class DreamerLearner:
    """CPU eager training and episode-boundary collection/checkpoint recovery.

    Policy actions are raw scalar Normal samples. An adapter may return the
    applied action for learned dynamics; replay also retains the raw sample.
    Episodes use N+1 observed vectors and N aligned actions/rewards/flags.
    """

    def __init__(self, config, *, imagination_env_factory=None):
        self.config = config
        world, actor, value, imagined_env, continuation = build(config)
        if imagination_env_factory is not None:
            extended, continuation = imagination_env_factory(imagined_env, continuation)
            if not isinstance(extended, DreamerEnv) or extended.world_model is not imagined_env.world_model:
                raise ValueError("imagination factory must retain the shared world model in a DreamerEnv")
            imagined_env = extended
            model_parameters = {id(parameter) for parameter in world.parameters()}
            if any(id(parameter) not in model_parameters
                   for parameter in list(extended.parameters()) + list(continuation.parameters())):
                raise ValueError("imagination adapter must reuse trained world-model parameters")
        self.imagination_env_type = f"{type(imagined_env).__module__}:{type(imagined_env).__qualname__}"
        self.world_model = world
        self.actor = actor
        self.model_loss = DreamerV3ModelLoss(
            world, num_reward_bins=config.bins, free_bits=config.free_bits,
            kl_mode="separate", lambda_dynamic=1., lambda_representation=0.1,
            unimix=0.01, lambda_continue=1., reco_symlog=True,
            continue_target_scale=1 - 1 / config.continuation_horizon,
            global_average=False, detach_output=False,
        )
        self.model_loss.set_keys(pixels="observation", reco_pixels="reco_pixels")
        self.actor_loss = DreamerV3ActorLoss(
            actor, value, imagined_env, continuation_model=continuation,
            imagination_horizon=config.imagination_horizon, use_reinforce=True,
            entropy_bonus=config.entropy_bonus, return_normalization=True,
        )
        self.actor_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=1., lmbda=config.lmbda)
        self.value_loss = DreamerV3ValueLoss(
            value, value_loss="two_hot", num_value_bins=config.bins,
            actor_loss=self.actor_loss, slow_critic_regularization=1.,
        )
        self.target_updater = SoftUpdate(self.value_loss, tau=config.slow_critic_tau)
        self.loss = DreamerV3Loss(self.model_loss, self.actor_loss, self.value_loss,
                                 replay_value_loss_weight=0.3,
                                 continuation_horizon=config.continuation_horizon,
                                 lmbda=config.lmbda)
        # ActorLoss keeps this by reference; register adapter buffers for recovery.
        self.loss.add_module("imagination_continuation", continuation)
        parameters = list(world.parameters()) + list(actor.parameters()) + list(self.value_loss.parameters())
        if len({id(parameter) for parameter in parameters}) != len(parameters) or not all(
            parameter.requires_grad for parameter in parameters
        ):
            raise RuntimeError("optimizer must own unique trainable online parameters")
        self.optimizer = DreamerV3Optimizer(parameters, lr=config.learning_rate,
                                            warmup_steps=config.warmup_steps)
        self.episodes = []
        self.updates = 0
        self.behavior_updates = 0
        self.environment_steps = 0

    def add_episode(self, *, observations, actions, rewards, terminated, truncated,
                    requested_actions=None):
        """Validate and atomically store a complete single-stream episode."""
        n = len(actions)
        values = {name: torch.as_tensor(value).detach().cpu().clone() for name, value in
                  dict(observations=observations, actions=actions, rewards=rewards,
                       terminated=terminated, truncated=truncated,
                       requested_actions=actions if requested_actions is None else requested_actions).items()}
        if n == 0 or values["observations"].shape != (n + 1, self.config.obs_dim):
            raise ValueError("episode needs N+1 observation vectors and N transitions")
        for name in ("actions", "requested_actions", "rewards", "terminated", "truncated"):
            if values[name].shape != (n, 1):
                raise ValueError(f"{name} must have shape (N, 1)")
        for name in ("observations", "actions", "requested_actions", "rewards"):
            values[name] = values[name].float()
            if not torch.isfinite(values[name]).all():
                raise ValueError(f"{name} must be finite")
        for name in ("terminated", "truncated"):
            if values[name].dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        done = values["terminated"] | values["truncated"]
        if done[:-1].any() or not done[-1].item() or (values["terminated"] & values["truncated"]).any():
            raise ValueError("episode must end once with terminated or truncated")
        self.episodes.append(values)
        self.episodes = self.episodes[-self.config.replay_episodes:]
        self.environment_steps += n

    def initial_state(self):
        return {"state": torch.zeros(self.config.state_dim),
                "belief": torch.zeros(self.config.belief_dim), "previous_action": torch.zeros(1)}

    def collect_episode(self, env, *, seed, deterministic=False, store=True):
        """Collect one complete episode through a vector environment adapter.

        ``reset(seed=...)`` returns a vector. ``step(raw_action)`` returns
        ``(next_vector, reward, terminated, truncated[, info])``. Optional
        ``info['dynamics_action']`` is the applied scalar in model action units.
        Replay and recurrent carry use it; requested_actions keeps the raw
        sample. Evaluation uses
        ``deterministic=True, store=False`` and resets carry each episode.
        The adapter owns action clipping and conversion to physical units.
        """
        observation = torch.as_tensor(env.reset(seed=seed), dtype=torch.float32).clone()
        observations, actions, requests, rewards, terminateds, truncateds = [observation], [], [], [], [], []
        carry = None
        while True:
            action, carry = self.act(observation, carry, deterministic=deterministic)
            result = env.step(action.clone())
            next_observation, reward, terminated, truncated = result[:4]
            info = result[4] if len(result) == 5 else {}
            dynamics_action = torch.as_tensor(info.get("dynamics_action", action), dtype=torch.float32).clone()
            if dynamics_action.shape != (1,) or not torch.isfinite(dynamics_action).all():
                raise ValueError("dynamics_action must be one finite scalar vector")
            carry["previous_action"] = dynamics_action
            observation = torch.as_tensor(next_observation, dtype=torch.float32).clone()
            observations.append(observation)
            actions.append(dynamics_action)
            requests.append(action)
            rewards.append([reward])
            terminateds.append([terminated])
            truncateds.append([truncated])
            if terminated or truncated:
                break
        episode = dict(observations=torch.stack(observations), actions=torch.stack(actions),
                       requested_actions=torch.stack(requests),
                       rewards=torch.tensor(rewards, dtype=torch.float32),
                       terminated=torch.tensor(terminateds, dtype=torch.bool),
                       truncated=torch.tensor(truncateds, dtype=torch.bool))
        if store:
            self.add_episode(**episode)
        return episode

    @torch.no_grad()
    def act(self, observation, carry=None, *, deterministic=True):
        """Return raw scalar action and fresh recurrent carry, without changing statistics.

        MODE on TorchRL's actor alone does not disable posterior sampling in
        v0.14.0. The pinned posterior logits path avoids every random draw here.
        """
        observation = torch.as_tensor(observation, dtype=torch.float32)
        if observation.shape != (self.config.obs_dim,) or not torch.isfinite(observation).all():
            raise ValueError("observation must be one finite vector of configured size")
        carry = self.initial_state() if carry is None else carry
        prior = self.world_model[1].rssm_prior.module
        posterior = self.world_model[1].rssm_posterior.module
        belief = prior._update_belief(carry["state"], carry["belief"], carry["previous_action"])
        encoded = self.world_model[0](TensorDict({"next": {"observation": observation}}, []))
        embedding = encoded["next", "encoded_latents"]
        if deterministic:
            logits = posterior._logits(belief, embedding)
            state = torch.nn.functional.one_hot(logits.argmax(-1), self.config.classes).float().flatten(-2)
        else:
            _, state = posterior(belief, embedding)
        inputs = TensorDict({"state": state, "belief": belief}, [])
        distribution = self.actor.get_dist(inputs)
        action = distribution.mode if deterministic else distribution.sample()
        return action.clone(), {"state": state, "belief": belief, "previous_action": action.clone()}

    @torch.no_grad()
    def sample(self):
        """Sample contiguous chunks; rebuild their starting context from the episode prefix.

        Every record predicts its own observation from the preceding action.
        Record zero is the reset observation with zero action/reward. No chunk
        crosses an episode or stream. Prefix refresh uses categorical modes.
        """
        length = self.config.sequence_length
        eligible = [episode for episode in self.episodes if len(episode["actions"]) + 1 >= length]
        if not eligible:
            raise ValueError("replay has no episode long enough for sequence_length")
        chunks = []
        for _ in range(self.config.batch_size):
            episode = eligible[torch.randint(len(eligible), ()).item()]
            count = len(episode["observations"])
            start = torch.randint(count - length + 1, ()).item()
            carry = None
            for t in range(start):
                _, carry = self.act(episode["observations"][t], carry)
                carry["previous_action"] = episode["actions"][t]
            carry = self.initial_state() if carry is None else carry
            # carry is the posterior of observation start-1. The next record
            # consumes the action leading from start-1 into observation start.
            action = torch.cat([torch.zeros(1, 1), episode["actions"]])[start:start + length]
            reward = torch.cat([torch.zeros(1, 1), episode["rewards"]])[start:start + length]
            terminated = torch.cat([torch.zeros(1, 1, dtype=torch.bool), episode["terminated"]])[start:start + length]
            truncated = torch.cat([torch.zeros(1, 1, dtype=torch.bool), episode["truncated"]])[start:start + length]
            chunks.append(TensorDict({
                "state": carry["state"].expand(length, -1).clone(),
                "belief": carry["belief"].expand(length, -1).clone(),
                "action": action,
                "requested_action": torch.cat([torch.zeros(1, 1), episode["requested_actions"]])[start:start + length],
                "is_init": (torch.arange(start, start + length) == 0).unsqueeze(-1),
                "next": {"observation": episode["observations"][start:start + length],
                         "reward": reward, "terminated": terminated, "truncated": truncated,
                         "done": terminated | truncated},
            }, [length]))
        return torch.stack(chunks)

    def update(self, *, model_only=False):
        """Update model and behavior, or only the model for the forecast readiness gate."""
        self.optimizer.zero_grad(set_to_none=True)
        sample = self.sample()
        if model_only:
            losses, _ = self.model_loss(sample)
        else:
            imagined_env = self.actor_loss.model_based_env
            if hasattr(imagined_env, "prepare_imagination"):
                imagined_env.prepare_imagination(sample)
            losses = self.loss(sample)
        total = sum(value for key, value in losses.items() if key.startswith("loss_"))
        if not torch.isfinite(total):
            raise FloatingPointError("non-finite Dreamer loss")
        total.backward()
        self.optimizer.step()
        if not model_only:
            self.target_updater.step()
            self.behavior_updates += 1
        self.updates += 1
        return {key: value.detach().item() for key, value in losses.items()
                if torch.is_tensor(value) and value.numel() == 1}

    def save(self, path):
        """Trusted local checkpoint at an episode boundary, including replay and RNG."""
        state = dict(format_version=1, config=asdict(self.config), source=UPSTREAM_COMMIT,
                     imagination_env_type=self.imagination_env_type,
                     versions=versions(), loss=self.loss.state_dict(),
                     optimizer=self.optimizer.state_dict(), episodes=self.episodes,
                     updates=self.updates, behavior_updates=self.behavior_updates,
                     environment_steps=self.environment_steps,
                     torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                     python_rng=random.getstate())
        path = Path(path)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
                temporary = Path(file.name)
                torch.save(state, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, imagination_env_factory=None):
        """Load only trusted local checkpoints; pickle contains RNG and replay state."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["format_version"] != 1 or state["source"] != UPSTREAM_COMMIT:
            raise ValueError("unsupported learner checkpoint/source")
        if state["versions"] != versions():
            raise ValueError("checkpoint recovery requires the same dependency versions and CPU stack")
        learner = cls(LearnerConfig(**state["config"]), imagination_env_factory=imagination_env_factory)
        if learner.imagination_env_type != state["imagination_env_type"]:
            raise ValueError("checkpoint requires the same imagination environment factory")
        learner.loss.load_state_dict(state["loss"])
        learner.optimizer.load_state_dict(state["optimizer"])
        learner.episodes = state["episodes"]
        learner.updates = state["updates"]
        learner.behavior_updates = state["behavior_updates"]
        learner.environment_steps = state["environment_steps"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        return learner

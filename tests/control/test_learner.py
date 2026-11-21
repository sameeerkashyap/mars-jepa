import torch
import pytest

pytest.importorskip("torchrl", minversion="0.14.0")

from tensordict import TensorDict  # noqa: E402
from torchrl.objectives import categorical_kl_terms  # noqa: E402
from torchrl.modules import SymExpTwoHot  # noqa: E402
from torchrl.objectives import symexp  # noqa: E402

from benchmarks.dreamer_control import DreamerLearner, LearnerConfig  # noqa: E402
from torchrl.envs.model_based.dreamer import DreamerEnv  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic_cpu_check():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.fixture
def learner():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    return DreamerLearner(LearnerConfig(obs_dim=3, hidden_dim=8, belief_dim=8,
                                      categoricals=2, classes=2, bins=16,
                                      imagination_horizon=3, batch_size=2, sequence_length=3,
                                      warmup_steps=0))


def episode(offset=0):
    return dict(
        observations=torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10 + offset,
        actions=torch.arange(4, dtype=torch.float32).reshape(4, 1) / 10,
        rewards=torch.tensor([[1.], [2.], [3.], [4.]]),
        terminated=torch.tensor([[False], [False], [False], [True]]),
        truncated=torch.zeros(4, 1, dtype=torch.bool),
    )


def test_replay_configuration_requires_at_least_two_records():
    with pytest.raises(ValueError, match="sequence_length must be at least two"):
        LearnerConfig(obs_dim=3, sequence_length=1)
    assert LearnerConfig(obs_dim=3, sequence_length=2).sequence_length == 2


def test_vector_episode_replay_update_and_recovery(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    learner = DreamerLearner(LearnerConfig(obs_dim=3, hidden_dim=8, belief_dim=8,
                                         categoricals=2, classes=2, bins=16,
                                         imagination_horizon=3, batch_size=2, sequence_length=3,
                                         warmup_steps=4))
    learner.add_episode(**episode())
    learner.add_episode(**episode(10))
    metrics = learner.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert learner.updates == 1
    learner.update()
    before = {k: v.clone() for k, v in learner.loss.state_dict().items()}
    checkpoint = tmp_path / "learner.pt"
    learner.save(checkpoint)
    expected = learner.update()
    expected_weights = {k: v.clone() for k, v in learner.loss.state_dict().items()}
    restored = DreamerLearner.load(checkpoint)
    assert restored.update() == expected
    assert restored.updates == 3
    assert restored.environment_steps == 8 and restored.behavior_updates == 3
    assert any(not torch.equal(v, before[k]) for k, v in expected_weights.items())
    for name, value in restored.loss.state_dict().items():
        assert torch.equal(value, expected_weights[name]), name


class VectorEnv:
    def reset(self, *, seed):
        self.step_count = 0
        self.state = torch.tensor([0., float(seed), 1.])
        return self.state.clone()

    def step(self, action):
        assert action.shape == (1,)
        self.step_count += 1
        self.state = self.state + action.clamp(-1, 1)
        return (self.state.clone(), -self.state.square().sum().item(),
                self.step_count == 4, False)


def test_collect_deterministic_posterior_and_actor_without_rng_or_state_changes(learner):
    before = {k: v.clone() for k, v in learner.loss.state_dict().items()}
    rng = torch.get_rng_state().clone()
    first = learner.collect_episode(VectorEnv(), seed=2, deterministic=True, store=False)
    assert torch.equal(torch.get_rng_state(), rng)
    assert learner.episodes == [] and learner.environment_steps == 0
    second = learner.collect_episode(VectorEnv(), seed=2, deterministic=True, store=False)
    for key in first:
        assert torch.equal(first[key], second[key])
    for name, value in learner.loss.state_dict().items():
        assert torch.equal(value, before[name]), name
    learner.collect_episode(VectorEnv(), seed=2)
    assert learner.environment_steps == 4
    assert len(learner.episodes) == 1
    assert torch.isfinite(torch.tensor(learner.update()["loss_actor"]))


def test_projected_applied_action_reaches_replay_and_policy_carry(learner):
    class ProjectedEnv(VectorEnv):
        def step(self, raw_action):
            # Budget is empty. Applied normalized heat is always -1 (zero heat).
            observation, reward, terminated, truncated = super().step(torch.tensor([-1.]))
            return observation, reward, terminated, truncated, {"dynamics_action": torch.tensor([-1.])}

    data = learner.collect_episode(ProjectedEnv(), seed=2, deterministic=True)
    assert torch.equal(data["actions"], torch.full((4, 1), -1.))
    carry = None
    for t, observation in enumerate(data["observations"][:-1]):
        action, carry = learner.act(observation, carry)
        assert torch.equal(action, data["requested_actions"][t])
        carry["previous_action"] = data["actions"][t]
    sample = learner.sample()
    assert (sample["action"][~sample["is_init"]] == -1.).all()
    assert (sample["requested_action"][~sample["is_init"]] != -1.).all()


def test_imagination_adapter_prepares_context_and_recovers(learner, tmp_path):
    class ContextEnv(DreamerEnv):
        def prepare_imagination(self, sample):
            assert sample.batch_size == (2, 3)
            self.preparations += 1

    def factory(env, continuation):
        extended = ContextEnv(env.world_model, env.prior_shape, env.belief_shape)
        extended.action_spec = env.action_spec.clone()
        extended.reward_spec = env.reward_spec.clone()
        extended.state_spec = env.state_spec.clone()
        extended.observation_spec = env.observation_spec.clone()
        extended.register_buffer("preparations", torch.zeros(()))
        return extended, continuation

    adapted = DreamerLearner(learner.config, imagination_env_factory=factory)
    adapted.add_episode(**episode())
    adapted.update()
    assert adapted.actor_loss.model_based_env.preparations == 1
    checkpoint = tmp_path / "adapted.pt"
    adapted.save(checkpoint)
    with pytest.raises(ValueError, match="same imagination"):
        DreamerLearner.load(checkpoint)
    restored = DreamerLearner.load(checkpoint, imagination_env_factory=factory)
    assert restored.actor_loss.model_based_env.preparations == 1
    restored.update()
    assert restored.actor_loss.model_based_env.preparations == 2


def has_gradient(parameters):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


def test_loss_gradient_ownership_and_slow_target_update(learner):
    optimized = [parameter for group in learner.optimizer.param_groups for parameter in group["params"]]
    assert len(optimized) == len({id(parameter) for parameter in optimized})
    assert all(parameter.requires_grad for parameter in optimized)
    expected = (list(learner.world_model.parameters()) + list(learner.actor.parameters()) +
                [value for value in learner.value_loss.value_model_params.values(True, True)
                 if value.requires_grad])
    assert {id(parameter) for parameter in optimized} == {id(parameter) for parameter in expected}
    learner.add_episode(**episode())
    sample = learner.sample()
    model_losses, posterior = learner.model_loss(sample)
    sum(model_losses.values()).backward()
    assert has_gradient(learner.world_model[0].parameters())
    assert has_gradient(learner.world_model[1].rssm_prior.parameters())
    assert has_gradient(learner.world_model[1].rssm_posterior.parameters())
    for head in learner.world_model[2:]:
        assert has_gradient(head.parameters())
    assert not has_gradient(learner.actor.parameters())
    assert not has_gradient(learner.value_loss.parameters())

    learner.optimizer.zero_grad(set_to_none=True)
    features = posterior["next"].select("state", "belief").detach().reshape(-1)
    features["state"].requires_grad_()
    actor_losses, imagined = learner.actor_loss(features)
    actor_losses["loss_actor"].backward()
    assert has_gradient(learner.actor.parameters())
    assert not has_gradient(learner.world_model.parameters())
    assert not has_gradient(learner.value_loss.parameters())
    assert features["state"].grad is None
    assert not imagined["discount_weight"].requires_grad

    learner.optimizer.zero_grad(set_to_none=True)
    # Detached critic targets must not train either the actor or representation.
    imagined["lambda_target"] = imagined["lambda_target"].detach().requires_grad_()
    value_losses, _ = learner.value_loss(imagined.detach())
    value_losses["loss_value"].backward()
    assert has_gradient(learner.value_loss.parameters())
    assert not has_gradient(learner.actor.parameters())
    assert not has_gradient(learner.world_model.parameters())
    assert imagined["lambda_target"].grad is None
    targets = dict(learner.value_loss.target_value_model_params.items(True, True))
    online = {name: value for name, value in learner.value_loss.value_model_params.items(True, True)
              if value.requires_grad}
    before = {name: value.clone() for name, value in targets.items()}
    with torch.no_grad():
        for value in online.values():
            value.add_(1.)
    learner.target_updater.step()
    for name, value in targets.items():
        assert not value.requires_grad and value.grad is None
        expected = before[name] + learner.config.slow_critic_tau if name in online else before[name]
        torch.testing.assert_close(value, expected)


def test_lambda_return_arithmetic_and_terminal_bootstrap(learner):
    learner.actor_loss.make_value_estimator(gamma=1., lmbda=0.5)
    reward = torch.tensor([[[1.], [2.], [3.]]])
    value = torch.tensor([[[10.], [20.], [30.]]])
    continuation = torch.full_like(reward, 0.5)
    torch.testing.assert_close(learner.actor_loss.lambda_target(reward, value, continuation),
                               torch.tensor([[[6.375], [11.5], [18.]]]))
    continuation[:, -1] = 0.
    torch.testing.assert_close(learner.actor_loss.lambda_target(reward, value, continuation),
                               torch.tensor([[[5.4375], [7.75], [3.]]]))
    value[:, -1] = 1e8
    assert learner.actor_loss.lambda_target(reward, value, continuation)[0, -1, 0] == 3.

    # Replay terminal labels, not collector truncation labels, stop bootstrap.
    replay = TensorDict({
        "state": torch.zeros(1, 2, learner.config.state_dim),
        "belief": torch.zeros(1, 2, learner.config.belief_dim),
        "bootstrap": torch.zeros(1, 2),
        "next": {"reward": torch.tensor([[[0.], [3.]]]),
                 "done": torch.tensor([[[False], [True]]]),
                 "terminated": torch.tensor([[[False], [True]]])},
    }, [1, 2])
    terminal_loss = learner.value_loss.replay_value_loss(replay)["loss_replay_value"]
    replay["bootstrap"][:, -1] = 100.
    assert torch.equal(terminal_loss, learner.value_loss.replay_value_loss(replay)["loss_replay_value"])
    # Use nonzero logits: zero logits give the same cross entropy for every target.
    with torch.no_grad():
        for parameter in learner.value_loss.value_model_params.values(True, True):
            if parameter.requires_grad:
                parameter.add_(torch.linspace(-0.2, 0.2, parameter.numel()).reshape(parameter.shape))
    replay["next", "terminated"][:] = False
    truncated_loss = learner.value_loss.replay_value_loss(replay)["loss_replay_value"]
    replay["bootstrap"][:, -1] = 0.
    assert not torch.equal(truncated_loss, learner.value_loss.replay_value_loss(replay)["loss_replay_value"])


def test_replay_keeps_episode_identity_alignment_and_refreshes_context(learner):
    learner.add_episode(**episode())
    learner.add_episode(**episode(100))
    for _ in range(10):
        sample = learner.sample()
        observations = sample["next", "observation"]
        assert ((observations[:, 1:, 0] - observations[:, :-1, 0] - 0.3).abs() < 1e-4).all()
        assert (observations[:, :, 0] < 10).all(-1).logical_or((observations[:, :, 0] > 90).all(-1)).all()
        assert not sample["is_init"][:, 1:].any()
        assert not sample["next", "done"][:, :-1].any()
        indices = (observations[..., 0].remainder(100) / 0.3).round().long()
        torch.testing.assert_close(sample["next", "reward"].squeeze(-1), indices.float())
        torch.testing.assert_close(sample["action"].squeeze(-1), (indices - 1).clamp_min(0).float() / 10)
        for sequence in sample:
            reset = bool(sequence["is_init"][0])
            if reset:
                assert sequence["state"][0].count_nonzero() == 0
                assert sequence["belief"][0].count_nonzero() == 0
                assert sequence["action"][0] == 0
            else:
                assert sequence["state"][0].sum() == learner.config.categoricals
    # Replay is immutable and refuses resets inside a supplied episode.
    bad = episode()
    bad["terminated"][1] = True
    with pytest.raises(ValueError, match="end once"):
        learner.add_episode(**bad)
    assert len(learner.episodes) == 2


def test_pinned_split_kl_golden_and_stopped_gradients():
    # Golden from TorchRL v0.14.0 test_dreamer_v3_reference_kl_fixture_and_gradients.
    posterior = torch.tensor([[[2., -1., 0.5], [-0.5, 1.5, 0.]]], requires_grad=True)
    prior = torch.tensor([[[0., 1., -1.], [1., -0.5, 0.5]]], requires_grad=True)
    dynamics, representation = categorical_kl_terms(posterior, prior, free_nats=0., unimix=0.01)
    assert dynamics.item() == pytest.approx(1.9163513, abs=1e-6)
    assert representation.item() == pytest.approx(1.9163513, abs=1e-6)
    dynamics.backward(retain_graph=True)
    assert posterior.grad is None and prior.grad.norm() > 0
    prior.grad = None
    representation.backward()
    assert prior.grad is None and posterior.grad.norm() > 0
    dynamic, representation = categorical_kl_terms(prior, prior, free_nats=1., unimix=0.01)
    assert dynamic.item() == 1. and representation.item() == 1.


def test_model_readiness_update_leaves_actor_critics_and_return_stats_unchanged(learner):
    learner.add_episode(**episode())
    actor = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    critic = {k: v.clone() for k, v in learner.value_loss.state_dict().items()}
    returns = {k: v.clone() for k, v in learner.actor_loss.retnorm.state_dict().items()}
    model = {k: v.clone() for k, v in learner.world_model.state_dict().items()}
    metrics = learner.update(model_only=True)
    assert "loss_actor" not in metrics
    assert learner.updates == 1 and learner.behavior_updates == 0
    assert any(not torch.equal(v, model[k]) for k, v in learner.world_model.state_dict().items())
    for module, before in ((learner.actor, actor), (learner.value_loss, critic),
                           (learner.actor_loss.retnorm, returns)):
        for name, value in module.state_dict().items():
            assert torch.equal(value, before[name]), name


def test_two_hot_decoder_zero_midpoint_and_support_boundaries():
    decoder = SymExpTwoHot(5)
    assert decoder(torch.zeros(1, 5)).item() == 0.
    for index in (0, 4):
        logits = torch.full((1, 5), -torch.inf)
        logits[0, index] = 0.
        torch.testing.assert_close(decoder(logits).squeeze(), decoder.bins[index])
    logits = torch.full((1, 5), -torch.inf)
    logits[0, 2:4] = 0.
    # Decode is the mean on the symexp support, not symexp of the mean symlog bin.
    torch.testing.assert_close(decoder(logits).squeeze(), symexp(torch.tensor(10.)) / 2)

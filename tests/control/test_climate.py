import numpy as np
import pytest
import torch

pytest.importorskip("torchrl", minversion="0.14.0")

from benchmarks.dreamer_control.climate import ClimateVectorEnv, climate_imagination  # noqa: E402
from benchmarks.dreamer_control import DreamerLearner, LearnerConfig  # noqa: E402
from tensordict import TensorDict  # noqa: E402


def test_public_vector_and_real_heat_projection():
    env = ClimateVectorEnv()
    vector = env.reset(seed=5000)
    assert vector.shape == (133,) and vector.dtype == np.float32
    assert vector[-2:].tolist() == [1., 1.]
    assert vector[128] == 0.
    before = env.state_dict()
    with pytest.raises(ValueError, match="finite scalar"):
        env.step(torch.tensor([float("nan")]))
    assert env.observation().tolist() == vector.tolist()
    for i in range(12):
        vector, reward, done, truncated, info = env.step(torch.tensor([3.]))
        assert info["requested_action"] == 3.
        assert info["requested_heat"] == 10.
        assert info["applied_heat"] == (10. if i < 3 else 0.)
        assert info["dynamics_action"].item() == (1. if i < 3 else -1.)
        assert set(info) == {"requested_action", "requested_heat", "applied_heat", "dynamics_action",
                             "daily_T_s", "daily_S", "daily_t_sol"}
        assert reward <= 0 and not truncated
        assert done == (i == 11)
    assert vector[-2:].tolist() == [0., 0.]
    with pytest.raises(RuntimeError):
        env.step(torch.tensor([-1.]))
    env.load_state_dict(before)
    restored = env.step(torch.tensor([-4.]))
    assert restored[4]["applied_heat"] == 0.
    assert restored[4]["dynamics_action"].item() == -1.
    assert restored[1] == -40.


def tiny_learner():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    return DreamerLearner(LearnerConfig(obs_dim=133, hidden_dim=8, belief_dim=8,
                                       categoricals=2, classes=2, bins=16,
                                       imagination_horizon=4, batch_size=2, sequence_length=3,
                                       warmup_steps=0), imagination_env_factory=climate_imagination)


def test_imagined_projection_time_alignment_and_no_refill():
    learner = tiny_learner()
    learner.add_episode(**fixture_episode())
    learner.update(model_only=True)
    learner.update(model_only=True)
    env = learner.actor_loss.model_based_env
    observation = torch.zeros(2, 2, 133)
    observation[..., -2] = torch.tensor([[20., 0.], [10., 30.]]) / 120
    observation[..., -1] = torch.tensor([[25., 100.], [0., 300.]]) / 300
    sample = TensorDict({"next": {"observation": observation}}, [2, 2])
    env.prepare_imagination(sample)
    start = TensorDict({"state": torch.zeros(4, 4), "belief": torch.zeros(4, 8)}, [4])
    state = env.reset(start)
    assert state["remaining_sols"].flatten().tolist() == [20., 0., 10., 30.]
    torch.testing.assert_close(state["remaining_budget"].flatten(), torch.tensor([25., 100., 0., 300.]))
    for step in range(5):
        state["action"] = torch.full((4, 1), 3.)
        rng = torch.get_rng_state()
        transition = env.step(state)
        assert transition["action"].flatten().tolist() == [3.] * 4
        next_state = transition["next"]
        expected_heat = [[2.5, 0., 0., 10.], [0., 0., 0., 10.],
                         [0., 0., 0., 10.], [0., 0., 0., 0.], [0., 0., 0., 0.]][step]
        torch.testing.assert_close(next_state["applied_heat"].flatten(), torch.tensor(expected_heat))
        torch.set_rng_state(rng)
        oracle_input = state.clone()
        oracle_input["action"] = torch.tensor(expected_heat).reshape(4, 1) / 5 - 1
        expected_latents = env.world_model(oracle_input)
        active = state["remaining_sols"].squeeze(-1) > 0
        for key in ("state", "belief"):
            assert next_state[key][active].equal(expected_latents[key][active])
            assert next_state[key][~active].equal(state[key][~active])
        assert (next_state["remaining_budget"] <= state["remaining_budget"]).all()
        assert (next_state["remaining_sols"] <= state["remaining_sols"]).all()
        terminal = state["remaining_sols"] <= 0
        if (~terminal).any():
            assert (next_state["reward"][~terminal] != 0).any()
        assert (next_state["reward"][terminal] == 0).all()
        continuation = learner.loss.imagination_continuation(next_state.clone())["continuation"]
        assert (continuation[next_state["remaining_sols"] <= 0] == 0).all()
        assert next_state["terminated"].equal(next_state["remaining_sols"] <= 0)
        state = next_state.exclude("reward", "terminated", "done")
    fresh = env.reset()
    assert fresh["remaining_sols"].item() == 120
    assert fresh["remaining_budget"].item() == 300


def test_imagination_keeps_raw_scores_and_actor_gradient_ownership():
    learner = tiny_learner()
    observations = torch.zeros(2, 3, 133)
    observations[..., -2] = torch.tensor([[0., 10., 20.], [30., 40., 50.]]) / 120
    observations[..., -1] = 0.
    env = learner.actor_loss.model_based_env
    env.prepare_imagination(TensorDict({"next": {"observation": observations}}, [2, 3]))
    features = TensorDict({"state": torch.zeros(6, 4, requires_grad=True),
                           "belief": torch.zeros(6, 8)}, [6])
    losses, imagined = learner.actor_loss(features)
    assert "remaining_sols" in learner.loss.imagination_continuation.in_keys
    assert (imagined["next", "applied_heat"] == 0).all()
    assert (imagined["action"] != -1).any()
    assert (imagined["discount_weight"][0] == 0).all()
    losses["loss_actor"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.actor.parameters())
    assert all(p.grad is None for p in learner.world_model.parameters())
    assert all(p.grad is None for p in learner.value_loss.parameters())
    assert features["state"].grad is None


def test_physical_collect_replay_model_update_and_exact_recovery(tmp_path):
    learner = tiny_learner()
    env = ClimateVectorEnv()
    episode = learner.collect_episode(env, seed=5000, deterministic=True)
    assert episode["observations"].shape == (13, 133)
    assert episode["observations"][-1, -2] == 0
    torch.testing.assert_close(episode["actions"].flatten(), episode["observations"][1:, 128] * 2 - 1)
    assert torch.allclose(episode["rewards"].sum(), torch.tensor(-120 * env.metrics()["tracking_rmse_K"]**2))
    actor = {k: v.clone() for k, v in learner.actor.state_dict().items()}
    critic = {k: v.clone() for k, v in learner.value_loss.state_dict().items()}
    statistics = {k: v.clone() for k, v in learner.actor_loss.retnorm.state_dict().items()}
    for _ in range(2):
        assert all(np.isfinite(v) for v in learner.update(model_only=True).values())
    assert learner.behavior_updates == 0
    for module, before in ((learner.actor, actor), (learner.value_loss, critic),
                           (learner.actor_loss.retnorm, statistics)):
        for key, value in module.state_dict().items():
            assert value.equal(before[key])
    learner.save(tmp_path / "climate.pt")
    expected = learner.update(model_only=True)
    weights = {k: v.clone() for k, v in learner.loss.state_dict().items()}
    restored = DreamerLearner.load(tmp_path / "climate.pt", imagination_env_factory=climate_imagination)
    assert restored.update(model_only=True) == expected
    assert all(v.equal(weights[k]) for k, v in restored.loss.state_dict().items())
    rng = torch.get_rng_state()
    first = learner.act(episode["observations"][0])[0]
    second = restored.act(episode["observations"][0])[0]
    assert first.equal(second) and torch.get_rng_state().equal(rng)
    assert restored.actor_loss.model_based_env.task_constants.tolist() == [120., 300., 10., 10.]
    expected_episode = learner.collect_episode(env, seed=5000, deterministic=True, store=False)
    replay_size = len(restored.episodes)
    rng = torch.get_rng_state()
    actual_episode = restored.collect_episode(env, seed=5000, deterministic=True, store=False)
    assert torch.get_rng_state().equal(rng) and len(restored.episodes) == replay_size
    assert all(value.equal(actual_episode[key]) for key, value in expected_episode.items())


def fixture_episode():
    observations = torch.zeros(4, 133)
    observations[:, -2] = torch.tensor([30., 20., 10., 0.]) / 120
    observations[:, -1] = 10 / 300
    return dict(observations=observations, actions=torch.full((3, 1), -1.),
                rewards=torch.full((3, 1), -1.),
                terminated=torch.tensor([[False], [False], [True]]),
                truncated=torch.zeros(3, 1, dtype=torch.bool))


def test_full_fixture_loss_and_adapter_checkpoint_resume(tmp_path):
    learner = tiny_learner()
    learner.add_episode(**fixture_episode())
    assert np.isfinite(learner.update()["loss_actor"])
    learner.save(tmp_path / "fixture.pt")
    expected = learner.update()
    weights = {k: v.clone() for k, v in learner.loss.state_dict().items()}
    with pytest.raises(ValueError, match="same imagination"):
        DreamerLearner.load(tmp_path / "fixture.pt")
    restored = DreamerLearner.load(tmp_path / "fixture.pt", imagination_env_factory=climate_imagination)
    assert restored.update() == expected
    assert all(v.equal(weights[k]) for k, v in restored.loss.state_dict().items())

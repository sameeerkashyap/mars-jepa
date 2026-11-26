import numpy as np
import pytest
import torch


def test_pair_public_training_temperatures_at_each_action_boundary():
    from scripts.pilot.dreamer_paired_response import matched_training_pairs

    records, episodes = [], []
    for seed, schedule, heat in ((5000, 'constant_2p5', -.5), (5001, 'zero', -1.),
                                 (5000, 'zero', -1.), (5001, 'random_pulse', 1.)):
        observations = np.full((13, 133), seed - 5000, np.float32)
        observations[1:, 2] = np.arange(1, 13) + (heat + 1) * np.arange(1, 13)
        records.append(dict(condition_seed=seed, schedule=schedule))
        episodes.append(dict(policy=dict(observations=observations,
                                          actions=np.full((12, 1), heat, np.float32))))
    manifest = dict(split='train', episodes=records)
    pairs = matched_training_pairs(manifest, episodes, np.array([2]))
    assert len(pairs) == 2
    for (reset, actions, response), seed, first_heat, first_response in zip(
            pairs, [5000, 5001], [-.5, 1.], [5., 20.]):
        assert reset[0] == seed - 5000 and (actions == first_heat).all()
        torch.testing.assert_close(response[:, 0], first_response * torch.arange(1, 13))
    with pytest.raises(ValueError, match='train'):
        matched_training_pairs(dict(manifest, split='validation'), episodes, np.array([2]))
    episodes[0]['policy']['observations'][0, 0] = 99.
    with pytest.raises(ValueError, match='reset'):
        matched_training_pairs(manifest, episodes, np.array([2]))


def test_paired_loss_matches_open_loop_forecasts_and_trains_prior_and_decoder():
    from benchmarks.dreamer_control import DreamerLearner, LearnerConfig
    from scripts.pilot.dreamer_control_forecast import prior_forecast
    from scripts.pilot.dreamer_paired_response import paired_response_loss

    torch.set_num_threads(1)
    torch.manual_seed(71)
    learner = DreamerLearner(LearnerConfig(obs_dim=133, hidden_dim=16, belief_dim=16,
                                          categoricals=4, classes=4))
    reset = torch.linspace(-1, 1, 133)
    actions = torch.zeros(12, 1)
    actions[5:] = -1.
    target = torch.arange(24, dtype=torch.float32).reshape(12, 2) / 10
    band, weights = [0, 2], torch.tensor([1., 3.])
    prediction = prior_forecast(learner, reset, actions)['observations']
    control = prior_forecast(learner, reset, torch.full_like(actions, -1))['observations']
    residual = 10 * (prediction[:, band] - control[:, band]) - target.numpy()
    expected = np.average(residual**2, axis=1, weights=weights.numpy()).mean()
    rng = torch.get_rng_state().clone()
    loss = paired_response_loss(learner, reset, actions, target, band, weights)
    assert loss.item() == pytest.approx(expected, rel=1e-4)
    assert torch.equal(rng, torch.get_rng_state())
    loss.backward()
    prior = learner.world_model[1].rssm_prior.module
    for module in (prior.rnn, prior.rnn_to_prior_projector, learner.world_model[2]):
        assert any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in module.parameters())
    assert all(p.grad is None for p in learner.actor.parameters())

"""Prior-only validation. Future truth is a score label, never a model input."""
import numpy as np
import torch
from tensordict import TensorDict


FORECAST_METRICS = ('model_response', 'zero_response', 'linear_response',
                    'model_temperature', 'persistence_temperature',
                    'model_reward', 'zero_reward', 'linear_reward')


@torch.no_grad()
def prior_forecast(learner, reset_observation, applied_actions):
    """Condition once on reset, then advance deterministic prior modes only."""
    reset = torch.as_tensor(reset_observation, dtype=torch.float32)
    actions = torch.as_tensor(applied_actions, dtype=torch.float32)
    if reset.shape != (133,) or actions.shape != (12, 1) or not (
            torch.isfinite(reset).all() and torch.isfinite(actions).all()):
        raise ValueError('forecast requires one reset vector and twelve finite applied actions')
    _, carry = learner.act(reset, deterministic=True)
    prior = learner.world_model[1].rssm_prior.module
    observations, rewards = [], []
    for action in actions:
        logits, belief = prior._belief_and_logits(carry['state'], carry['belief'], action)
        state = torch.nn.functional.one_hot(logits.argmax(-1), learner.config.classes).float().flatten(-2)
        td = TensorDict({'next': {'state': state, 'belief': belief}}, [])
        td = learner.world_model[2](td)
        td = learner.world_model[3](td)
        observations.append(td['next', 'reco_pixels'].numpy().copy())
        rewards.append(td['next', 'reward'].item())
        carry = {'state': state, 'belief': belief}
    return dict(observations=np.stack(observations), rewards=np.array(rewards))


def fit_response_baseline(episodes):
    """Fit one training-only linear response recurrence per latitude cell."""
    x, y = [], []
    for episode in episodes:
        response = episode['oracle_diagnostics']['daily_response_T_s'][:, -1]
        previous = np.vstack([np.zeros((1, 64)), response[:-1]])
        heat = (episode['policy']['actions'][:, 0] + 1) * 5
        x.append(np.stack([previous, np.broadcast_to(heat[:, None], previous.shape)], -1))
        y.append(response)
    x, y = np.concatenate(x), np.concatenate(y)
    return np.stack([np.linalg.lstsq(x[:, cell], y[:, cell], rcond=None)[0] for cell in range(64)])


def forecast_validation(learner, training, validation, band, weights, *, sols=(10, 30, 60, 120)):
    """Report ten-sol boundary errors and daily-integrated reward prediction."""
    coefficients = fit_response_baseline(training)
    errors = {name: [] for name in FORECAST_METRICS}
    for episode in validation:
        policy = episode['policy']
        reset, actions = policy['observations'][0], policy['actions']
        prediction = prior_forecast(learner, reset, actions)
        control = prior_forecast(learner, reset, np.full((12, 1), -1.))
        response = (prediction['observations'][:, :64] - control['observations'][:, :64]) * 10
        truth = episode['oracle_diagnostics']['daily_response_T_s'][:, -1]
        linear, previous = [], np.zeros(64)
        for action in actions[:, 0]:
            previous = coefficients[:, 0]*previous + coefficients[:, 1]*(action+1)*5
            linear.append(previous.copy())
        linear = np.stack(linear)
        for name, value in (('model_response', response), ('zero_response', np.zeros_like(truth)),
                            ('linear_response', linear)):
            errors[name].append(np.average((value[:, band]-truth[:, band])**2, axis=-1, weights=weights))
        target = policy['observations'][1:, :64]
        errors['model_temperature'].append(np.mean(((prediction['observations'][:, :64]-target)*10)**2, -1))
        errors['persistence_temperature'].append(np.mean(((reset[:64]-target)*10)**2, -1))
        # The simple baseline interpolates boundary response for ten daily labels.
        daily = np.concatenate([np.linspace(a, b, 11)[1:] for a, b in
                                zip(np.vstack([np.zeros((1, 64)), linear[:-1]]), linear)])
        linear_reward = -np.average((daily[:, band]-2)**2, axis=-1, weights=weights).sum()
        true_reward = policy['rewards'].sum()
        errors['model_reward'].append(float((prediction['rewards'].sum()-true_reward)**2))
        errors['zero_reward'].append(float((-480.-true_reward)**2))
        errors['linear_reward'].append(float((linear_reward-true_reward)**2))
    result = {}
    for name, values in errors.items():
        a = np.asarray(values)
        result[name] = {'rmse': float(np.sqrt(a.mean()))}
        if a.ndim == 2:
            result[name]['at_sols'] = {str(sol): float(np.sqrt(a[:, sol//10-1].mean()))
                                      for sol in sols}
    return result


def readiness(metrics, improvement=0.10):
    return (metrics['model_response']['rmse'] <= (1-improvement)*min(
                metrics['zero_response']['rmse'], metrics['linear_response']['rmse'])
            and metrics['model_temperature']['rmse'] <= metrics['persistence_temperature']['rmse']
            and metrics['model_reward']['rmse'] < min(metrics['zero_reward']['rmse'],
                                                     metrics['linear_reward']['rmse']))

"""Auxiliary open-loop response objective for the model-only ablation."""
import numpy as np
import torch
from tensordict import TensorDict


def matched_training_pairs(manifest, episodes, band):
    """Pair public measurements by condition in a verified training dataset."""
    if manifest['split'] != 'train':
        raise ValueError('paired response supervision requires the train split')
    if len(manifest['episodes']) != len(episodes):
        raise ValueError('each training record needs one trajectory')
    records = list(zip(manifest['episodes'], episodes))
    controls = {record['condition_seed']: episode['policy'] for record, episode in records
                if record['schedule'] == 'zero'}
    pairs = []
    for record, episode in records:
        if record['schedule'] == 'zero':
            continue
        control = controls[record['condition_seed']]
        observations = episode['policy']['observations']
        if not (control['actions'] == -1).all() or not np.array_equal(
                observations[0], control['observations'][0]):
            raise ValueError('matched schedules require one shared reset and a no-heating control')
        # Each public temperature unit is 10 K. Action t leads to observation t+1.
        delta = observations[1:, :64][:, band] - control['observations'][1:, :64][:, band]
        pairs.append((torch.as_tensor(observations[0]), torch.as_tensor(episode['policy']['actions']),
                      torch.as_tensor(delta * 10)))
    return pairs


def paired_response_loss(learner, reset, actions, target_kelvin, band, weights):
    """Run both schedules from one reset posterior. Score each ten-sol boundary."""
    _, carry = learner.act(reset, deterministic=True)
    state = carry['state'].expand(2, -1)
    belief = carry['belief'].expand(2, -1)
    prior = learner.world_model[1].rssm_prior.module
    responses = []
    for action in torch.stack((actions, torch.full_like(actions, -1)), dim=1):
        logits, belief = prior._belief_and_logits(state, belief, action)
        probs = logits.softmax(-1)
        mode = torch.nn.functional.one_hot(logits.argmax(-1), learner.config.classes)
        # Match forecast modes on the forward pass and propagate categorical gradients.
        state = (mode + (probs - probs.detach())).flatten(-2)
        td = TensorDict({'next': {'state': state, 'belief': belief}}, [2])
        temperatures = learner.world_model[2](td)['next', 'reco_pixels'][:, :64][:, band]
        responses.append(10 * (temperatures[0] - temperatures[1]))
    squared = (torch.stack(responses) - target_kelvin).square()
    return (squared * weights).sum(-1).mean() / weights.sum()

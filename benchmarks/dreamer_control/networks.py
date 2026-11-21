# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from pytorch/rl cc3fe0f02fc599ab367acbd75d1195139c6f7cef,
# sota-implementations/dreamer_v3/dreamer_v3_agent.py. See LICENSE.torchrl.
"""Vector-only eager composition of the pinned official DreamerV3 modules."""

import torch
from tensordict.nn import (InteractionType, ProbabilisticTensorDictModule,
                          ProbabilisticTensorDictSequential, TensorDictModule,
                          TensorDictSequential)
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.modules import DreamerV3MLP, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal
from torchrl.modules.models.model_based import (RSSMPosteriorV3, RSSMPriorV3,
                                               RSSMRolloutV3, _dreamer_v3_init)
from torchrl.objectives import symexp, symlog


class Actor(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = DreamerV3MLP(cfg.state_dim + cfg.belief_dim, None,
                                     depth=1, num_cells=cfg.hidden_dim, norm_eps=1e-3)
        self.mean = torch.nn.Linear(cfg.hidden_dim, 1)
        self.std = torch.nn.Linear(cfg.hidden_dim, 1)
        self.mean.apply(_dreamer_v3_init)
        self.std.apply(_dreamer_v3_init)
        with torch.no_grad():
            self.mean.weight.mul_(0.01)
            self.std.weight.mul_(0.01)

    def forward(self, state, belief):
        hidden = self.backbone(belief, state)
        return self.mean(hidden).tanh(), 0.9 * (self.std(hidden) + 2).sigmoid() + 0.1


def build(cfg):
    def mlp(inputs, outputs, outscale=1.):
        return DreamerV3MLP(inputs, outputs, depth=1, num_cells=cfg.hidden_dim,
                            norm_eps=1e-3, outscale=outscale)

    def td(module, inputs, outputs):
        return TensorDictModule(module, in_keys=inputs, out_keys=outputs)

    features = cfg.state_dim + cfg.belief_dim
    encoder = TensorDictSequential(
        td(symlog, [("next", "observation")], [("next", "symlog_observation")]),
        td(mlp(cfg.obs_dim, None), [("next", "symlog_observation")],
           [("next", "encoded_latents")]),
    )
    prior = RSSMPriorV3(action_shape=torch.Size([1]), action_dim=1,
                       hidden_dim=cfg.hidden_dim, rnn_hidden_dim=cfg.belief_dim,
                       num_categoricals=cfg.categoricals, num_classes=cfg.classes,
                       unimix=0.01, recurrent_model="gru", num_blocks=1,
                       num_layers=1, prior_num_layers=1, norm_eps=1e-3)
    posterior = RSSMPosteriorV3(hidden_dim=cfg.hidden_dim,
                               num_categoricals=cfg.categoricals, num_classes=cfg.classes,
                               rnn_hidden_dim=cfg.belief_dim, obs_embed_dim=cfg.hidden_dim,
                               unimix=0.01, use_rms_norm=True, num_layers=1, norm_eps=1e-3)
    rollout = RSSMRolloutV3(
        td(prior, ["state", "belief", "action"],
           [("next", "prior_logits"), ("next", "state"), ("next", "belief")]),
        td(posterior, [("next", "belief"), ("next", "encoded_latents")],
           [("next", "posterior_logits"), ("next", "state")]), reset_key="is_init")
    # One vector event has one decoder head, as in the official recipe.
    decoder = TensorDictSequential(
        td(mlp(features, cfg.obs_dim), [("next", "state"), ("next", "belief")],
           [("next", "reco_symlog_observation")]),
        td(symexp, [("next", "reco_symlog_observation")], [("next", "reco_pixels")]),
    )
    reward = mlp(features, cfg.bins, outscale=0.)
    reward_decoder = SymExpTwoHot(cfg.bins)
    continuation = mlp(features, 1)
    world = TensorDictSequential(
        encoder, rollout, decoder,
        TensorDictSequential(
            td(reward, [("next", "belief"), ("next", "state")], [("next", "reward_logits")]),
            td(reward_decoder, [("next", "reward_logits")], [("next", "reward")]),
        ),
        td(continuation, [("next", "belief"), ("next", "state")], [("next", "continue_pred")]),
    )
    imagination = WorldModelWrapper(
        td(prior, ["state", "belief", "action"], ["_", "state", "belief"]),
        TensorDictSequential(td(reward, ["belief", "state"], ["reward_logits"]),
                             td(reward_decoder, ["reward_logits"], ["reward"])),
    )
    imagined_env = DreamerEnv(imagination, (cfg.state_dim,), (cfg.belief_dim,), device="cpu")
    imagined_env.action_spec = Bounded(-1., 1., shape=(1,))
    imagined_env.reward_spec = Unbounded(1)
    imagined_env.state_spec = Composite(state=Unbounded(cfg.state_dim),
                                        belief=Unbounded(cfg.belief_dim))
    imagined_env.observation_spec = imagined_env.state_spec.clone()
    actor = ProbabilisticTensorDictSequential(
        td(Actor(cfg), ["state", "belief"], ["loc", "scale"]),
        ProbabilisticTensorDictModule(in_keys=["loc", "scale"], out_keys=["action"],
                                     default_interaction_type=InteractionType.RANDOM,
                                     distribution_class=IndependentNormal,
                                     return_log_prob=True, log_prob_key="action_log_prob"),
    )
    value = TensorDictSequential(
        td(mlp(features, cfg.bins, outscale=0.), ["belief", "state"], ["state_value_logits"]),
        td(SymExpTwoHot(cfg.bins), ["state_value_logits"], ["state_value"]),
    )
    continuation_model = TensorDictSequential(
        td(continuation, ["belief", "state"], ["continue_logits"]),
        td(torch.sigmoid, ["continue_logits"], ["continuation"]),
    )
    return world, actor, value, imagined_env, continuation_model

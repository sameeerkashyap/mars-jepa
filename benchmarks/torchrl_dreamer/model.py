"""Forecast-only DreamerV3 components; heating and sunlight condition dynamics."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchrl.modules.models.model_based import RSSMPosteriorV3, RSSMPriorV3
from torchrl.objectives.dreamer_v3 import categorical_kl_terms

from models.sequence import BaseSeqModel


class TorchRLDreamer(BaseSeqModel):
    stochastic = True

    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        self.categories, self.classes = cfg["categoricals"], cfg["classes"]
        self.belief_dim = cfg["belief"]
        self.state_dim = self.categories * self.classes
        self.unimix, self.free_nats = cfg["unimix"], cfg["free_nats"]
        self.dynamics_weight = cfg["dynamics_weight"]
        self.representation_weight = cfg["representation_weight"]
        self.reconstruction_weight = cfg["reconstruction_weight"]
        self.enc = nn.Sequential(nn.Linear(n_cell, cfg["embed"]), nn.SiLU())
        self.prior = RSSMPriorV3(
            action_shape=(n_cell + 1,), action_dim=n_cell + 1,
            hidden_dim=cfg["hidden"], rnn_hidden_dim=self.belief_dim,
            num_categoricals=self.categories, num_classes=self.classes,
            recurrent_model="block_gru", num_blocks=cfg["blocks"],
            num_layers=1, prior_num_layers=1, unimix=self.unimix)
        self.posterior = RSSMPosteriorV3(
            hidden_dim=cfg["hidden"], rnn_hidden_dim=self.belief_dim,
            obs_embed_dim=cfg["embed"], num_categoricals=self.categories,
            num_classes=self.classes, use_rms_norm=True, num_layers=1,
            unimix=self.unimix)
        self.dec = nn.Sequential(
            nn.Linear(self.belief_dim + self.state_dim, cfg["decoder_hidden"]), nn.SiLU(),
            nn.Linear(cfg["decoder_hidden"], n_cell))

    def _state(self, logits, sampled_state, sample):
        if sample:
            return sampled_state
        return F.one_hot(logits.argmax(-1), self.classes).flatten(-2).to(logits.dtype)

    def _observe(self, temperatures, previous_inputs, state, belief, sample):
        embeddings = self.enc(temperatures)
        uniform = None if sample else belief.new_full((belief.shape[0], self.categories), 0.5)
        priors, posteriors, reconstructions = [], [], []
        for t in range(temperatures.shape[1]):
            prior, _, belief = self.prior(state, belief, previous_inputs[:, t], _uniform=uniform)
            posterior, sampled = self.posterior(belief, embeddings[:, t], _uniform=uniform)
            state = self._state(posterior, sampled, sample)
            priors.append(prior)
            posteriors.append(posterior)
            reconstructions.append(self.dec(torch.cat([belief, state], dim=-1)))
        return state, belief, {
            "prior": torch.stack(priors, dim=1),
            "posterior": torch.stack(posteriors, dim=1),
            "reconstruction": torch.stack(reconstructions, dim=1),
        }

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        if (Ts_ctx.ndim != 3 or S_full.ndim != 3 or Fa_full.ndim != 2
                or Ts_ctx.shape[0] != S_full.shape[0] or S_full.shape[:2] != Fa_full.shape
                or Ts_ctx.shape[-1] != self.n_cell or S_full.shape[-1] != self.n_cell):
            raise ValueError("Expected temperatures/sunlight [batch,time,cell] and heating [batch,time]")
        batch, context, _ = Ts_ctx.shape
        if context < 1 or S_full.shape[1] <= context:
            raise ValueError("A nonempty context and forecast horizon are required")
        sample = self.training or sample
        inputs = torch.cat([self.nS(S_full), self.nFa(Fa_full).unsqueeze(-1)], dim=-1)
        normalized = self.nTs(Ts_ctx)
        # The stored input at t drives the transition from temperature t to t+1.
        previous = torch.cat([torch.zeros_like(inputs[:, :1]), inputs[:, :context - 1]], dim=1)
        state, belief, aux = self._observe(
            normalized, previous, normalized.new_zeros(batch, self.state_dim),
            normalized.new_zeros(batch, self.belief_dim), sample)
        future_inputs = inputs[:, context - 1:-1]
        aux.update(context_targets=normalized, context_state=state, context_belief=belief,
                   future_inputs=future_inputs, sample=sample)
        uniform = None if sample else belief.new_full((batch, self.categories), 0.5)
        predictions = []
        for t in range(future_inputs.shape[1]):
            logits, sampled, belief = self.prior(state, belief, future_inputs[:, t], _uniform=uniform)
            state = self._state(logits, sampled, sample)
            predictions.append(self.dec(torch.cat([belief, state], dim=-1)))
        return self.dTs(torch.stack(predictions, dim=1)), aux

    def latent_loss(self, aux, Ts_target):
        """Posterior reconstruction and KL use future temperatures as training targets only."""
        if Ts_target.ndim != 3 or Ts_target.shape[-1] != self.n_cell:
            raise ValueError("Training targets must have shape [batch,horizon,cell]")
        target = self.nTs(Ts_target)
        if target.shape[:2] != aux["future_inputs"].shape[:2]:
            raise ValueError("Training targets must match the forecast batch and horizon")
        _, _, future = self._observe(
            target, aux["future_inputs"], aux["context_state"], aux["context_belief"], aux["sample"])
        posterior = torch.cat([aux["posterior"], future["posterior"]], dim=1)
        prior = torch.cat([aux["prior"], future["prior"]], dim=1)
        dynamics, representation = categorical_kl_terms(
            posterior, prior, free_nats=self.free_nats, unimix=self.unimix)
        reconstruction = torch.cat([aux["reconstruction"], future["reconstruction"]], dim=1)
        targets = torch.cat([aux["context_targets"], target], dim=1)
        return (self.reconstruction_weight * F.mse_loss(reconstruction, targets)
                + self.dynamics_weight * dynamics + self.representation_weight * representation)

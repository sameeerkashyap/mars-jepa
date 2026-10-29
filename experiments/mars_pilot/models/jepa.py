"""Arm 7 (PREREG D9): JEPA -- joint-embedding predictive model with a T_s decoder.

  online encoder  enc(T_s_t)                 -> e_t   (per-step embedding)
  context encoder GRU over [e_k, S_k, F_a_k] -> s_K   (k < K, observed only)
  latent predictor s_t = P(s_{t-1}, [S_t, F_a_t])     (action-conditioned)
  target encoder  EMA copy of enc, applied to the TRUE future T_s (training
                  targets only; stop-gradient, not trainable)
  decoder         dec(s_t) -> T_s_t

The open-loop rollout stays in latent space: it never feeds T_s back in, and
never sees ground truth after the context.  The latent loss (predicted vs
target embeddings) is computed in `latent_loss`, which train.py calls with
the horizon truth; `rollout` itself receives only the context.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSeqModel


class JEPAModel(BaseSeqModel):
    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        e, d, dh = cfg["embed"], cfg["latent"], cfg["decoder_hidden"]
        self.ema_decay = cfg["ema_decay"]
        self.latent_weight = cfg["latent_weight"]
        act_dim = n_cell + 1                      # action = [S, F_a]
        self.enc = nn.Sequential(nn.Linear(n_cell, e), nn.SiLU(),
                                 nn.Linear(e, e))
        self.ctx = nn.GRUCell(e + act_dim, d)
        self.pred = nn.GRUCell(act_dim, d)
        self.proj = nn.Linear(d, e)               # latent -> embedding space
        self.dec = nn.Sequential(nn.Linear(d, dh), nn.SiLU(),
                                 nn.Linear(dh, n_cell))
        self.s0 = nn.Parameter(torch.zeros(d))
        self.target_enc = copy.deepcopy(self.enc)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def ema_update(self):
        """Momentum update of the target encoder (call after each opt step)."""
        for pt, po in zip(self.target_enc.parameters(), self.enc.parameters()):
            pt.mul_(self.ema_decay).add_(po.detach(), alpha=1 - self.ema_decay)

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        nTs = self.nTs(Ts_ctx)
        nS = self.nS(S_full)
        nFa = self.nFa(Fa_full)
        s = self.s0.unsqueeze(0).expand(B, -1).contiguous()
        e_ctx = self.enc(nTs)                     # (B, K, e)
        for k in range(K):
            s = self.ctx(torch.cat([e_ctx[:, k], nS[:, k], nFa[:, k:k + 1]],
                                   dim=-1), s)
        # open-loop latent rollout: actions only, no T_s fed back
        lats = []
        for j in range(H):
            t = K + j
            s = self.pred(torch.cat([nS[:, t], nFa[:, t:t + 1]], dim=-1), s)
            lats.append(s)
        z = torch.stack(lats, dim=1)              # (B, H, d)
        return self.dTs(self.dec(z)), {"latent": z}

    def latent_loss(self, aux, Ts_target):
        """JEPA loss: projected predicted latents vs EMA-encoded true T_s."""
        with torch.no_grad():
            tgt = F.normalize(self.target_enc(self.nTs(Ts_target)), dim=-1)
        pred = F.normalize(self.proj(aux["latent"]), dim=-1)
        return self.latent_weight * ((pred - tgt) ** 2).sum(-1).mean()

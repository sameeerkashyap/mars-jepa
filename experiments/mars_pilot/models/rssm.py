"""Arm 2: RSSM -- deterministic h_t + Gaussian stochastic z_t.

prior p(z|h), posterior q(z|h, obs).  KL balancing + free bits (config).
Open-loop rollouts SAMPLE from the prior.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSeqModel


def _gauss(x):
    mean, log_std = x.chunk(2, dim=-1)
    std = F.softplus(log_std) + 1e-4
    return mean, std


class RSSMModel(BaseSeqModel):
    stochastic = True

    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        d, z, hdim = cfg["deter"], cfg["stoch"], cfg["hidden"]
        self.d, self.z = d, z
        self.kl_balance = cfg["kl_balance"]
        self.free_bits = cfg["free_bits"]
        self.kl_weight = cfg["kl_weight"]
        act_dim = n_cell + 1                      # action = [S, F_a]
        self.rnn = nn.GRUCell(z + act_dim, d)
        self.prior = nn.Sequential(nn.Linear(d, hdim), nn.SiLU(),
                                   nn.Linear(hdim, 2 * z))
        self.obs_enc = nn.Linear(n_cell, hdim)
        self.post = nn.Sequential(nn.Linear(d + hdim, hdim), nn.SiLU(),
                                  nn.Linear(hdim, 2 * z))
        self.dec = nn.Sequential(nn.Linear(d + z, hdim), nn.SiLU(),
                                 nn.Linear(hdim, n_cell))
        self.h0 = nn.Parameter(torch.zeros(d))
        self.z0 = nn.Parameter(torch.zeros(z))

    def _kl(self, pm, ps, qm, qs):
        def kld(m1, s1, m2, s2):
            return (torch.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2)
                    / (2 * s2 ** 2) - 0.5).sum(-1)
        # balanced: toward prior (stop-grad posterior) and toward posterior
        kl_lhs = kld(qm.detach(), qs.detach(), pm, ps)   # train prior
        kl_rhs = kld(qm, qs, pm.detach(), ps.detach())   # train posterior
        kl = self.kl_balance * kl_lhs + (1 - self.kl_balance) * kl_rhs
        return torch.clamp(kl, min=self.free_bits).mean()

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        nTs = self.nTs(Ts_ctx)
        nS = self.nS(S_full)
        nFa = self.nFa(Fa_full)
        h = self.h0.unsqueeze(0).expand(B, -1).contiguous()
        z = self.z0.unsqueeze(0).expand(B, -1).contiguous()
        kl_terms = []
        # context with posterior
        for k in range(K):
            act = torch.cat([nS[:, k], nFa[:, k:k + 1]], dim=-1)
            h = self.rnn(torch.cat([z, act], dim=-1), h)
            pm, ps = _gauss(self.prior(h))
            e = self.obs_enc(nTs[:, k])
            qm, qs = _gauss(self.post(torch.cat([h, e], dim=-1)))
            kl_terms.append(self._kl(pm, ps, qm, qs))
            z = qm + qs * torch.randn_like(qs) if (self.training or sample) else qm
        # open-loop rollout from the PRIOR (no ground-truth obs)
        preds = []
        for j in range(H):
            t = K + j
            act = torch.cat([nS[:, t], nFa[:, t:t + 1]], dim=-1)
            h = self.rnn(torch.cat([z, act], dim=-1), h)
            pm, ps = _gauss(self.prior(h))
            z = pm + ps * torch.randn_like(ps) if (self.training or sample) else pm
            preds.append(self.dec(torch.cat([h, z], dim=-1)))
        pred_norm = torch.stack(preds, dim=1)
        aux = {"kl": self.kl_weight * torch.stack(kl_terms).mean()}
        return self.dTs(pred_norm), aux

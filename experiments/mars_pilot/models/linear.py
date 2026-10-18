"""Arm 5: LINEAR baseline -- ARX linear state-space on T_s.

Per-cell autoregressive model with exogenous inputs:
    T_s[t] = sum_{i=1..p} a_i T_s[t-i] + b_S S[t] + b_F F_a[t] + c
operating in NORMALIZED space, with cell-shared coefficients plus a per-cell
bias.  This is the ridge/ARX control baseline; it is intentionally small and is
NOT part of the parameter-count matching set.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSeqModel


class LinearARX(BaseSeqModel):
    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        self.p = cfg["ar_order"]
        # cell-shared AR + exogenous weights; per-cell bias
        self.a = nn.Parameter(torch.zeros(self.p))
        self.bS = nn.Parameter(torch.zeros(1))
        self.bF = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(n_cell))
        with torch.no_grad():
            self.a[0] = 1.0                       # persistence init

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        nTs = self.nTs(Ts_ctx)
        nS = self.nS(S_full)
        nFa = self.nFa(Fa_full)
        # history buffer of last p normalized T_s
        hist = [nTs[:, K - 1 - i] if (K - 1 - i) >= 0 else nTs[:, 0]
                for i in range(self.p)]
        preds = []
        for j in range(H):
            t = K + j
            pred = self.bias.unsqueeze(0).expand(B, -1).clone()
            for i in range(self.p):
                pred = pred + self.a[i] * hist[i]
            pred = pred + self.bS * nS[:, t] + self.bF * nFa[:, t].unsqueeze(-1)
            preds.append(pred)
            hist = [pred] + hist[:-1]
        pred_norm = torch.stack(preds, dim=1)
        return self.dTs(pred_norm), {}

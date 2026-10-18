"""Arm 1: deterministic action-conditioned GRU with an MLP decoder to T_s."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseSeqModel


class GRUModel(BaseSeqModel):
    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        h = cfg["hidden"]
        dh = cfg["decoder_hidden"]
        self.hidden = h
        in_dim = n_cell + n_cell + 1          # [T_s, S, F_a]
        self.cell = nn.GRUCell(in_dim, h)
        self.dec = nn.Sequential(
            nn.Linear(h, dh), nn.SiLU(), nn.Linear(dh, n_cell))
        self.h0 = nn.Parameter(torch.zeros(h))

    def _step_input(self, Ts, S, Fa):
        # Ts, S: (B, ncell) normalized; Fa: (B,) normalized
        return torch.cat([Ts, S, Fa.unsqueeze(-1)], dim=-1)

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        nTs = self.nTs(Ts_ctx)
        nS = self.nS(S_full)
        nFa = self.nFa(Fa_full)
        hid = self.h0.unsqueeze(0).expand(B, -1).contiguous()
        # context (teacher-forced on observed T_s)
        for k in range(K):
            inp = self._step_input(nTs[:, k], nS[:, k], nFa[:, k])
            hid = self.cell(inp, hid)
        # open-loop rollout: feed back own predictions (NO ground truth)
        prev = nTs[:, K - 1]
        preds = []
        for j in range(H):
            t = K + j
            pred = self.dec(hid)                  # normalized T_s
            preds.append(pred)
            inp = self._step_input(pred, nS[:, t], nFa[:, t])
            hid = self.cell(inp, hid)
            prev = pred
        pred_norm = torch.stack(preds, dim=1)      # (B, H, ncell)
        return self.dTs(pred_norm), {}

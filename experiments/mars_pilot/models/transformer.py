"""Arm 3: causal Transformer over (obs, S, F_a) tokens, autoregressive."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseSeqModel


class TransformerModel(BaseSeqModel):
    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        d = cfg["d_model"]
        self.d = d
        self.embed = nn.Linear(n_cell + n_cell + 1, d)   # token = [T_s,S,F_a]
        self.register_buffer("pos", self._sinusoid(4096, d))  # fixed, non-trainable
        layer = nn.TransformerEncoderLayer(
            d, cfg["n_heads"], cfg["ff"], batch_first=True, activation="gelu",
            dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, cfg["n_layers"])
        self.head = nn.Linear(d, n_cell)

    @staticmethod
    def _sinusoid(T, d):
        pos = torch.arange(T).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(1, T, d)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _tokens(self, Ts, S, Fa):
        return self.embed(torch.cat([Ts, S, Fa.unsqueeze(-1)], dim=-1))

    def _forward(self, tok):
        T = tok.shape[1]
        tok = tok + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, device=tok.device), diagonal=1).bool()
        out = self.enc(tok, mask=mask)
        return self.head(out[:, -1])              # predict next T_s (normalized)

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        nTs = self.nTs(Ts_ctx)
        nS = self.nS(S_full)
        nFa = self.nFa(Fa_full)
        # build context tokens
        seq = self._tokens(nTs, nS[:, :K], nFa[:, :K])   # (B, K, d)
        preds = []
        cur_Ts = nTs[:, -1:]                      # last observed
        for j in range(H):
            t = K + j
            pred = self._forward(seq)             # (B, ncell) for step t
            preds.append(pred)
            # append the predicted token (open-loop; no ground truth)
            new_tok = self._tokens(pred.unsqueeze(1), nS[:, t:t + 1],
                                   nFa[:, t:t + 1])
            seq = torch.cat([seq, new_tok], dim=1)
        pred_norm = torch.stack(preds, dim=1)
        return self.dTs(pred_norm), {}

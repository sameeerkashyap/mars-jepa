"""Arm 4: HYBRID = fixed simplified-EBM backbone + learned correction g_theta.

    C_eff dT/dt = [EBM RHS: S(1-A) - eps*sigma*T^4 + D*diff + F_a] + g_theta(feat)

with the CO2 frost clamp applied to the combined update.  The EBM core (fitted
C_eff, D, eps and the frost clamp) is FIXED; only g_theta trains.  g_theta sees
ONLY observable features (recent T_s, S, F_a, latitude).

The backbone's internal ice state is initialized from the OBSERVED context via
EBMBackbone.init_ice_from_context (documented; NEVER from hidden m).

Also defines EBMOnly: the fitted EBM backbone with g_theta == 0 (no learning).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from common import load_config
from physics_torch import EBMBackbone, load_ebm_params, stored_dt_seconds
from .base import BaseSeqModel


class HybridModel(BaseSeqModel):
    needs_norm = False

    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        full = load_config()
        params = load_ebm_params()
        dt = stored_dt_seconds(full)
        self.backbone = EBMBackbone(
            params, n_cell, dt, m_ref=cfg["m_ref"],
            frost_init_band_K=cfg["frost_init_band_K"])
        self.register_buffer("lat_buf", torch.zeros(n_cell))
        self._lat_set = False
        gh = cfg["g_hidden"]
        layers = []
        in_dim = n_cell + n_cell + 1 + n_cell     # [T_s, S, F_a, lat]
        d = in_dim
        for _ in range(cfg["g_layers"]):
            layers += [nn.Linear(d, gh), nn.SiLU()]
            d = gh
        layers += [nn.Linear(d, n_cell)]
        self.g = nn.Sequential(*layers)
        # scale the correction so initial g is small relative to fluxes
        self.g_scale = 10.0

    def _device_backbone(self, device):
        bb = self.backbone
        if bb.L.device != device:
            bb.L = bb.L.to(device)
            bb.diff_inv = bb.diff_inv.to(device)
            bb.device = device
        return bb

    def _feat(self, Ts, S, Fa, lat):
        # normalized observable features for g_theta
        nTs = self.nTs(Ts)
        nS = self.nS(S)
        nFa = self.nFa(Fa).unsqueeze(-1)
        latf = lat.unsqueeze(0).expand(Ts.shape[0], -1)
        return torch.cat([nTs, nS, nFa, latf], dim=-1)

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        device = Ts_ctx.device
        bb = self._device_backbone(device)
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        # init ice state from OBSERVED context (never hidden m)
        m = bb.init_ice_from_context(Ts_ctx)
        T = Ts_ctx[:, -1].clone()                 # physical K
        preds = []
        for j in range(H):
            t = K + j
            S_t = S_full[:, t]
            Fa_t = Fa_full[:, t]
            g = self.g(self._feat(T, S_t, Fa_t, lat)) * self.g_scale
            T, m = bb.step(T, m, S_t, Fa_t, g)
            preds.append(T)
        return torch.stack(preds, dim=1), {}


class EBMOnly(BaseSeqModel):
    """Fitted EBM backbone, NO learning (g == 0).  A baseline."""
    needs_norm = False

    def __init__(self, stats, n_cell, cfg):
        super().__init__(stats, n_cell)
        full = load_config()
        params = load_ebm_params()
        dt = stored_dt_seconds(full)
        self.backbone = EBMBackbone(
            params, n_cell, dt, m_ref=cfg["m_ref"],
            frost_init_band_K=cfg["frost_init_band_K"])

    def rollout(self, Ts_ctx, S_full, Fa_full, lat, sample=False):
        device = Ts_ctx.device
        bb = self.backbone
        if bb.L.device != device:
            bb.L = bb.L.to(device)
            bb.diff_inv = bb.diff_inv.to(device)
        B, K, _ = Ts_ctx.shape
        H = S_full.shape[1] - K
        m = bb.init_ice_from_context(Ts_ctx)
        T = Ts_ctx[:, -1].clone()
        zero = torch.zeros(B, self.n_cell, device=device)
        preds = []
        for j in range(H):
            t = K + j
            T, m = bb.step(T, m, S_full[:, t], Fa_full[:, t], zero)
            preds.append(T)
        return torch.stack(preds, dim=1), {}

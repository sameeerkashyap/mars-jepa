"""
Window slicing + TRAIN-ONLY normalization for the open-loop protocol.

Protocol: observe K context steps, then roll out H steps open-loop given FUTURE
S and F_a (known exogenous inputs).  A window has length K+H.

Normalization statistics (mean/std of T_s, S, F_a) are computed from the
training SUBSET a run actually trains on (its data-fraction subsample of the
TRAIN split) and then applied everywhere (train/val/test).  Tests assert stats
depend only on that subset.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_norm_stats(trajs):
    """Mean/std of T_s, S, F_a over a list of (train) trajectory dicts."""
    Ts = np.concatenate([np.asarray(t["T_s"]).reshape(-1) for t in trajs])
    S = np.concatenate([np.asarray(t["S"]).reshape(-1) for t in trajs])
    Fa = np.concatenate([np.asarray(t["F_a"]).reshape(-1) for t in trajs])
    stats = {
        "T_s_mean": float(Ts.mean()), "T_s_std": float(Ts.std() + 1e-8),
        "S_mean": float(S.mean()), "S_std": float(S.std() + 1e-8),
        "F_a_mean": float(Fa.mean()), "F_a_std": float(Fa.std() + 1e-8),
    }
    return stats


def make_windows(trajs, K, H, stride):
    """Slice trajectories into fixed windows of length K+H.

    Returns dict of numpy arrays:
      Ts  (N, K+H, ncell)   S (N, K+H, ncell)   Fa (N, K+H)
      lat (ncell,)
    """
    W = K + H
    Ts_list, S_list, Fa_list = [], [], []
    lat = None
    for t in trajs:
        Ts = np.asarray(t["T_s"], dtype=np.float32)
        S = np.asarray(t["S"], dtype=np.float32)
        Fa = np.asarray(t["F_a"], dtype=np.float32)
        if lat is None:
            lat = np.asarray(t["lat"], dtype=np.float32)
        n = Ts.shape[0]
        if n < W:
            continue
        for s in range(0, n - W + 1, stride):
            Ts_list.append(Ts[s:s + W])
            S_list.append(S[s:s + W])
            Fa_list.append(Fa[s:s + W])
    if not Ts_list:
        raise ValueError("no windows produced; trajectory shorter than K+H?")
    return {
        "Ts": np.stack(Ts_list), "S": np.stack(S_list),
        "Fa": np.stack(Fa_list), "lat": lat,
    }


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows, stats, K, H):
        self.K = K
        self.H = H
        self.stats = stats
        self.Ts = torch.tensor(windows["Ts"])
        self.S = torch.tensor(windows["S"])
        self.Fa = torch.tensor(windows["Fa"])
        self.lat = torch.tensor(windows["lat"])

    def __len__(self):
        return self.Ts.shape[0]

    def __getitem__(self, i):
        return self.Ts[i], self.S[i], self.Fa[i]


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean

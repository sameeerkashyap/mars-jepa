"""Build the world model or a comparison architecture for the heating pilot."""

import torch

from models.sequence import BaseSeqModel as BaseSeqModel
from models.sequence import count_params as count_params


def build_model(arm, stats, n_cell, arch, seed=0):
    """Factory.  `arm` in {gru, rssm, transformer, hybrid, linear, ebm_only,
    gru_noforce, jepa}."""
    torch.manual_seed(seed)
    if arm == "gru":
        from benchmarks.gru.model import GRUModel
        return GRUModel(stats, n_cell, arch["gru"])
    if arm == "rssm":
        from benchmarks.rssm.model import RSSMModel
        return RSSMModel(stats, n_cell, arch["rssm"])
    if arm == "transformer":
        from benchmarks.transformer.model import TransformerModel
        return TransformerModel(stats, n_cell, arch["transformer"])
    if arm == "hybrid":
        from benchmarks.hybrid.model import HybridModel
        return HybridModel(stats, n_cell, arch["hybrid"])
    if arm == "linear":
        from benchmarks.linear.model import LinearARX
        return LinearARX(stats, n_cell, arch["linear"])
    if arm == "ebm_only":
        from benchmarks.hybrid.model import EBMOnly
        return EBMOnly(stats, n_cell, arch["hybrid"])
    if arm == "gru_noforce":
        from benchmarks.gru.model import GRUNoForce
        return GRUNoForce(stats, n_cell, arch["gru"])
    if arm == "jepa":
        from models.latent_jepa import JEPAModel
        return JEPAModel(stats, n_cell, arch["jepa"])
    raise ValueError(f"unknown arm {arm}")

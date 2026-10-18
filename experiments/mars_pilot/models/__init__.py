"""Phase-2 model arms.

Every arm implements the same interface (see base.BaseSeqModel):

    rollout(Ts_ctx, S_full, Fa_full, lat, sample=False) -> (pred_phys, aux)

where predictions are PHYSICAL T_s (Kelvin) over the H-step open-loop horizon.
Models normalize internally from TRAIN-only stats passed at construction.

IMPORT-SAFETY: nothing in this package may import the eval-label loader
(enforced by a test).  Models never receive hidden variables.
"""

from .base import BaseSeqModel, count_params, build_model  # noqa: F401

"""Device-agnostic training loop for the PI-JEPA model.

Designed to run unchanged on a laptop CPU or inside a Kubernetes GPU pod. It
handles multi-step rollout, EMA target updates, validation, and checkpointing.

The trainer is intentionally framework-light (no Lightning/accelerate) to keep
the container image small and the control flow explicit for a research code
base.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from marsworld.config import Config
from marsworld.models.jepa import PIJEPA, JEPAOutput
from marsworld.training.losses import CompositeLoss
from marsworld.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(choice: str) -> torch.device:
    """Map a config device string to an actual torch device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


class Trainer:
    """Own the optimisation loop, rollout logic and checkpointing.

    Parameters
    ----------
    model:
        A :class:`PIJEPA` instance.
    cfg:
        Full :class:`Config`.
    """

    def __init__(self, model: PIJEPA, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.loss_fn = CompositeLoss(cfg.train, cfg.physics)
        self.opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.ckpt_dir = Path(cfg.train.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Trainer on device=%s | params=%s", self.device, model.num_parameters())

    # --- rollout -----------------------------------------------------------
    def _rollout_step(self, context, target, forcing) -> Dict[str, torch.Tensor]:
        """Run a (possibly multi-step) latent rollout and accumulate losses.

        ``target`` has shape ``(B, R, C, H, W)``. For R>1 the predictor is
        applied autoregressively in latent space; the forecast/physics losses
        are evaluated at every step against the ground-truth future fields.
        """
        context = context.to(self.device)
        target = target.to(self.device)
        forcing = forcing.to(self.device)

        z = self.model.encode(context)
        x_prev = context
        accum: Dict[str, torch.Tensor] = {}
        rollout = target.shape[1]

        for step in range(rollout):
            x_next_true = target[:, step]
            z_pred = self.model.predict_latent(z, forcing)
            z_target = self.model.encode_target(x_next_true)
            # Residual-aware decode: next field is x_prev + decode(z) in
            # residual mode, so the loss is measured against persistence.
            x_pred = self.model.decode_field(z_pred, x_prev)
            out = JEPAOutput(z_context=z, z_pred=z_pred, z_target=z_target, x_pred=x_pred)

            terms = self.loss_fn(out, x_prev, x_next_true)
            # Persistence-relative skill: forecast MSE vs the x_next:=x_prev baseline.
            with torch.no_grad():
                persist = torch.nn.functional.mse_loss(x_prev, x_next_true)
                accum["persistence_mse"] = accum.get("persistence_mse", 0.0) + persist
            for k, v in terms.items():
                accum[k] = accum.get(k, 0.0) + v

            z = z_pred  # autoregress in latent space
            x_prev = x_next_true

        return {k: v / rollout for k, v in accum.items()}

    # --- epochs ------------------------------------------------------------
    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        for i, (context, target, forcing) in enumerate(loader):
            terms = self._rollout_step(context, target, forcing)
            self.opt.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.model.update_target_encoder()

            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
            if i % self.cfg.train.log_every == 0:
                logger.info(
                    "epoch %d step %d | total=%.4f", epoch, i, float(terms["total"].detach())
                )
        return {k: v / len(loader) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running: Dict[str, float] = {}
        for context, target, forcing in loader:
            terms = self._rollout_step(context, target, forcing)
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + float(v.detach())
        n = max(1, len(loader))
        return {k: v / n for k, v in running.items()}

    # --- orchestration -----------------------------------------------------
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        best = float("inf")
        for epoch in range(self.cfg.train.epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            msg = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            logger.info("[train] epoch %d | %s", epoch, msg)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                vmsg = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                logger.info("[val]   epoch %d | %s", epoch, vmsg)
                if val_metrics["total"] < best:
                    best = val_metrics["total"]
                    self.save_checkpoint("best.pt", epoch, val_metrics)

        self.save_checkpoint("last.pt", self.cfg.train.epochs - 1, {})

    def save_checkpoint(self, name: str, epoch: int, metrics: Dict[str, float]) -> None:
        path = self.ckpt_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "config": self.cfg.to_dict(),
                "metrics": metrics,
            },
            path,
        )
        logger.info("saved checkpoint -> %s", path)

import argparse
import logging
import os

import torch
import torch.nn as nn

from models.pgc import PGCNetwork


logger = logging.getLogger(__name__)


class PGCTrainer:

    def __init__(self, opt: argparse.Namespace):
        self.opt = opt

        # -------- Persisted scalar config (no hasattr fallbacks) --------
        self.label_smoothing = float(opt.label_smoothing)
        self.accumulation_steps = int(opt.accumulation_steps)
        self.max_grad_norm = float(opt.max_grad_norm)

        # -------- Filesystem layout --------
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        # -------- Device --------
        self.device: torch.device = opt.device

        # -------- Build model --------
        self.model: PGCNetwork = PGCNetwork(
            dino_variant=opt.dino_variant,
            num_classes=1,
            lora_rank=int(opt.lora_rank),
            lora_alpha=float(opt.lora_alpha),
            lora_dropout=float(opt.lora_dropout),
            lora_targets=None,  # use PGCNetwork's documented default
            pretrained_root=opt.dino_pretrained_root,
            tau_rgb=float(opt.tau_rgb),
            tau_res=float(opt.tau_res),
        )
        self.model.to(self.device)

        # -------- Parameter group(s) --------
        params = self.model.get_trainable_params()
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in params)
        logger.info(
            "[PGCTrainer] Total params: %s | Trainable: %s (%.2f%%)",
            f"{total_params:,}",
            f"{trainable_params:,}",
            (trainable_params / total_params * 100.0) if total_params else 0.0,
        )

        # -------- Optimizer --------
        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=float(opt.lr),
                betas=(0.9, 0.999),
                weight_decay=float(opt.weight_decay),
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params,
                lr=float(opt.lr),
                momentum=0.0,
                weight_decay=float(opt.weight_decay),
            )
        else:
            raise ValueError("opt.optim must be one of {'adam', 'sgd'}")

        # -------- Scheduler (filled in via setup_scheduler) --------
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # -------- Loss --------
        self.loss_fn = nn.BCEWithLogitsLoss()
        logger.info(
            "[PGCTrainer] BCEWithLogitsLoss + label smoothing (eps=%.3f), "
            "grad_clip=%.2f, accumulation=%d",
            self.label_smoothing, self.max_grad_norm, self.accumulation_steps,
        )

        # -------- Step counters / live tensors (declared for explicitness) --------
        self.total_steps: int = 0
        self.current_step: int = 0
        self.input_imgs: torch.Tensor | None = None
        self.labels: torch.Tensor | None = None
        # Most-recent fused feature / logit / loss (populated by forward).
        self.feature: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self.loss: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Scheduler                                                           #
    # ------------------------------------------------------------------ #

    def setup_scheduler(self, total_steps: int):
        """Create a CosineAnnealingLR scheduler with ``T_max=total_steps``.

        Called from ``train.py`` once the dataloader length is known.
        ``eta_min`` is set to ``1%`` of the initial LR.
        """
        eta_min = float(self.opt.lr) * 0.01
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=int(total_steps), eta_min=eta_min,
        )
        logger.info(
            "[PGCTrainer] Scheduler ready: CosineAnnealingLR(T_max=%d, eta_min=%.2e)",
            int(total_steps), eta_min,
        )

    # ------------------------------------------------------------------ #
    # Train / eval mode toggles                                           #
    # ------------------------------------------------------------------ #

    def train(self):
        """Set the underlying model to training mode."""
        self.model.train()

    def eval(self):
        """Set the underlying model to evaluation mode."""
        self.model.eval()

    # ------------------------------------------------------------------ #
    # Step API                                                            #
    # ------------------------------------------------------------------ #

    def set_input(self, batch: dict):
        """Move a batch onto the trainer device.

        Args:
            batch: dict with keys
              * ``"image"``  - ``[B, 6, H, W]`` (RGB ++ residual concat)
              * ``"label"``  - ``[B]``           (0=real, 1=fake)
              * ``"source"`` - ``List[str]``     (subset names; not used here)
        """
        self.input_imgs = batch["image"].to(self.device)
        self.labels = batch["label"].to(self.device).float()

    def _forward(self):
        """Run a forward pass and store ``self.feature`` / ``self.output``."""
        feature, logits = self.model(self.input_imgs, return_feature=True)
        self.feature = feature
        # Standardize logits to [B, 1] then to [B] for the BCE call site.
        self.output = logits.view(-1, 1)

    def optimize_parameters(self):
        """Single training step with grad accumulation + label smoothing."""
        self.current_step += 1
        self.total_steps += 1

        self._forward()

        # Manual label smoothing: y_smooth = y * (1 - eps) + eps / 2
        labels_smooth = (
            self.labels * (1.0 - self.label_smoothing)
            + self.label_smoothing / 2.0
        )

        # Compute loss; store the unscaled value on self.loss for logging.
        loss = self.loss_fn(self.output.squeeze(1), labels_smooth)
        self.loss = loss

        # Scale by accumulation count for gradient averaging across micro-batches.
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        if self.current_step % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

    def finalize_epoch(self):

        if self.current_step % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_networks(self, filename: str):

        save_path = os.path.join(self.save_dir, filename)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "opt": vars(self.opt),
        }
        torch.save(state, save_path)
        logger.info("[PGCTrainer] Saved checkpoint to: %s", save_path)

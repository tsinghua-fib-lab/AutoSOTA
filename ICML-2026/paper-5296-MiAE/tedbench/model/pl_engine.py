from typing import Any

import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.text import Perplexity

from ..lr_schedulers import get_layerwise_lr_decay
from .loss import RMSDLoss


class MiAE(pl.LightningModule):
    """PyTorch Lightning module for MiAE pretraining.

    Wraps :class:`~tedbench.model.MiAECore` and manages the training loop,
    masking ratio, Gaussian coordinate noise, and the composite
    :class:`~tedbench.model.ESM3Loss` objective.

    Args:
        cfg: Hydra configuration node.  Expected sub-keys:

            - ``cfg.model`` — instantiated by :func:`tedbench.model.miae_model`.
            - ``cfg.mask_ratio`` — fraction of residue frames to mask (default 0.9).
            - ``cfg.noise`` — std of coordinate noise added during training (Å).
            - ``cfg.train.optimizer`` — AdamW config.
            - ``cfg.train.lr_scheduler`` — cosine schedule config.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = hydra.utils.call(cfg.model)
        self.mask_ratio = cfg.mask_ratio
        self.noise = getattr(cfg, "noise", 0.0)

        self.instantiate_loss()
        self.save_hyperparameters()

    def instantiate_loss(self) -> None:
        self.loss_fn = self.model.loss_fn
        self.val_loss_fn = RMSDLoss()

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        seq_tokens: torch.Tensor | None = None,
        mask_ratio: float = 0.0,
        noise: float = 0.0,
    ) -> dict:
        return self.model(
            coords,
            mask,
            residue_index,
            seq_tokens=seq_tokens,
            mask_ratio=mask_ratio,
            noise=noise,
        )

    def forward_with_masked_ids(self, data):
        coords, mask, residue_index, seq_tokens, ids_masked, ids_restore = (
            data["coords"],
            data["mask"],
            data["residue_index"] - 1,
            data["seq_ids"],
            data["ids_masked"],
            data["ids_restore"],
        )
        return self.model.forward_with_masked_ids(
            coords,
            mask,
            residue_index,
            seq_tokens=seq_tokens,
            ids_masked=ids_masked,
            ids_restore=ids_restore,
        )

    def shared_step(self, data, batch_idx, phase="train"):
        coords, mask, residue_index, seq_tokens, protein_chain = (
            data["coords"],
            data["mask"],
            data["residue_index"] - 1,
            data["seq_ids"],
            data["protein_chain"],
        )
        bs = coords.shape[0]
        pred_dict = self(
            coords,
            mask,
            residue_index,
            seq_tokens,
            mask_ratio=self.mask_ratio,
            noise=self.noise,
        )
        true_dict = dict(
            coords=coords, mask=mask, seq_tokens=seq_tokens, protein_chain=protein_chain
        )
        loss, metrics = self.loss_fn(pred_dict, true_dict)
        if phase != "train":
            val_metrics = self.val_loss_fn(pred_dict, true_dict)
            metrics = {**metrics, **val_metrics}
        metrics["loss"] = loss
        log_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        self.log_dict(log_metrics, sync_dist=True, batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="val")
        return loss

    def configure_optimizers(self):
        import inspect

        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if fused_available else dict()
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer,
            optim_groups,
            **extra_args,
        )
        lr_scheduler = hydra.utils.call(
            self.cfg.train.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }


class MiAEClassifier(pl.LightningModule):
    """PyTorch Lightning module for supervised training and fine-tuning on TEDBench.

    Wraps :class:`~tedbench.model.MiAEEncoder` for classification.  Supports
    both full end-to-end fine-tuning (with optional layer-wise learning rate
    decay) and linear probing (encoder frozen, only the head trained).

    Args:
        cfg: Hydra configuration node.  Expected sub-keys:

            - ``cfg.model`` — instantiated by :func:`tedbench.model.miae_encoder_model`.
            - ``cfg.train.loss`` — classification loss (CrossEntropyLoss).
            - ``cfg.train.optimizer`` — AdamW config.
            - ``cfg.train.lr_scheduler`` — cosine schedule config.
            - ``cfg.train.llrd`` — layer-wise LR decay factor (use ``< 1`` to enable).
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = hydra.utils.call(cfg.model)

        self.instantiate_loss()
        self.save_hyperparameters()

    def instantiate_loss(self) -> None:
        self.loss_fn = hydra.utils.call(self.cfg.train.loss)
        if not getattr(self.cfg.model, "dense", False):
            self.metric_fn = MetricCollection(
                {
                    "balanced_acc": MulticlassAccuracy(
                        num_classes=self.cfg.model.num_classes, average="macro"
                    ),
                    "macro_f1": MulticlassF1Score(
                        num_classes=self.cfg.model.num_classes, average="macro"
                    ),
                }
            )
        else:
            from minesm.utils.constants import esm3 as C

            from .metrics import MedianRecovery

            self.metric_fn = MetricCollection(
                {
                    "ppl": Perplexity(ignore_index=C.SEQUENCE_PAD_TOKEN),
                    "acc_median": MedianRecovery(ignore_index=C.SEQUENCE_PAD_TOKEN),
                }
            )

    def set_mode(self, mode: str = "linprobe") -> None:
        if mode == "linprobe":
            torch.nn.init.trunc_normal_(self.model.head.weight, std=0.01)
            self.model.head = torch.nn.Sequential(
                torch.nn.BatchNorm1d(
                    self.model.head.in_features, affine=False, eps=1e-6
                ),
                self.model.head,
            )
            for _, p in self.model.named_parameters():
                p.requires_grad = False
            for _, p in self.model.head.named_parameters():
                p.requires_grad = True

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        residue_index: torch.Tensor,
        seq_tokens: torch.Tensor | None = None,
        repr_only: bool = False,
    ) -> torch.Tensor:
        return self.model(
            coords, mask, residue_index, seq_tokens=seq_tokens, repr_only=repr_only
        )

    def shared_step(self, data, batch_idx, phase="train"):
        coords, mask, residue_index, seq_tokens, protein_chain = (
            data["coords"],
            data["mask"],
            data["residue_index"] - 1,
            data["seq_ids"],
            data["protein_chain"],
        )
        y_true = data.get("label")
        bs = coords.shape[0]
        y_pred = self(coords, mask, residue_index, seq_tokens=seq_tokens)
        loss = self.loss_fn(y_pred, y_true)
        with torch.no_grad():
            acc = (y_pred.argmax(dim=-1) == y_true).float().mean()
        if phase != "train":
            self.metric_fn.update(y_pred, y_true)
        metrics = {"loss": loss, "acc": acc}
        log_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        self.log_dict(
            log_metrics, sync_dist=True, batch_size=bs, on_step=False, on_epoch=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, phase="test")
        return loss

    def on_validation_epoch_end(self):
        metrics = self.metric_fn.compute()
        for key, value in metrics.items():
            self.log(f"val/{key}", value, sync_dist=True)
        self.metric_fn.reset()

    def on_test_epoch_end(self):
        metrics = self.metric_fn.compute()
        for key, value in metrics.items():
            self.log(f"test/{key}", value, sync_dist=True)
        self.metric_fn.reset()

    def configure_optimizers(self):
        if self.cfg.train.llrd < 1:
            optim_groups = get_layerwise_lr_decay(
                self.model, self.cfg.train.optimizer.lr, self.cfg.train.llrd
            )
        else:
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
        optimizer = hydra.utils.instantiate(self.cfg.train.optimizer, optim_groups)
        lr_scheduler = hydra.utils.call(
            self.cfg.train.lr_scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

from __future__ import annotations

from typing import Mapping, Optional, Type

import pytorch_lightning as pl
import torch
from torchmetrics import Metric, MetricCollection

from tsl.metrics.torch import MaskedMetric


class scalePredictor(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: MaskedMetric,
        metrics: Optional[Mapping[str, Metric]] = None,
        *,
        optim_class: Type = torch.optim.Adam,
        optim_kwargs: Optional[Mapping] = None,
        scheduler_class: Optional[Type] = None,
        scheduler_kwargs: Optional[Mapping] = None,
        scale_target: bool = False,
        quantiles: list[float] | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.scale_target = scale_target
        self.quantiles = list(quantiles or [])

        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {}

        if metrics is None:
            metrics = {}
        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.val_metrics = MetricCollection(metrics, prefix="val_")
        self.test_metrics = MetricCollection(metrics, prefix="test_")

    def forward(self, x, u=None):
        if u is None:
            batch, steps, _nodes = x.shape[:3]
            u = torch.zeros(batch, steps, 0, device=x.device, dtype=x.dtype)
        return self.model(x, u)

    def _unpack_batch(self, batch):
        inputs, targets = batch.input, batch.target
        mask = batch.get("mask_target")
        if mask is None and "mask_target" in inputs:
            mask = inputs["mask_target"]
        return inputs, targets, mask

    def _shared_step(self, batch):
        inputs, targets, mask = self._unpack_batch(batch)
        x = inputs["x"]
        u = inputs.get("u")
        if u is None:
            batch_size, steps, _nodes = x.shape[:3]
            u = torch.zeros(batch_size, steps, 0, device=x.device, dtype=x.dtype)

        y = targets["y"]
        if y.dim() == 4:
            if y.size(-1) == 1:
                y = y.squeeze(-1)
            elif y.size(1) == 1:
                y = y.squeeze(1).permute(0, 2, 1)
            elif y.size(1) == y.size(-1):
                y = torch.diagonal(y, dim1=1, dim2=3).permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Unsupported target shape {tuple(y.shape)} for scale; "
                    "expected last dim = 1 or time/feature dims to align."
                )

        if mask is not None and mask.dim() == 4:
            if mask.size(-1) == 1:
                mask = mask.squeeze(-1)
            elif mask.size(1) == 1:
                mask = mask.squeeze(1).permute(0, 2, 1)
            elif mask.size(1) == mask.size(-1):
                mask = torch.diagonal(mask, dim1=1, dim2=3).permute(0, 2, 1)
            else:
                raise ValueError(
                    f"Unsupported mask shape {tuple(mask.shape)} for scale; "
                    "expected last dim = 1 or time/feature dims to align."
                )

        y_hat = self.forward(x, u)  # (B,H,N,Q)
        y_hat_q = y_hat.permute(3, 0, 1, 2).contiguous()  # (Q,B,H,N)
        loss = self.loss_fn(y_hat_q, y, mask)
        batch_size = int(y.shape[0])
        return y_hat_q, y, loss, mask, batch_size

    def training_step(self, batch, batch_idx):
        y_hat, y, loss, mask, batch_size = self._shared_step(batch)
        self.train_metrics.update(y_hat, y, mask)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, loss, mask, batch_size = self._shared_step(batch)
        self.val_metrics.update(y_hat, y, mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, loss, mask, batch_size = self._shared_step(batch)
        self.test_metrics.update(y_hat, y, mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        if self.scheduler_class is None:
            return optimizer
        scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


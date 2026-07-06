from typing import Callable

import lightning.pytorch as pl
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR

from quarc.models_code.modules.ffn_heads import (
    FFNBaseHead,
    FFNAgentAmountHead,
    FFNAgentHeadWithRxnClass,
    FFNAgentHead,
    FFNReactantAmountHead,
    FFNTemperatureHead,
)


class BaseFFN(pl.LightningModule):
    """Base FFN model for reaction prediction tasks."""

    def __init__(
        self,
        predictor: FFNBaseHead,
        metrics: dict[str, Callable[[Tensor, Tensor], Tensor]] = None,
        warmup_epochs: int = 2,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
        gamma: float = 0.98,
    ):
        super().__init__()
        if metrics is None:
            raise ValueError("Need callable metrics")
        self.save_hyperparameters(ignore=["predictor", "metrics"])

        self.predictor = predictor
        self.criterion = predictor.criterion
        self.metrics = metrics

        # Learning rate parameters
        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.gamma = gamma

    def forward(self, FP_inputs: Tensor, agent_input: Tensor) -> Tensor:
        return self.predictor(FP_inputs, agent_input)

    def loss_fn(self, preds: Tensor, targets: Tensor, *args) -> Tensor:
        return self.criterion(preds, targets)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

        l = self.loss_fn(preds, targets)
        self.log(
            "loss/train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "loss/val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)

        return val_loss

    # def on_train_epoch_start(self) -> None:
    #     lr = self.trainer.optimizers[0].param_groups[0]["lr"]
    #     self.log("learning_rate", lr, on_epoch=True, on_step=False, sync_dist=True)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr, on_epoch=False, on_step=True, sync_dist=True)
        self.log("epoch", self.current_epoch, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        # TODO: sync update to quarc
        opt = Adam(self.parameters())
        scheduler = OneCycleLR(
            opt,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            div_factor=25,  # FIXME:change back to 25?
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


class AgentFFN(BaseFFN):
    """FFN model for agent prediction with sample weights"""

    def __init__(self, predictor: FFNAgentHead, *args, **kwargs):
        if not isinstance(predictor, FFNAgentHead):
            raise TypeError("AgentFFN requires FFNAgentHead")
        super().__init__(predictor, *args, **kwargs)

    def loss_fn(self, preds: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        loss = self.criterion(preds, targets)
        return (loss * weights).mean()

    def training_step(self, batch, batch_idx):

        FP_inputs, a_inputs, targets, weights = batch
        preds = self(FP_inputs, a_inputs)
        l = self.loss_fn(preds, targets, weights)

        self.log(
            "loss/train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)
        dummy_weights = torch.ones(FP_inputs.shape[0]).to(preds.device)

        val_loss = self.loss_fn(preds, targets, dummy_weights)
        self.log(
            "loss/val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss

    # def configure_optimizers(self):
    #     """For training, count total steps for augmented training data"""
    #     opt = Adam(self.parameters(), lr=self.init_lr)

    #     lr_sched = {
    #         "scheduler": ExponentialLR(optimizer=opt, gamma=self.gamma),
    #         "interval": "epoch",
    #     }

    #     return {
    #         "optimizer": opt,
    #         "lr_scheduler": lr_sched,
    #     }


class AgentFFNWithRxnClass(BaseFFN):
    """FFN model for agent prediction with sample weights"""

    def __init__(
        self,
        predictor: FFNAgentHeadWithRxnClass,
        *args,
        **kwargs,
    ):
        if not isinstance(predictor, FFNAgentHeadWithRxnClass):
            raise TypeError("AgentFFN requires FFNAgentHeadWithRxnClass")
        super().__init__(predictor, *args, **kwargs)

    def loss_fn(self, preds: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
        loss = self.criterion(preds, targets)
        return (loss * weights).mean()

    def forward(self, FP_inputs: Tensor, agent_input: Tensor, rxn_class: Tensor) -> Tensor:
        return self.predictor(FP_inputs, agent_input, rxn_class)

    def training_step(self, batch, batch_idx):

        FP_inputs, a_inputs, targets, weights, rxn_class = batch
        preds = self(FP_inputs, a_inputs, rxn_class)
        l = self.loss_fn(preds, targets, weights)
        self.log(
            "loss/train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets, rxn_class = batch
        preds = self(FP_inputs, a_inputs, rxn_class)
        dummy_weights = torch.ones(FP_inputs.shape[0]).to(preds.device)

        val_loss = self.loss_fn(preds, targets, dummy_weights)
        self.log(
            "loss/val_loss",
            val_loss,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)

            if "multilabel" in metric_name:
                metric = metric_f(F.sigmoid(preds), targets)
            else:
                metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        """For training, count total steps for augmented training data"""
        opt = Adam(self.parameters(), lr=self.init_lr)

        lr_sched = ExponentialLR(optimizer=opt, gamma=0.98)

        return {"optimizer": opt, "lr_scheduler": lr_sched}


class TemperatureFFN(BaseFFN):
    """FFN model for predicting temperature."""

    def __init__(self, predictor: FFNTemperatureHead, *args, **kwargs):
        if not isinstance(predictor, FFNTemperatureHead):
            raise TypeError("TemperatureFFN requires FFNTemperatureHead")
        super().__init__(predictor, *args, **kwargs)


class ReactantAmountFFN(BaseFFN):
    """FFN model for predicting binned reactant amounts."""

    def __init__(self, predictor: FFNReactantAmountHead, *args, **kwargs):
        if not isinstance(predictor, FFNReactantAmountHead):
            raise TypeError("ReactantAmountFFN requires FFNReactantAmountHead")
        super().__init__(predictor, *args, **kwargs)

    def forward(
        self,
        FP_inputs: Tensor,
        agent_input: Tensor,
        FP_reactants: Tensor,
    ) -> Tensor:
        return self.predictor(FP_inputs, agent_input, FP_reactants)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, FP_reactants, targets = batch
        preds = self(
            FP_inputs, a_inputs, FP_reactants
        )  # (batch_size, MAX_NUM_REACTANTS, num_binned_classes)
        preds = preds.view(-1, preds.shape[-1])
        targets = targets.view(-1)

        l = self.loss_fn(preds, targets)
        self.log(
            "loss/train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, FP_reactants, targets = batch
        preds = self(
            FP_inputs, a_inputs, FP_reactants
        )  # (batch_size, MAX_NUM_REACTANTS, num_binned_classes)
        preds = preds.view(
            -1, preds.shape[-1]
        )  # (batch_size * MAX_NUM_REACTANTS, num_binned_classes)
        targets = targets.view(-1)  # (batch_size * MAX_NUM_REACTANTS)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "loss/val_loss",
            val_loss,
            batch_size=len(batch[0]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss


class AgentAmountFFN(BaseFFN):
    """FFN model for predicting binned agent amounts"""

    def __init__(self, predictor: FFNAgentAmountHead, *args, **kwargs):
        if not isinstance(predictor, FFNAgentAmountHead):
            raise TypeError("AgentAmountFFN requires FFNAgentAmountHead")
        super().__init__(predictor, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)  # (batch_size, num_classes, num_bins)

        # flatten preds and targets
        preds = preds.view(-1, preds.shape[-1])  # (batch_size * num_classes, num_bins)
        targets = targets.view(-1)

        l = self.loss_fn(preds, targets)
        self.log(
            "loss/train_loss",
            l,
            batch_size=len(batch[0]),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return l

    def validation_step(self, batch, batch_idx):
        FP_inputs, a_inputs, targets = batch
        preds = self(FP_inputs, a_inputs)

        # flatten preds and targets
        preds = preds.view(-1, preds.shape[-1])  # (batch_size * num_classes, num_bins)
        targets = targets.view(-1)

        val_loss = self.loss_fn(preds, targets)
        self.log(
            "loss/val_loss",
            val_loss,
            batch_size=len(batch[0]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for metric_name, metric_f in self.metrics.items():
            if isinstance(metric_f, nn.Module):
                metric_f = metric_f.to(self.device)
            metric = metric_f(preds, targets)
            self.log(metric_name, metric, batch_size=len(batch[0]), on_epoch=True, sync_dist=True)
        return val_loss

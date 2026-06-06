from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import random
from lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, MeanMetric
from ..metrics.custom_metrics import masked_mape_np, masked_mae_np, masked_mse_np, masked_rmse_np
from typing import List

class BaseModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_names: List[str],
        compile: bool,
        scale: bool,
        scaler: Any,
        scheduler: torch.optim.lr_scheduler = None,
        if_sample: bool = False,
        sample_size: int = 0,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net() if not if_sample else net(num_channels=sample_size)
        self.scaler = scaler
        self.if_sample = if_sample
        self.sample_size = sample_size

        # metric objects for calculating and averaging accuracy across batches
        self.train_mse = MeanMetric()
        self.train_mae = MeanMetric()
        self.train_mape = MeanMetric()

        self.val_mse = nn.ModuleList([MeanMetric() for _ in range(2)])
        self.val_mae = nn.ModuleList([MeanMetric() for _ in range(2)])
        self.val_mape = nn.ModuleList([MeanMetric() for _ in range(2)])

        self.test_mse = MeanMetric()
        self.test_mae = MeanMetric()
        self.test_mape = MeanMetric()

        # for averaging loss across batches
        self.hparams.loss_names += ['loss_total']
        self.losses = nn.ModuleDict({loss_name: MeanMetric() for loss_name in self.hparams.loss_names})

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()
        self.test_mse_best = MinMetric()
        self.test_mae_best = MinMetric()

    def forward(self, x: torch.Tensor,
                y: torch.Tensor,
                x_mark: Dict[str, torch.Tensor],
                y_mark: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.net(x, y, x_mark, y_mark)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for item in self.val_mse:
            item.reset()
        for item in self.val_mae:
            item.reset()
        for item in self.val_mape:
            item.reset()
        self.val_mae_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[Dict[str, Any], torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y, x_mark, y_mark = batch

        result = self.forward(x, y, x_mark, y_mark)
        return result, y

    def on_train_epoch_start(self) -> None:
        for item in self.val_mse:
            item.reset()
        for item in self.val_mae:
            item.reset()
        for item in self.val_mape:
            item.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y, x_mark, y_mark = batch
        if self.hparams.scale:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)
        batch = (x, y, x_mark, y_mark)
        if self.if_sample:
            _, N, _ = x.shape
            index = np.stack([random.randint(0, N - 1) for _ in range(self.sample_size)])
            x = x[:, index, :]
            y = y[:, index, :]
            batch = (x, y, x_mark, y_mark)
        result, targets = self.model_step(batch)
        preds = result['y_hat']

        # update and log metrics
        for loss_name in self.hparams.loss_names:
            if loss_name not in result:
                raise ValueError(f"Loss name {loss_name} not found in result")
            self.losses[loss_name](result[loss_name])

        self.train_mse(masked_mse_np(targets, preds))
        self.train_mae(masked_mae_np(targets, preds))
        self.train_mape(masked_mape_np(targets, preds))

        return result['loss_total']

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        for loss_name in self.hparams.loss_names:
            self.log(f"train/{loss_name}", self.losses[loss_name], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mape", self.train_mape, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, dataloader_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        label = {0: "val", 1: "test"}
        x, y, x_mark, y_mark = batch
        if self.hparams.scale:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)
        batch = (x, y, x_mark, y_mark)
        if self.if_sample:
            _, N, _ = x.shape
            index = np.arange(
                (N + self.sample_size - 1) // self.sample_size * self.sample_size)
            index %= N
            # index[:N] = np.random.permutation(N)

            index = index.reshape(-1, self.sample_size)
            batch_x_sample = x[:, index, :]
            batch_y_sample = y[:, index, :]
            targets = []
            results = []
            for sample_id in range(batch_x_sample.shape[1]):
                batch_x_i = batch_x_sample[:, sample_id, :, :]
                batch_y_i = batch_y_sample[:, sample_id, :, :]
                batch_i = (batch_x_i, batch_y_i, x_mark, y_mark)
                result_i, target_i = self.model_step(batch_i)
                results.append(result_i)
                targets.append(target_i)
            preds = torch.cat([r['y_hat'] for r in results], dim=1)[:, :N, :].contiguous()
            targets = torch.cat(targets, dim=1)[:, :N, :].contiguous()
        else:
            result, targets = self.model_step(batch)
            preds = result['y_hat']

        # update and log metrics
        # masking out zeros in targets
        self.val_mse[dataloader_idx](masked_mse_np(targets, preds))
        self.val_mae[dataloader_idx](masked_mae_np(targets, preds))
        self.val_mape[dataloader_idx](masked_mape_np(targets, preds))
        

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        label = {0: "val", 1: "test"}
        for idx in range(2):
            self.log(f"{label[idx]}/mse", self.val_mse[idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{label[idx]}/mae", self.val_mae[idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            self.log(f"{label[idx]}/mape", self.val_mape[idx], on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        
        mae = self.val_mae[0].compute()  # get current val acc
        if mae < self.val_mae_best.compute():
            self.log("test/mse_best", self.val_mse[1].compute(), sync_dist=True)
            self.log("test/mae_best", self.val_mae[1].compute(), sync_dist=True)
            self.log("val/mse_best", self.val_mse[0].compute(), sync_dist=True)
        self.val_mae_best(mae)  # update best so far val acc
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)
    
    def on_test_start(self) -> None:
        self.test_mse.reset()
        self.test_mae.reset()
        self.test_mape.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        x, y, x_mark, y_mark = batch
        if self.hparams.scale:
            x = self.scaler.transform(x)
            y = self.scaler.transform(y)
        batch = (x, y, x_mark, y_mark)
        if self.if_sample:
            _, N, _ = x.shape
            index = np.arange(
                (N + self.sample_size - 1) // self.sample_size * self.sample_size)
            index %= N
            # index[:N] = np.random.permutation(N)

            index = index.reshape(-1, self.sample_size)
            batch_x_sample = x[:, index, :]
            batch_y_sample = y[:, index, :]
            targets = []
            results = []
            for sample_id in range(batch_x_sample.shape[1]):
                batch_x_i = batch_x_sample[:, sample_id, :, :]
                batch_y_i = batch_y_sample[:, sample_id, :, :]
                batch_i = (batch_x_i, batch_y_i, x_mark, y_mark)
                result_i, target_i = self.model_step(batch_i)
                results.append(result_i)
                targets.append(target_i)
            preds = torch.cat([r['y_hat'] for r in results], dim=1)[:, :N, :].contiguous()
            targets = torch.cat(targets, dim=1)[:, :N, :].contiguous()
        else:
            result, targets = self.model_step(batch)
            preds = result['y_hat']
        # update and log metrics
        self.test_mse(masked_mse_np(preds, targets))
        self.test_mae(masked_mae_np(preds, targets))
        self.test_mape(masked_mape_np(preds, targets))
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss_total",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = LitModule(None, None, None, None)

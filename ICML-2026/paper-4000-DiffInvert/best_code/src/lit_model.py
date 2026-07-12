# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,arguments-differ,unused-argument
from typing import Union, Any, Optional
import torch
from torch import nn, Tensor
import lightning.pytorch as pl
from tqdm import tqdm

from .groups import ConnectedMatrixLieGroup
from .group_sampler import GroupSampler
from .group_optimizer import GroupOptimizer


class LitModel(pl.LightningModule):
    def __init__(
        self,
        inner_model: nn.Module,
        group: ConnectedMatrixLieGroup,
        outer_model: Optional[Union[GroupSampler, GroupOptimizer]],
        task: str,
        ensemble: bool
    ):
        super().__init__()
        assert task in [
            "training",
            "test/image_classification",
            "test/image2image"
        ]
        self.invariant_tasks = ["test/image_classification"]
        self.equivariant_tasks = ["test/image2image"]

        self.inner_model = inner_model
        self.group = group
        self.outer_model = outer_model
        self.task = task
        self.ensemble = ensemble

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.test_total_correct = 0
        self.test_total_seen = 0
        self.test_step_outputs = []

        self.saved_x_transformed_all = [] 

    @staticmethod
    def _repeat_and_flatten(x: Tensor, repeats: int) -> Tensor:
        bsize = x.shape[0]
        if repeats == 1:
            return x.clone()
        return x[:, None].expand(bsize, repeats, *x.shape[1:]).flatten(0, 1)

    @staticmethod
    def _deflatten(x: Tensor, bsize: int, repeats: int) -> Tensor:
        return x.view(bsize, repeats, *x.shape[1:])

    def forward(self, x: Tensor, y: Optional[Tensor]) -> Tensor:
        assert self.inner_model is not None, "Inner model is required."

        if self.outer_model is None:
            preds = self.inner_model(x)
            return preds

        g = self.outer_model(x, y)

        if self.ensemble:
            bsize = x.shape[0]
            num_hypothesis = g.shape[0] // bsize

            x = self._repeat_and_flatten(x, num_hypothesis)
            y = self._repeat_and_flatten(y, num_hypothesis) if y is not None else None

            g_inv = self.group.inverse(g)
            x_transformed = self.group.act(g_inv, x)
            if not self.training:
                self.saved_x_transformed_all.append(x_transformed.detach().cpu())
            preds_transformed = self.inner_model(x_transformed)

            if self.task in self.invariant_tasks:
                preds = preds_transformed
            else:
                assert self.task in self.equivariant_tasks
                preds = self.group.act(g, preds_transformed)

            preds = self._deflatten(preds, bsize, num_hypothesis)
            preds = preds.mean(dim=1)

        else:
            g_inv = self.group.inverse(g)
            x_transformed = self.group.act(g_inv, x)
            if not self.training:
                self.saved_x_transformed_all.append(x_transformed.detach().cpu())
            preds_transformed = self.inner_model(x_transformed)

            if self.task in self.invariant_tasks:
                preds = preds_transformed
            else:
                assert self.task in self.equivariant_tasks
                preds = self.group.act(g, preds_transformed)

        return preds

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch: Any, batch_idx: int):
        assert self.outer_model is not None, "Outer model is required for training."

        if self.task == "training":
            x, y = batch
            loss = self.outer_model.compute_loss(x, y)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return loss

        raise NotImplementedError(f"training_step not implemented for task {self.task}")

    def validation_step(self, batch: Any, batch_idx: int):
        assert self.outer_model is not None, "Outer model is required for training."

        if self.task == "training":
            x, y = batch
            loss = self.outer_model.compute_loss(x, y)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        else:
            raise NotImplementedError(f"validation_step not implemented for task {self.task}")

    def test_step(self, batch: Any, batch_idx: int):
        if self.task == "training":
            assert self.outer_model is not None, "Outer model is required for training."

            x, y = batch
            loss = self.outer_model.compute_loss(x, y)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.task == "test/image_classification":
            x, y = batch

            preds = self.forward(x, y).argmax(dim=1)
            accuracy = (preds == y).float().mean()

            correct = (preds == y).sum().item()
            batch_size = y.shape[0]
            self.test_total_correct += correct
            self.test_total_seen += batch_size
            
            if self.outer_model.__class__.__name__ == "FoCalOptimizer":
                cumulative_acc = self.test_total_correct / self.test_total_seen
                tqdm.write(f"[Progress] cumulative accuracy: {cumulative_acc * 100:.2f}%")
                tqdm.write(f"[Progress] total evaluated samples: {self.test_total_seen}")
            else:
                print(f"batch accuracy: \t{accuracy * 100:.2f}%")

            self.test_step_outputs.append({
                "test/correct": (preds == y).sum(),
                "test/bsize": y.shape[0]
            })

            all_correct = sum(output["test/correct"] for output in self.test_step_outputs)
            all_bsize = sum(output["test/bsize"] for output in self.test_step_outputs)

            accuracy = all_correct / all_bsize

            print(f"average accuracy: \t{accuracy * 100:.2f}%")

        else:
            raise NotImplementedError(f"test_step not implemented for task {self.task}")

    def on_test_end(self):
        if self.task == "test/image_classification":

            all_correct = sum(output["test/correct"] for output in self.test_step_outputs)
            all_bsize = sum(output["test/bsize"] for output in self.test_step_outputs)

            accuracy = all_correct / all_bsize

            print(f"test accuracy: {accuracy * 100:.2f}%")

        else:
            raise NotImplementedError(f"on_test_end not implemented for task {self.task}")

        self.test_step_outputs.clear()

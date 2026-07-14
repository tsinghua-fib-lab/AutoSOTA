#!/usr/bin/env python3
"""
Description: This script defines a PyTorch Lightning trainer for
a masked-autoencoder model. It handles model training, validation. The
trainer processes OpenStreetMap basemaps at various patch sizes.
"""
import os
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule


class MAELitModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Callable,
        scheduler: Callable,
        plot_dir: str = "plots_mae",
    ) -> None:
        """
        LightningModule for Masked Autoencoders (MAE) with training,
        validation, and visualization support.

        Parameters
        ----------
        model : torch.nn.Module
            The MAE model instance.
        optimizer : callable
            Optimizer constructor accepting model parameters.
        scheduler : callable
            Learning rate scheduler constructor.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer(self.model.parameters())
        self.scheduler = scheduler(self.optimizer)
        self.plot_dir = plot_dir
        self.validation_losses = {}
        self.sample_indices = {}

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Executes one training step.

        Parameters
        ----------
        batch : torch.Tensor
            Input images batch.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Training loss.
        """
        imgs = batch
        loss = self.model(imgs, return_loss_only=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Executes one validation step.

        Parameters
        ----------
        batch : torch.Tensor
            Input images batch.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader in use (default is 0).

        Returns
        -------
        torch.Tensor
            Validation loss.
        """
        imgs = batch
        loss = self.model(imgs, return_loss_only=True)

        self.log(
            f"val_loss_dataloader_{dataloader_idx}",
            loss,
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Logs validation loss from the first dataloader only and plots images.
        """
        self.plot_images()

    def plot_images(self) -> None:
        """
        Generates plots of original, masked, reconstructed, and combined
        images.
        """
        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataloader().dataset
        val_datasets = [dl.dataset for dl in datamodule.val_dataloader()]

        gen = torch.Generator().manual_seed(1)
        self.sample_indices["train"] = torch.randperm(
            len(train_dataset), generator=gen
        )[:5].tolist()

        for idx, val_dataset in enumerate(val_datasets):
            self.sample_indices[f"val_{idx}"] = torch.randperm(
                len(val_dataset), generator=gen
            )[:5].tolist()

        self.model.eval()
        with torch.no_grad():
            imgs_tr = torch.stack(
                [train_dataset[i] for i in self.sample_indices["train"]]
            ).to(self.device)

            imgs_vl = []
            for idx, val_dataset in enumerate(val_datasets):
                indices = self.sample_indices[f"val_{idx}"]
                imgs = torch.stack([val_dataset[i] for i in indices])
                imgs_vl.append(imgs.to(self.device))

            rec_tr, masked_tr = self.model(imgs_tr, return_loss_only=False)
            rec_vl, masked_vl = [], []

            for imgs in imgs_vl:
                rec, masked = self.model(imgs, return_loss_only=False)
                rec_vl.append(rec)
                masked_vl.append(masked)

            all_orig = torch.cat([imgs_tr] + imgs_vl, dim=0)
            all_masked = torch.cat([masked_tr] + masked_vl, dim=0)
            all_rec = torch.cat([rec_tr] + rec_vl, dim=0)

            orig_np = all_orig.cpu().numpy().transpose(0, 2, 3, 1)
            masked_np = all_masked.cpu().numpy().transpose(0, 2, 3, 1)
            rec_np = all_rec.cpu().numpy().transpose(0, 2, 3, 1)

            mask = np.all(masked_np == 0, axis=-1, keepdims=True)
            combined_np = np.where(mask, rec_np, masked_np)

            n_sections = 1 + len(val_datasets)
            imgs_per_sec = 5
            n_imgs = orig_np.shape[0]

            fig, axes = plt.subplots(n_imgs, 4, figsize=(12, 3 * n_imgs))
            sec_titles = [
                "Train",
                "Validation in-Region",
                "Validation out-of-Region",
            ]

            # Clip the images to [0, 1] for display
            orig_np = np.clip(orig_np, 0, 1)
            masked_np = np.clip(masked_np, 0, 1)
            rec_np = np.clip(rec_np, 0, 1)
            combined_np = np.clip(combined_np, 0, 1)

            for i in range(n_imgs):
                axes[i, 0].imshow(orig_np[i])
                axes[i, 0].set_title("Original")
                axes[i, 0].axis("off")

                axes[i, 1].imshow(masked_np[i])
                axes[i, 1].set_title("Masked")
                axes[i, 1].axis("off")

                axes[i, 2].imshow(rec_np[i])
                axes[i, 2].set_title("Reconstructed")
                axes[i, 2].axis("off")

                axes[i, 3].imshow(combined_np[i])
                axes[i, 3].set_title("Combined")
                axes[i, 3].axis("off")

                if i % imgs_per_sec == 0:
                    sec_idx = i // imgs_per_sec
                    fig.text(
                        0.5,
                        1 - (i / (n_sections * imgs_per_sec)),
                        sec_titles[sec_idx],
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        fontweight="bold",
                    )
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.35)
            os.makedirs(self.plot_dir, exist_ok=True)
            plot_path = os.path.join(
                self.plot_dir, f"epoch_{self.current_epoch}.png"
            )
            plt.savefig(plot_path)
            plt.close()

    def configure_optimizers(self) -> Dict[str, object]:
        """
        Configures and returns optimizer and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary with optimizer and scheduler.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

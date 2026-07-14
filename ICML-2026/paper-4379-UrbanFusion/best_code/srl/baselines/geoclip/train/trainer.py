#!/usr/bin/env python3
"""
Description: Implementation of the GeoCLIP model for training and evaluation
using PyTorch Lightning. This module is based on the original
SatCLIP codebase https://github.com/VicenteVivan/geo-clip.
"""
import os

import torch
from lightning import LightningModule
from torch import nn

from scripts.lr_schedule.cosine_schedule import CosineSchedule
from srl.baselines.geoclip.model.GeoCLIP import GeoCLIP


class GeoCLIPLightningModule(LightningModule):
    def __init__(
        self,
        from_pretrained: bool = True,
        precomputed_features=False,
        lr=3e-5,
        weight_decay=1e-6,
        scheduler_step_size=1,
        scheduler_gamma=0.87,
        queue_size: int = 4096,
        plot_representation_space: dict = {},
        synthetic_experiment: bool = False,
    ) -> None:
        """
        Initializes the GeoCLIPLightningModule with the specified parameters.

        Parameters
        ----------
        from_pretrained : bool, optional
            Whether to load a pre-trained model (on MP-16 dataset),
            by default True.
        precomputed_features : bool, optional
            Whether to use precomputed features of PP2-M dataset (no forward
            pass needed through the backbones), by default False.
        lr : float, optional
            Learning rate for the optimizer, by default 3e-5.
        weight_decay : float, optional
            Weight decay for the optimizer, by default 1e-6.
        scheduler_step_size : int, optional
            Step size for the learning rate scheduler, by default 1.
        scheduler_gamma : float, optional
            Gamma value for the learning rate scheduler, by default 0.87.
        queue_size : int, optional
            Size of the queue for the memory bank, by default 4096.
        plot_representation_space : dict, optional
            Dictionary containing parameters for plotting the representation
            space, by default an empty dictionary.
        synthetic_experiment : bool, optional
            Whether to use synthetic data for training (evaluating partial
            information decomposition), by default False.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.queue_size = queue_size
        self.plot_representation_space = plot_representation_space
        self.synthetic_experiment = synthetic_experiment

        self.model = GeoCLIP(
            from_pretrained=from_pretrained,
            precomputed_features=precomputed_features,
            queue_size=queue_size,
            synthetic_experiment=synthetic_experiment,
        )
        self.loss_fun = nn.CrossEntropyLoss()
        self.precomputed_features = precomputed_features

    def _process_batch(self, batch: dict) -> list:
        """
        Processes a batch dictionary by moving each tensor to the device and
        tokenizing text. Extracts the batch size from the first tensor.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.

        Returns
        -------
        processed : list
            A list with processed elements in the same order as they appear in
            the input dict.
        batch_size : int
            Batch size of the input batch.
        """
        processed = []
        batch_size = None
        for key, value in batch.items():
            if key.lower() == "coords_original":
                pass
            elif key.lower() == "gsv_img":
                pass
            else:
                processed.append(value.to(self.device))
                if batch_size is None:
                    batch_size = value.size(0)

        return processed, batch_size

    def common_step(self, batch: dict, batch_idx: int) -> tuple:
        """
        Common step for training, validation, and testing. It processes the
        batch, extracts the necessary inputs, and computes the loss.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the current batch in the dataloader.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the current batch.
        t_points : torch.Tensor
            Tensor containing the GPS coordinates or time points for the batch.
        """
        # Unpack and process batch
        inputs, batch_size = self._process_batch(batch)
        t_points = inputs[0]
        images = inputs[1]
        if self.precomputed_features:
            t_points = t_points[:, -2:]

        # Get current queue
        gps_queue = self.model.get_gps_queue().to(images.device)

        # Compose all GPS (current + negatives)
        all_gps = torch.cat([t_points, gps_queue], dim=0)

        # Forward pass
        # Expect shape [batch_size, total_gps]
        logits_per_image = self.model(images, all_gps)
        targets = torch.arange(batch_size, device=images.device)
        loss = self.loss_fun(logits_per_image, targets)

        return loss, t_points

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for the GeoCLIP model. It processes the batch, computes
        the loss, and updates the queue with the GPS batch.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the current batch in the dataloader.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the current batch.
        """

        loss, gps_batch = self.common_step(batch, batch_idx)

        # Update the queue (no_grad because the queue is not part of gradient
        # flow)
        with torch.no_grad():
            self.model.dequeue_and_enqueue(gps_batch)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step for the GeoCLIP model. It processes the batch,
        computes the loss, and logs it.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the current batch in the dataloader.
        dataloader_idx : int, optional
            Index of the dataloader, by default 0.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the current batch.
        """
        loss, _ = self.common_step(batch, batch_idx)
        if self.synthetic_experiment:
            batch_size = batch["coords"].shape[0]
        else:
            batch_size = batch["SVI"].shape[0]
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.log(
                f"val_loss_dataloader_{dataloader_idx}",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        else:
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Test step for the GeoCLIP model. It processes the batch, computes
        the loss, and logs it.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the current batch in the dataloader.
        dataloader_idx : int, optional
            Index of the dataloader, by default 0.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the current batch.
        """
        loss, _ = self.common_step(batch, batch_idx)
        if self.synthetic_experiment:
            batch_size = batch["coords"].shape[0]
        else:
            batch_size = batch["SVI"].shape[0]
        if (
            self.trainer.datamodule.dataset_name == "PlacePulse2"
            or self.trainer.datamodule.dataset_name == "SyntheticPID"
        ):
            self.log(
                f"test_loss_dataloader_{dataloader_idx}",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        else:
            self.log(
                "test_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
                sync_dist=True,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        """
        Saves the representations of the validation set at the end of the
        validation epoch.
        """

        representations = None

        # Plot the representation space if requested
        if (
            self.plot_representation_space["plot_val"]
            and self.current_epoch
            % self.plot_representation_space["val_n_epochs"]
            == 0
        ):
            if self.plot_representation_space["save_representations_val"]:
                if (
                    self.trainer.datamodule.dataset_name == "PlacePulse2"
                    or self.trainer.datamodule.dataset_name == "SyntheticPID"
                ):
                    val_loaders = self.trainer.datamodule.val_dataloader()
                    if not isinstance(val_loaders, list):
                        val_loaders = [val_loaders]

                    for dl_idx, val_loader in enumerate(val_loaders):
                        for i in range(
                            len(self.trainer.datamodule.modalities)
                        ):
                            representations, concatenated_modalities = (
                                self.create_representations(
                                    val_loader,
                                    modality=i,
                                    not_modality_list=[
                                        "coords",
                                        "SVI",
                                        "sentinel2",
                                        "OSM",
                                        "text",
                                    ],
                                )
                            )
                            if not os.path.exists(
                                self.plot_representation_space["save_dir"]
                            ):
                                os.makedirs(
                                    self.plot_representation_space["save_dir"]
                                )
                            torch.save(
                                representations,
                                f"{self.plot_representation_space['save_dir']}"
                                f"/representations_val_epoch_"
                                f"{self.current_epoch}_"
                                f"modality_{i}_dl_{dl_idx}.pt",
                            )
                            save_path = (
                                f"{self.plot_representation_space['save_dir']}"
                                f"/modalities_val_dl_{dl_idx}.pt"
                            )
                            (concatenated_modalities_save) = (
                                concatenated_modalities.get("gsv_img", None)
                            )
                            if (
                                concatenated_modalities_save is not None
                                and not os.path.exists(save_path)
                            ):
                                torch.save(
                                    concatenated_modalities_save, save_path
                                )

    def on_test_epoch_end(self) -> None:
        """
        Calculates and saves the representations of the test set at the end
        of the test epoch.
        """

        representations = None
        if self.plot_representation_space["plot_test"]:
            if (
                self.trainer.datamodule.dataset_name == "PlacePulse2"
                or self.trainer.datamodule.dataset_name == "SyntheticPID"
            ):
                test_loaders = self.trainer.datamodule.test_dataloader()
                if not isinstance(test_loaders, list):
                    test_loaders = [test_loaders]

                for dl_idx, val_loader in enumerate(test_loaders):
                    for i in range(3):
                        representations, concatenated_modalities = (
                            self.create_representations(
                                val_loader,
                                selected_modality=i,
                                not_modality_list=[
                                    "coords",
                                    "SVI",
                                    "sentinel2",
                                    "OSM",
                                    "text",
                                ],
                            )
                        )
                        if not os.path.exists(
                            self.plot_representation_space["save_dir"]
                        ):
                            os.makedirs(
                                self.plot_representation_space["save_dir"]
                            )
                        torch.save(
                            representations,
                            f"{self.plot_representation_space['save_dir']}/"
                            f"representations_test_epoch_{self.current_epoch}_"
                            f"modality_{i}_dl_{dl_idx}.pt",
                        )
                        save_path = (
                            f"{self.plot_representation_space['save_dir']}"
                            f"/modalities_test_dl_{dl_idx}.pt"
                        )
                        (concatenated_modalities_save) = (
                            concatenated_modalities.get("gsv_img", None)
                        )
                        if (
                            concatenated_modalities_save is not None
                            and not os.path.exists(save_path)
                        ):
                            torch.save(concatenated_modalities_save, save_path)

        if self.trainer.datamodule.coordinate_predictions is not None:
            for dl_idx, val_loader in enumerate(
                self.trainer.datamodule.coordinate_dataloaders
            ):
                representations, concatenated_modalities = (
                    self.create_representations(
                        val_loader,
                        selected_modality=0,
                        not_modality_list=[
                            "coords",
                            "SVI",
                            "sentinel2",
                            "OSM",
                            "text",
                        ],
                    )
                )
                dataset_name = val_loader.dataset.dataset_name

                if not os.path.exists(
                    self.plot_representation_space["save_dir"]
                ):
                    os.makedirs(self.plot_representation_space["save_dir"])
                torch.save(
                    representations,
                    f"{self.plot_representation_space['save_dir']}/"
                    f"representations_test_epoch_{self.current_epoch}_"
                    f"modality_{0}_dl_{dataset_name}.pt",
                )
                save_path = (
                    f"{self.plot_representation_space['save_dir']}"
                    f"/modalities_test_dl_{dataset_name}.pt"
                )
                concatenated_modalities_save = concatenated_modalities.get(
                    "gsv_img", None
                )
                if (
                    concatenated_modalities_save is not None
                    and not os.path.exists(save_path)
                ):
                    torch.save(concatenated_modalities_save, save_path)

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        modality: int = 0,
    ) -> torch.Tensor:
        """
        Predict step for the GeoCLIP model. It processes the batch and
        returns the representations for the specified modality.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the current batch in the dataloader.
        modality : int, optional
            Modality index to select the representation (0 for GPS, 1 for
            image, 2 for both), by default 0.

        Returns
        -------
        reps : torch.Tensor
            Representations for the specified modality.
        """
        if "gsv_img" in batch:
            del batch["gsv_img"]
        inputs, _ = self._process_batch(batch)
        images = inputs[1]
        t_points = inputs[0]
        if self.precomputed_features:
            t_points = t_points[:, -2:]
        if modality == 0:
            reps = self.model.location_encoder(t_points)
        elif modality == 1:
            reps = self.model.image_encoder(images)
        elif modality == 2:
            # If modality is 2, we return both image and GPS representations
            image_reps = self.model.image_encoder(images)
            gps_reps = self.model.location_encoder(t_points)
            reps = torch.cat((image_reps, gps_reps), dim=1)
        return reps

    def create_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        selected_modality: int = None,
        not_modality_list: list = [],
    ) -> tuple:
        """
        Creates representations for the specified modality from the given
        data loader. It iterates over the data loader, collects the
        representations, and concatenates them along the batch dimension.
        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing the dataset to create representations from.
        selected_modality : int, optional
            Modality index to select the representation (0 for GPS, 1 for
            image), by default None.
        not_modality_list : list, optional
            List of modalities to exclude from the concatenation, by default
            an empty list.
        Returns
        -------
        representations_cat : torch.Tensor
            Concatenated representations for the specified modality.
        concatenated_modalities : dict
            Dictionary containing concatenated modalities, excluding those in
            not_modality_list.
        """
        all_representations = []
        modality_inputs = {}

        # Only if this attribute exists in the dataset
        if hasattr(data_loader.dataset, "return_filename"):
            # Set it to True to return filenames
            data_loader.dataset.return_filename = True

        # Iterate over the dataloader
        for batch_idx, batch in enumerate(data_loader):
            # For every modality in the batch, store its data.
            for modality, value in batch.items():
                if modality not in modality_inputs:
                    modality_inputs[modality] = []
                modality_inputs[modality].append(value)
            reps = self.predict_step(batch, batch_idx, selected_modality)
            all_representations.append(reps)

        # Concatenate all representations along the batch dimension
        representations_cat = (
            torch.cat(all_representations, dim=0)
            if all_representations
            else None
        )

        # For each modality, if the values are tensors, concatenate them;
        # otherwise, keep them as a list.
        concatenated_modalities = {}
        for modality, values in modality_inputs.items():
            # Only add modalities that are in modality list
            if modality in not_modality_list:
                continue
            else:
                if isinstance(values[0], torch.Tensor):
                    concatenated_modalities[modality] = torch.cat(
                        values, dim=0
                    )
                else:
                    concatenated_modalities[modality] = values

        if hasattr(data_loader.dataset, "return_filename"):
            # Reset it to False to not return filenames in the future
            data_loader.dataset.return_filename = False

        return representations_cat, concatenated_modalities

    def configure_optimizers(self) -> dict:
        """
        Configures the optimizer and learning rate scheduler for the model.

        Returns
        -------
        dict
            A dictionary containing the optimizer and learning rate scheduler.
        """

        # Configure optimizer for PP2-M dataset
        if self.synthetic_experiment is False:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
            # Lightning expects a dict for schedulers if you want to step every
            # epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # step every epoch
                    "frequency": 1,
                },
            }

        # Configure optimizer for synthetic data experiment
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.0003,
                momentum=0.9,
                weight_decay=0.0,
            )
            scheduler = CosineSchedule(
                optimizer,
                steps_warmup=0,
                steps_total=23500,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # step every batch
                    "frequency": 1,
                },
            }

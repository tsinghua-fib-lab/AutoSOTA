#!/usr/bin/env python3
"""
Description: Implementation of the GAIR model for training and evaluation
using PyTorch Lightning. Our own implementation of the GAIR learning
framework, with using GeoCLIP location encoder and street-view image encoder,
and remote sensing encoder from SatCLIP.

Code is loosely based on: https://github.com/VicenteVivan/geo-clip
"""
import os

import torch
from lightning import LightningModule
from torch import nn

from scripts.lr_schedule.cosine_schedule import CosineSchedule
from srl.baselines.GAIR.model.GAIR import GAIR


class GAIRLightningModule(LightningModule):
    def __init__(
        self,
        precomputed_features: bool = False,
        lr: float = 1.5e-6,
        weight_decay: float = 0.01,
        queue_size: int = 4096,
        secl_weight: int = 1.0,
        plot_representation_space: dict = {},
        synthetic_experiment: bool = False,
    ) -> None:
        """
        Initializes the GAIRLightningModule with the specified parameters.

        Parameters
        ----------
        precomputed_features : bool, optional
            If True, uses precomputed features of the PP2-M dataset (without
            forward pass through the backbones). Defaults to False.
        lr : float, optional
            Learning rate for the optimizer. Defaults to 1.5e-6.
        weight_decay : float, optional
            Weight decay for the optimizer. Defaults to 0.01.
        queue_size : int, optional
            Size of the queue for the memory bank. Defaults to 4096.
        secl_weight : float, optional
            Weight for the secondary loss. Defaults to 1.0.
        plot_representation_space : dict, optional
            Dictionary containing parameters for plotting the representation
            space. It can include keys like 'plot_val', 'val_n_epochs',
            'save_representations_val', 'plot_test', and 'save_dir'.
            Defaults to an empty dictionary.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.secl_weight = secl_weight
        self.queue_size = queue_size
        self.plot_representation_space = plot_representation_space

        self.model = GAIR(
            queue_size=queue_size,
            precomputed_features=precomputed_features,
            synthetic_experiment=synthetic_experiment,
        )
        self.loss_fun = nn.CrossEntropyLoss()
        self.precomputed_features = precomputed_features
        self.synthetic_experiment = synthetic_experiment

    def _process_batch(self, batch: dict) -> list:
        """
        Processes a batch dictionary by moving each tensor to the device and
        tokenizing text. Extracts the batch size from the first tensor.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities (e.g., 'image', 'numeric',
            'text').

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

    def common_step(self, batch, batch_idx):
        # Unpack and process batch
        inputs, batch_size = self._process_batch(batch)
        t_points = inputs[0]
        svi_images = inputs[1]
        rs_images = inputs[2]
        if self.precomputed_features:
            t_points = t_points[:, -2:]

        # Get current queue
        gps_queue = self.model.get_gps_queue().to(svi_images.device)

        # Compose all GPS (current + negatives)
        all_gps = torch.cat([t_points, gps_queue], dim=0)

        # Forward pass
        (logits_SVI_location, logits_RS_location, logits_RS_SVIS) = self.model(
            svi_images, rs_images, all_gps
        )

        # Get labels
        targets = torch.arange(batch_size, device=svi_images.device)

        # Compute loss
        secl_svi = self.loss_fun(logits_SVI_location, targets)
        secl_rs = self.loss_fun(logits_RS_location, targets)
        secl = (secl_svi + secl_rs) / 2
        incl_RS = self.loss_fun(logits_RS_SVIS, targets)
        incl_SVI = self.loss_fun(logits_RS_SVIS.t(), targets)
        incl = (incl_RS + incl_SVI) / 2
        loss = incl + self.secl_weight * secl

        return loss, t_points

    def training_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Performs a training step by computing the loss and updating the model.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data for the training step.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The computed loss for the training step.
        """
        loss, gps_batch = self.common_step(batch, batch_idx)

        # Update the GPS queue
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
        Performs a validation step by computing the loss and logging it.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data for the validation step.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (default is 0).

        Returns
        -------
        torch.Tensor
            The computed loss for the validation step.
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
        Performs a test step by computing the loss and logging it.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data for the test step.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the dataloader (default is 0).

        Returns
        -------
        torch.Tensor
            The computed loss for the test step.
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
        Calculates and saves the representations at the end of the
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
        Calculates and saves the representations at the end of the
        test epoch.
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
                    for i in range(7):
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
        Performs a prediction step by extracting representations for the
        specified modality from the batch.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data for the prediction step.
        batch_idx : int
            Index of the current batch.
        modality : int, optional
            Modality index to extract representations. Defaults to 0.
        Returns
        -------
        torch.Tensor
            The extracted representations for the specified modality.
        """
        if "gsv_img" in batch:
            del batch["gsv_img"]
        inputs, _ = self._process_batch(batch)
        rs_images = inputs[2]
        svi_images = inputs[1]
        t_points = inputs[0]
        if self.precomputed_features:
            t_points = t_points[:, -2:]
        if modality == 0:
            reps = self.model.location_encoder(t_points)
        elif modality == 1:
            reps = self.model.svi_encoder(svi_images)
        elif modality == 2:
            reps = self.model.encode_rs(rs_images)
        elif modality == 3:
            # Concatenate SVI and RS representations
            svi_reps = self.model.svi_encoder(svi_images)
            rs_reps = self.model.encode_rs(rs_images)
            reps = torch.cat([svi_reps, rs_reps], dim=1)
        elif modality == 4:
            # Concatenate SVI, and Coords
            svi_reps = self.model.svi_encoder(svi_images)
            coords_reps = self.model.location_encoder(t_points)
            reps = torch.cat([svi_reps, coords_reps], dim=1)
        elif modality == 5:
            # Concatenate RS, and Coords
            rs_reps = self.model.encode_rs(rs_images)
            coords_reps = self.model.location_encoder(t_points)
            reps = torch.cat([rs_reps, coords_reps], dim=1)
        elif modality == 6:
            # Concatenate SVI, RS, and Coords
            svi_reps = self.model.svi_encoder(svi_images)
            rs_reps = self.model.encode_rs(rs_images)
            coords_reps = self.model.location_encoder(t_points)
            reps = torch.cat([svi_reps, rs_reps, coords_reps], dim=1)

        return reps

    def create_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        selected_modality: int = None,
        not_modality_list: list = [],
    ) -> tuple:
        """
        Creates representations for the specified modality from the data
        loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing the input data.
        selected_modality : int, optional
            Index of the modality to extract representations for. If None,
            all modalities are processed. Defaults to None.
        not_modality_list : list, optional
            List of modalities to exclude from the concatenation. Defaults to
            an empty list.

        Returns
        -------
        tuple
            A tuple containing:
            - representations_cat: Concatenated representations for the
              selected modality.
            - concatenated_modalities: Dictionary of concatenated modalities
              excluding those in not_modality_list.
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

        # Optimizer for PP2-M
        if self.synthetic_experiment is False:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            # estimate total number of training steps (iterations)
            total_steps = self.trainer.estimated_stepping_batches

            # warm-up for first 5% of steps
            warmup_steps = max(1, int(0.05 * total_steps))

            # schedule: linearly ramp over warmup_steps, then stay at constant
            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step + 1) / warmup_steps
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # step after every optimizer.step()
                    "frequency": 1,
                    "strict": True,
                },
            }
        # Optimizer for synthetic data experiments for partial information
        # decomposition (PID)
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

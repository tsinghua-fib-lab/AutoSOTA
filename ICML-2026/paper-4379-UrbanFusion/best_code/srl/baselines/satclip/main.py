#!/usr/bin/env python3
"""
Description: Implementation of the SatCLIP model for training and evaluation
using PyTorch Lightning. This module is based on the original
SatCLIP codebase https://github.com/microsoft/satclip.
"""
import os

import torch
from lightning import LightningModule

from srl.baselines.satclip.loss import SatCLIPLoss
from srl.baselines.satclip.model import SatCLIP


class SatCLIPLightningModule(LightningModule):
    def __init__(
        self,
        embed_dim=256,
        image_resolution=224,
        vision_layers="moco_vit16",
        vision_width=None,
        vision_patch_size=None,
        in_channels=None,
        le_type="sphericalharmonics",
        pe_type="siren",
        frequency_num=None,
        max_radius=None,
        min_radius=None,
        legendre_polys=10,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=0.0001,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=512,
        pretrained_weights_path=None,
        precomputed_features=False,
        plot_representation_space: dict = {},
        synthetic_experiment=False,
    ) -> None:
        """
        Initializes the SatCLIPLightningModule with the specified parameters.

        Parameters
        ----------
        embed_dim : int, optional
            Dimension of the embedding space (default is 256).
        image_resolution : int, optional
            Resolution of the input images (default is 224).
        vision_layers : str, optional
            Type of vision layers to use (default is "moco_vit16").
        vision_width : int, optional
            Width of the vision layers (default is None).
        vision_patch_size : int, optional
            Size of the patches for vision layers (default is None).
        in_channels : int, optional
            Number of input channels (default is None).
        le_type : str, optional
            Type of location encoding to use (default is "sphericalharmonics").
        pe_type : str, optional
            Type of positional encoding to use (default is "siren").
        frequency_num : int, optional
            Number of frequencies for encoding (default is None).
        max_radius : float, optional
            Maximum radius for encoding (default is None).
        min_radius : float, optional
            Minimum radius for encoding (default is None).
        legendre_polys : int, optional
            Number of Legendre polynomials to use (default is 10).
        harmonics_calculation : str, optional
            Method for calculating harmonics (default is "analytic").
        sh_embedding_dims : int, optional
            Dimensions for spherical harmonics embedding (default is 32).
        learning_rate : float, optional
            Learning rate for the optimizer (default is 0.0001).
        weight_decay : float, optional
            Weight decay for the optimizer (default is 0.01).
        num_hidden_layers : int, optional
            Number of hidden layers in the model (default is 2).
        capacity : int, optional
            Capacity of the model (default is 512).
        pretrained_weights_path : str, optional
            Path to pretrained weights (default is None).
        precomputed_features : bool, optional
            Whether to use precomputed features for the PP2-M dataset (no need
            for forward pass through backbones) (default is False).
        plot_representation_space : dict, optional
            Configuration for plotting the representation space, including
            options for validation and test plotting, saving representations,
            and the directory to save plots (default is an empty dict).
        synthetic_experiment : bool, optional
            Whether to use synthetic data for training, for partial
            information decomposition experiments (default is False).
        """
        super().__init__()
        self.plot_representation_space = plot_representation_space

        self.model = SatCLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
            pretrained_weights_path=pretrained_weights_path,
            precomputed_features=precomputed_features,
            synthetic_experiment=synthetic_experiment,
        )

        self.loss_fun = SatCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.precomputed_features = precomputed_features
        self.save_hyperparameters()

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

    def common_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Common step for training, validation, and testing. It processes the
        batch, extracts the images and coorinate points, and computes the
        logits for images and coordinates. Finally, it computes the loss using
        the loss function.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the batch in the dataloader.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the batch.
        """
        inputs, batch_size = self._process_batch(batch)
        images = inputs[1]
        t_points = inputs[0]
        logits_per_image, logits_per_coord = self.model(images, t_points)
        return self.loss_fun(logits_per_image, logits_per_coord)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for the model. It computes the loss for the given batch
        and logs it.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the batch in the dataloader.

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the batch.
        """
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Validation step for the model. It computes the loss for the given batch
        and logs it. The dataloader index is used to differentiate between
        different validation dataloaders.

        Parameters
        ----------
        batch : dict
            Dictionary containing modalities.
        batch_idx : int
            Index of the batch in the dataloader.
        dataloader_idx : int, optional
            Index of the dataloader (default is 0).

        Returns
        -------
        loss : torch.Tensor
            Computed loss for the batch.
        """
        loss = self.common_step(batch, batch_idx)
        if self.trainer.datamodule.dataset_name == "SyntheticPID":
            batch_size = batch["coords"].shape[0]
        else:
            batch_size = batch["sentinel2"].shape[0]
        if self.trainer.datamodule.dataset_name == "PlacePulse2":
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
        loss = self.common_step(batch, batch_idx)
        if self.trainer.datamodule.dataset_name == "SyntheticPID":
            batch_size = batch["coords"].shape[0]
        else:
            batch_size = batch["sentinel2"].shape[0]
        if self.trainer.datamodule.dataset_name == "PlacePulse2":
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
        Generates representations at the end of the validation epoch.
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
                if self.trainer.datamodule.dataset_name == "PlacePulse2":
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
        Calculates representations at the end of the test epoch.
        """

        representations = None
        if self.plot_representation_space["plot_test"]:
            if self.trainer.datamodule.dataset_name == "PlacePulse2":
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
            self.model.location.precomputed_features = False
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
            (self.model.location.precomputed_features) = (
                self.precomputed_features
            )

    def predict_step(
        self,
        batch: dict,
        batch_idx: int,
        modality: int = 0,
    ) -> torch.Tensor:
        if "gsv_img" in batch:
            del batch["gsv_img"]
        inputs, _ = self._process_batch(batch)
        images = inputs[1]
        t_points = inputs[0]
        if modality == 0:
            reps = self.model.encode_location(t_points).float()
        elif modality == 1:
            reps = self.model.encode_image(images)
        elif modality == 2:
            # location + image, concatenated
            reps = torch.cat(
                [
                    self.model.encode_location(t_points).float(),
                    self.model.encode_image(images),
                ],
                dim=1,
            )
        return reps

    def create_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        selected_modality: int = None,
        not_modality_list: list = [],
    ) -> tuple:
        """
        Creates representations for the given data loader by iterating over
        the batches and collecting the representations for the specified
        modality.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing the dataset to create representations from.
        selected_modality : int, optional
            Index of the modality to create representations for (default is
            None).
        not_modality_list : list, optional
            List of modalities to exclude from the concatenation (default is
            an empty list).
        Returns
        -------
        tuple
            A tuple containing:
            - representations_cat: Concatenated representations for the
            selected modality.
            - concatenated_modalities: Dictionary containing concatenated
            modalities.
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model. It separates parameters into
        two groups: those that are gain or bias parameters (e.g., batch norm,
        layer norm, bias, logit scale) and the rest. The gain or bias
        parameters are set with a weight decay of 0.0, while the rest are set
        with the specified weight decay. According to the original
        implementation.

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer for the model.
        """

        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )

        def include(n, p):
            return not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },
            ],
            lr=self.learning_rate,
        )

        return optimizer

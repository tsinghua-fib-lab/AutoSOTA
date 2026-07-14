#!/usr/bin/env python3
"""
Description: Implements a PyTorch Lightning module for evaluating a pretrained
CSP model. Code from https://github.com/gengchenmai/csp adapted to
PyTorch Lightning framework for evaluation.

Mai, Gengchen; Lao, Ni; He, Yutong; Song, Jiaming; Ermon, Stefano.
"CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual
Representations." Proceedings of the 40th International Conference on Machine
Learning (ICML), 2023.
"""
import os

import torch
from lightning import LightningModule

from srl.baselines.CSP.models import LocationImageEncoder
from srl.baselines.CSP.utils import get_model


class CSPLightningModule(LightningModule):
    def __init__(
        self,
        model_path: str,
        precomputed_features: bool = False,
        plot_representation_space: dict = {},
    ) -> None:
        """
        Initializes the CSP Lightning Module.
        Allows evaluation of a pretrained CSP model.

        Parameters
        ----------
        model_path : str
            Path to the pretrained CSP model checkpoint.
        precomputed_features : bool, optional
            If True, uses precomputed features from PP2-M dataset.
            Defaults to False.
        plot_representation_space : dict, optional
            Dictionary containing settings for plotting the representation
            space. Defaults to an empty dictionary.
        """
        super().__init__()
        self.model = self._load_pretrained_model(model_path)
        self.model.eval()
        self.plot_representation_space = plot_representation_space
        self.precomputed_features = precomputed_features

    def _load_pretrained_model(self, path: str) -> LocationImageEncoder:
        """
        Loads a pretrained CSP model from the specified path.

        Parameters
        ----------
        path : str
            Path to the pretrained model checkpoint.

        Returns
        -------
        LocationImageEncoder
            An instance of the LocationImageEncoder model initialized with the
            parameters from the checkpoint.
        """
        checkpoint = torch.load(path)
        params = checkpoint["params"]

        loc_enc = get_model(
            train_locs=None,
            params=params,
            spa_enc_type=params["spa_enc_type"],
            num_inputs=params["num_loc_feats"],
            num_classes=params["num_classes"],
            num_filts=params["num_filts"],
            num_users=params["num_users"],
            device=params["device"],
        )

        model = LocationImageEncoder(
            loc_enc=loc_enc,
            train_loss=params["train_loss"],
            unsuper_loss=params["unsuper_loss"],
            cnn_feat_dim=params["cnn_feat_dim"],
            spa_enc_type=params["spa_enc_type"],
        ).to(params["device"])

        model.load_state_dict(checkpoint["state_dict"])
        return model

    def _process_batch(self, batch: dict) -> list:
        """
        Processes a batch dictionary by moving each tensor to the device.
        Extracts the batch size from the first tensor.

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

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0) -> int:
        """
        Dummy test step to satisfy LightningModule requirements.

        Parameters
        ----------
        batch : dict
            A batch of data from the test dataloader.
        batch_idx : int
            Index of the batch in the dataloader.
        dataloader_idx : int, optional
            Index of the dataloader, defaults to 0.
        Returns
        -------
        int
            Returns 0 to satisfy the LightningModule requirements.
        """
        return 0

    def predict_step(
        self, batch: dict, batch_idx: int, modality: int = 0
    ) -> torch.Tensor:
        """
        Predicts embeddings for a batch of data.

        Parameters
        ----------
        batch : dict
            A batch of data containing modalities (e.g., 'coords', 'gsv_img').
        batch_idx : int
            Index of the batch in the dataloader.
        modality : int, optional
            Index of the modality to use for prediction, defaults to 0.

        Returns
        -------
        torch.Tensor
            A tensor containing the predicted embeddings.
        """
        if "gsv_img" in batch:
            del batch["gsv_img"]
        inputs, _ = self._process_batch(batch)
        t_points = inputs[0]

        if self.precomputed_features:
            t_points = t_points[:, -2:]
        t_points = t_points[:, [1, 0]]  # convert to [lon, lat]
        loc_tensor = self.convert_loc_to_tensor(t_points)

        embeddings = self.model.loc_enc(loc_tensor, return_feats=True)

        return embeddings

    def on_test_epoch_end(self) -> None:
        """
        Create and save representations from test dataloaders.
        """
        representations = None
        if self.plot_representation_space["plot_test"]:
            if self.trainer.datamodule.dataset_name == "PlacePulse2":
                test_loaders = self.trainer.datamodule.test_dataloader()
                if not isinstance(test_loaders, list):
                    test_loaders = [test_loaders]

                for dl_idx, test_loader in enumerate(test_loaders):
                    (representations, concatenated_modalities) = (
                        self.create_representations(
                            test_loader,
                            selected_modality=0,  # location only
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
                        os.makedirs(self.plot_representation_space["save_dir"])
                    torch.save(
                        representations,
                        f"{self.plot_representation_space['save_dir']}/"
                        f"representations_test_epoch_{self.current_epoch}_"
                        f"modality_0_dl_{dl_idx}.pt",
                    )
                    save_path = (
                        f"{self.plot_representation_space['save_dir']}"
                        f"/modalities_test_dl_{dl_idx}.pt"
                    )
                    concatenated_modalities_save = concatenated_modalities.get(
                        "gsv_img", None
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

    def create_representations(
        self,
        data_loader: torch.utils.data.DataLoader,
        selected_modality: int = 0,
        not_modality_list: list = None,
    ) -> tuple:
        """
        Creates representations from a data loader by iterating through the
        batches and collecting the outputs of the predict_step method.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing the dataset to process.
        selected_modality : int, optional
            Index of the modality to use for prediction, defaults to 0.
        not_modality_list : list, optional
            List of modalities to exclude from concatenated_modalities.
            Defaults to None, which means no modalities are excluded.

        Returns
        -------
        tuple
            A tuple containing:
            - representations_cat: torch.Tensor or None
                Concatenated tensor of representations from all batches,
                or None if no representations were collected.
            - concatenated_modalities: dict
                Dictionary containing concatenated modalities from the batches,
                excluding those in not_modality_list.
        """
        all_representations = []
        modality_inputs = {}

        if hasattr(data_loader.dataset, "return_filename"):
            data_loader.dataset.return_filename = True

        for batch_idx, batch in enumerate(data_loader):
            for modality, value in batch.items():
                if modality not in modality_inputs:
                    modality_inputs[modality] = []
                modality_inputs[modality].append(value)

            reps = self.predict_step(batch, batch_idx, selected_modality)
            all_representations.append(reps)

        representations_cat = (
            torch.cat(all_representations, dim=0)
            if all_representations
            else None
        )

        concatenated_modalities = {}
        for modality, values in modality_inputs.items():
            if modality in not_modality_list:
                continue
            if isinstance(values[0], torch.Tensor):
                concatenated_modalities[modality] = torch.cat(values, dim=0)
            else:
                concatenated_modalities[modality] = values

        if hasattr(data_loader.dataset, "return_filename"):
            data_loader.dataset.return_filename = False

        return representations_cat, concatenated_modalities

    @staticmethod
    def convert_loc_to_tensor(x):
        """
        Normalizes and converts coordinates to a PyTorch tensor.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor
            Shape [batch_size, 2] with (lon, lat) coordinates.

        Returns
        -------
        torch.Tensor
            A tensor with normalized coordinates.
        """
        x[:, 0] /= 180.0  # normalize longitude
        x[:, 1] /= 90.0  # normalize latitude
        return x

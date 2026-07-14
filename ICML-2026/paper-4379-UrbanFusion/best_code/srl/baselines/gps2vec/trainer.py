#!/usr/bin/env python3
"""
Description: Implementation of GPS2Vec (tag) and GPS2Vec++ (visual) as a
LightningModule.
"""
import os

import numpy as np
import torch
from lightning import LightningModule

from srl.baselines.gps2vec.gps2vec import georep


class GPS2VecLightningModule(LightningModule):
    def __init__(
        self,
        basedir: str,
        model_type: str = "visual",
        precomputed_features: bool = False,
        plot_representation_space: dict = {},
    ):
        """
        LightningModule for obtaining GPS embeddings using GPS2Vec

        Parameters
        ----------
        basedir : str
            Base directory where the GPS2Vec model folders (e.g.,
            "models_visual" or "models_tag") reside.
        model_type : str, optional
            Either "visual" or "tag" to select which GPS2Vec variant to use.
            Defaults to "visual".
        precomputed_features : bool, optional
            If True, assumes the last two columns of the input location tensor
            are already processed features and directly feeds them to georep.
            Otherwise, uses raw lon/lat pairs. Defaults to False.
        plot_representation_space : dict, optional
            Dictionary with keys:
                - "plot_test": bool, whether to run on_test_epoch_end plotting
                logic.
                - "save_dir": str, directory in which to save the test
                representations.
        """
        super().__init__()

        # Set up GPS2Vec model directory and flag via helper
        self.modeldir, self.flag = self._load_pretrained_model(
            basedir, model_type
        )
        self.precomputed_features = precomputed_features
        self.plot_representation_space = plot_representation_space

        # Default GPS2Vec hyperparameters
        self.nrows = 20
        self.ncols = 20
        self.sigma = 20000

    def _load_pretrained_model(self, basedir: str, model_type: str) -> tuple:
        """
        Determines the GPS2Vec model directory and flag from the given base
        directory and model type.

        Parameters
        ----------
        base_dir : str
            The base directory containing the GPS2Vec model folders.
        model_type : str
            The type of GPS2Vec model to load, either "visual" or "tag".

        Returns
        -------
        tuple:
            - modeldir (str): Full path to the GPS2Vec model folder.
            - flag (int): 0 for "visual", 1 for "tag".
        """
        if model_type == "visual":
            modeldir = os.path.join(basedir, "GPS2Vec_visual")
            flag = 0
        elif model_type == "tag":
            modeldir = os.path.join(basedir, "GPS2Vec_tag")
            flag = 1
        else:
            raise ValueError(
                f"Invalid GPS2Vec model_type: {model_type}. Choose 'visual'"
                f" or 'tag'."
            )
        return modeldir, flag

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

    def convert_loc_to_tensor(self, x: torch.Tensor) -> np.ndarray:
        """
        Converts [lat, lon] tensor to NumPy array in the same order (no flip).

        Returns:
            np.ndarray of shape [B, 2], with [lat, lon] format.
        """
        if self.precomputed_features:
            x = x[:, -2:]
        return x.cpu().numpy()

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> int:
        """
        Dummy test step to satisfy LightningModule requirements.
        """
        return 0

    def predict_step(
        self, batch: dict, batch_idx: int, modality: int = 0
    ) -> torch.Tensor:
        """
        Given a batch, produces GPS2Vec embeddings for the coordinate modality.

        Parameters
        ----------
        batch : dict
            The input batch dictionary containing modalities.
        batch_idx : int
            The index of the batch in the DataLoader.
        modality : int
            The index of the selected modality (not used for GPS2Vec but kept
            for compatibility).

        Returns
        -------
        torch.Tensor
            A tensor of GPS2Vec embeddings of shape [B, D], where D is the
            output dimension of the GPS2Vec model.
        """
        batch = batch.copy()
        if "gsv_img" in batch:
            del batch["gsv_img"]
        inputs, _ = self._process_batch(batch)
        t_points = inputs[0]
        if self.precomputed_features:
            t_points_np = t_points[:, -2:]
            t_points_np = self.convert_loc_to_tensor(t_points)
        else:
            t_points_np = self.convert_loc_to_tensor(t_points)

        embeddings_list = []
        for single_loc in t_points_np:
            geofea = georep(
                single_loc,
                self.modeldir,
                self.nrows,
                self.ncols,
                self.sigma,
                self.flag,
            )
            embeddings_list.append(np.asarray(geofea))
        embeddings_np = np.stack(embeddings_list, axis=0)
        embeddings = torch.from_numpy(embeddings_np).to(self.device)
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
        Create representations from a DataLoader by iterating through the
        dataset and applying the predict_step method.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            The DataLoader containing the dataset to process.
        selected_modality : int, optional
            The index of the selected modality to process. Defaults to 0.
        not_modality_list : list, optional
            A list of modalities to exclude from the concatenation.
            Defaults to None, which means no modalities are excluded.

        Returns
        -------
        tuple:
            - representations_cat : torch.Tensor
                Concatenated representations from the predict_step method.
            - concatenated_modalities : dict
                Dictionary containing concatenated modalities from the dataset.
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

        # Concatenate results
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

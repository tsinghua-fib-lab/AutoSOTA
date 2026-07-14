#!/usr/bin/env python3
"""
Description: Implementation of PDFM model, by loading precomputed embeddings.
Embeddings are from https://github.com/google-research/population-dynamics.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from lightning import LightningModule
from shapely.geometry import Point


class PDFMLightningModule(LightningModule):
    def __init__(
        self,
        basedir: str,
        precomputed_features: bool = False,
        plot_representation_space: dict = {},
    ) -> None:
        """
        Initialize the PDFM Lightning Module.

        Parameters
        ----------
        basedir : str
            Base directory where the precomputed embeddings and regions are
            stored.
        precomputed_features : bool, optional
            Whether to use precomputed features of PP2-M dataset (default is
            False).
        plot_representation_space : dict, optional
            Configuration for plotting the representation space, including
            save directory and whether to plot test data (default is empty
            dict).
        """
        super().__init__()

        self.basedir = basedir
        self.precomputed_features = precomputed_features
        self.plot_representation_space = plot_representation_space

        self.embeddings = pd.read_csv(
            os.path.join(self.basedir, "zcta_embeddings.csv")
        )
        self.regions = gpd.read_file(
            os.path.join(self.basedir, "zcta.geojson")
        )
        # Ensure geometry is valid and build spatial index
        self.regions = self.regions[self.regions.is_valid]
        self.regions.sindex  # force spatial index build

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: int = 0
    ) -> int:
        """
        Dummy test step to satisfy LightningModule requirements.

        Parameters
        ----------
        batch : dict
            Batch of data from the test dataloader.
        batch_idx : int
            Index of the batch in the dataloader.
        dataloader_idx : int, optional
            Index of the dataloader (default is 0).

        Returns
        -------
        int
            Returns 0 to indicate successful completion of the step.
        """
        return 0

    def get_embedding_for_latlon(self, lat, lon) -> np.ndarray:
        """
        Get the embedding for a given latitude and longitude.

        Parameters
        ----------
        lat : float
            Latitude of the point.
        lon : float
            Longitude of the point.

        Returns
        -------
        np.ndarray
            Embedding vector for the given latitude and longitude, or None if
            no matching region is found.
        """
        point = Point(lon, lat)

        # Use spatial index to reduce search space
        possible_matches_idx = list(
            self.regions.sindex.intersection(point.bounds)
        )
        possible_matches = self.regions.iloc[possible_matches_idx]
        match = possible_matches[possible_matches.contains(point)]

        if match.empty:
            return None

        zip_code = match.iloc[0]["place"]
        row = self.embeddings[self.embeddings["place"] == zip_code]

        if row.empty:
            return None

        feature_columns = [
            col for col in self.embeddings.columns if col.startswith("feature")
        ]
        return row[feature_columns].values[0]

    def predict_step(
        self, batch: dict, batch_idx: int, modality: int = 0
    ) -> torch.Tensor:
        batch = batch.copy()
        t_points_np = batch["coords"].cpu().numpy()
        if self.precomputed_features:
            t_points_np = t_points_np[:, -2:]

        feature_dim = len(
            [
                col
                for col in self.embeddings.columns
                if col.startswith("feature")
            ]
        )
        embeddings_list = []

        for single_loc in t_points_np:
            embedding = self.get_embedding_for_latlon(
                lat=single_loc[0], lon=single_loc[1]
            )

            if embedding is not None:
                embeddings_list.append(embedding)
            else:
                embeddings_list.append(np.full(feature_dim, np.nan))

        embeddings_np = np.stack(embeddings_list, axis=0)
        embeddings = torch.from_numpy(embeddings_np).to(self.device)
        return embeddings

    def on_test_epoch_end(self):
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
        Create representations from the data loader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader containing the dataset from which to create
            representations.
        selected_modality : int, optional
            Index of the modality to select for representation creation
            (default is 0, which corresponds to location only).
        not_modality_list : list, optional
            List of modalities to exclude from the concatenation of
            representations (default is None, which means no modalities are
            excluded).

        Returns
        -------
        tuple
            A tuple containing:
            - representations_cat: Concatenated tensor of representations.
            - concatenated_modalities: Dictionary of concatenated modalities.
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

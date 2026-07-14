#!/usr/bin/env python3
"""
Description: Implementation of land use europe tasks.

Usage:
-- settings [config json name] -- model [linear or mlp] -- task [C or F, C is coarse, F is fine]
Example:
python scripts/downstream_tasks/landuse_eu.py --settings landuse_eu_within_region_UrbanFusion_trained_1_ridge.json --task C
"""
import os

print("Current working directory:", os.getcwd(), flush=True)
import argparse
import datetime
import json
import os
import time
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from shapely.geometry import Point

from scripts.downstream_tasks.downstream_eval import run_pipeline

# Argument parser
parser = argparse.ArgumentParser(
    description="Run PP2 perception with a specified settings JSON"
)
parser.add_argument(
    "-s",
    "--settings",
    default="PP2_perception_within_region_pretrained_baselines.json",
    help="Path to the settings JSON file",
)
parser.add_argument(
    "-m", "--model", default="both", help="Path to the settings JSON file"
)

parser.add_argument(
    "-t",
    "--task",
    default="C",
    help="Task type: C = coarsely grained, F = fine grained",
)
args = parser.parse_args()
SETTINGS_NAME = args.settings
MODEL = args.model
FINE_GRAINED = args.task
print(f"Using settings: {SETTINGS_NAME}", flush=True)
print(f"Using model: {MODEL}", flush=True)

# Configuration
CLUSTER = True
DATA_LOADERS_NAME = {0: "in-region", 1: "out-of-region"}
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if CLUSTER:
    data_dir = "/svi_data"
    log_dir = "/logs"
    output_dir = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "landuse_eu",
    )
    settings_path = os.path.join(output_dir, SETTINGS_NAME)
    output_path = os.path.join(
        output_dir, SETTINGS_NAME[:-5] + f"_{MODEL}_{FINE_GRAINED}.csv"
    )
else:
    data_dir = os.path.join(os.getcwd(), "svi_data")
    log_dir = os.path.join(os.getcwd(), "logs")
    output_dir = os.path.join(
        os.getcwd(),
        "svi_data",
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "landuse_eu",
    )
    settings_path = os.path.join(output_dir, SETTINGS_NAME)
    output_path = os.path.join(
        output_dir, SETTINGS_NAME[:-5] + f"_{MODEL}.csv"
    )

# Read config
if not os.path.isfile(settings_path):
    raise FileNotFoundError(f"Settings file not found: {settings_path}")
with open(settings_path) as f:
    REPRESENTATIONS_TO_EVALUATE = json.load(f)

# Preprocess landcover data
ua_root = os.path.join(
    data_dir, "place-pulse-2.0", "downstreamtask_data", "UrbanAtlas"
)
gpkg_paths = [
    os.path.join(root, f)
    for root, _, files in os.walk(ua_root)
    for f in files
    if f.endswith(".gpkg")
]
ua_gdfs = []
for p in gpkg_paths:
    gdf = gpd.read_file(p, layer=0)
    if "class_2018" in gdf.columns:
        ua_gdfs.append(gdf[["class_2018", "geometry"]])
    else:
        print(f"{os.path.basename(p)} has no class_2018 column skipped")
urban_atlas = gpd.GeoDataFrame(
    pd.concat(ua_gdfs, ignore_index=True), crs=ua_gdfs[0].crs
).to_crs("EPSG:4326")


class DownstreamLocations:
    """
    Store and manage location lists for downstream data loaders.
    """

    def __init__(self) -> None:
        self.locations_dl_0: list = []
        self.locations_dl_1: list = []

    def set_locations_dl_0(self, locations: list) -> None:
        """
        Set locations for data loader 0.

        Parameters
        ----------
        locations : list
            List of location strings in the format "lat_lon_loc_id_city".
        """
        self.locations_dl_0 = locations

    def set_locations_dl_1(self, locations: list) -> None:
        """
        Set locations for data loader 1.

        Parameters
        ----------
        locations : list
            List of location strings in the format "lat_lon_loc_id_city".
        """
        self.locations_dl_1 = locations


locations_data = DownstreamLocations()


def parse_locations(loc_strings: list) -> gpd.GeoDataFrame:
    """
    Parse location strings into a GeoDataFrame.

    Parameters
    ----------
    loc_strings : list
        List of location strings in the format "lat_lon_loc_id_city".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the parsed locations.
    """
    rows = []
    for i, s in enumerate(loc_strings):
        lat, lon, loc_id, city = s.split("_")
        rows.append(
            {
                "idx": i,
                "lat": float(lat),
                "lon": float(lon),
                "loc_id": loc_id,
                "city": city,
                "geometry": Point(float(lon), float(lat)),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def make_X_y(
    loc_strings: list,
    repr_tensor: torch.Tensor,
    dl_idx: int,
    fine_grained: bool = False,
) -> tuple:
    """
    Attach land-class labels and build feature matrix X and target y.

    Parameters
    ----------
    loc_strings : list
        List of location strings in the format "lat_lon_loc_id_city".
    repr_tensor : torch.Tensor
        Tensor containing the representations for each location.
    dl_idx : int
        Data loader index (0 or 1).
    fine_grained : bool, optional
        Whether to use fine-grained labels (default is False).

    Returns
    -------
    tuple
        A tuple containing the feature matrix X and target vector y.
    """
    # parse locations and spatially join to Urban Atlas
    pts = parse_locations(loc_strings)
    joined = gpd.sjoin(
        pts,
        urban_atlas,
        how="left",
        predicate="within",
    ).sort_values("idx")

    raw_y = joined["class_2018"]
    # build feature matrix
    coords = np.column_stack((pts["lat"].to_numpy(), pts["lon"].to_numpy()))
    if repr_tensor is not None:
        X = repr_tensor.numpy()
        X = np.column_stack((X, coords))
    else:
        X = coords

    init_mask = raw_y != "Construction sites"
    raw_y = raw_y[init_mask]
    X = X[init_mask.values]

    # Count and print number of samples per class
    class_counts = raw_y.value_counts()
    print(
        f"⇒ Coordinates has {len(class_counts)} classes: {class_counts.to_dict()}"
    )

    # Create superclasses
    if FINE_GRAINED == "C":
        if dl_idx == 0:
            grouping = {
                # Urban fabric
                "Continuous urban fabric (S.L. : > 80%)": "Urban fabric",
                "Discontinuous dense urban fabric (S.L. : 50% -  80%)": "Urban fabric",
                "Discontinuous medium density urban fabric (S.L. : 30% - 50%)": "Urban fabric",
                "Discontinuous low density urban fabric (S.L. : 10% - 30%)": "Urban fabric",
                "Discontinuous very low density urban fabric (S.L. : < 10%)": "Urban fabric",
                # Transportation
                "Other roads and associated land": "Transportation",
                "Fast transit roads and associated land": "Transportation",
                "Railways and associated land": "Transportation",
                #'Port areas': 'Transportation',
                #'Airports': 'Transportation',
                # Industrial & built-up
                "Industrial, commercial, public, military and private units": "Industrial & built-up",
                #'Isolated structures': 'Industrial & built-up',
                #'Mineral extraction and dump sites': 'Industrial & built-up',
                # Green & recreation
                "Green urban areas": "Green & recreation",
                "Sports and leisure facilities": "Green & recreation",
                # Cropland & pasture
                "Arable land (annual crops)": "Cropland & pasture",
                "Pastures": "Cropland & pasture",
                #'Permanent crops (vineyards, fruit trees, olive groves)': 'Cropland & pasture',
                #'Complex and mixed cultivation patterns': 'Cropland & pasture',
                # Natural vegetation
                "Forests": "Natural vegetation",
                "Herbaceous vegetation associations (natural grassland, moors...)": "Natural vegetation",
                "Wetlands": "Natural vegetation",
                # Water & unused land
                "Water": "Water & unused land",
                #'Land without current use': 'Water & unused land',
            }
        elif dl_idx == 1:
            grouping = {
                # Urban fabric
                "Continuous urban fabric (S.L. : > 80%)": "Urban fabric",
                "Discontinuous dense urban fabric (S.L. : 50% -  80%)": "Urban fabric",
                "Discontinuous medium density urban fabric (S.L. : 30% - 50%)": "Urban fabric",
                "Discontinuous low density urban fabric (S.L. : 10% - 30%)": "Urban fabric",
                # 'Discontinuous very low density urban fabric (S.L. : < 10%)': 'Urban fabric',
                # Transportation
                "Other roads and associated land": "Transportation",
                "Fast transit roads and associated land": "Transportation",
                "Railways and associated land": "Transportation",
                "Port areas": "Transportation",
                # 'Airports': 'Transportation',
                # Industrial & built-up
                "Industrial, commercial, public, military and private units": "Industrial & built-up",
                # 'Isolated structures': 'Industrial & built-up',
                # 'Mineral extraction and dump sites': 'Industrial & built-up',
                # Green & recreation
                "Green urban areas": "Green & recreation",
                "Sports and leisure facilities": "Green & recreation",
                # Cropland & pasture
                "Arable land (annual crops)": "Cropland & pasture",
                # 'Pastures': 'Cropland & pasture',
                "Permanent crops (vineyards, fruit trees, olive groves)": "Cropland & pasture",
                "Complex and mixed cultivation patterns": "Cropland & pasture",
                # Natural vegetation
                "Forests": "Natural vegetation",
                "Herbaceous vegetation associations (natural grassland, moors...)": "Natural vegetation",
                "Wetlands": "Natural vegetation",
                # Water & unused land
                "Water": "Water & unused land",
                "Land without current use": "Water & unused land",
            }
        super_y = raw_y.map(grouping)
        map_mask = super_y.notna()
        y_filt = super_y[map_mask]
        X_filt = X[map_mask.values]

    # Remove small classes
    if FINE_GRAINED == "F":
        # apply masking / exclusion rules on raw labels
        valid_mask = ~raw_y.isna()
        if dl_idx == 0:
            exclude = {
                "Mineral extraction and dump sites",
                "Complex and mixed cultivation patterns",
                "Permanent crops (vineyards, fruit trees, olive groves)",
                "Wetlands",
                "Land without current use" "Airports",
                "Isolated structures",
                "Port areas",
            }
            exclude = {
                "Airports",
                "Discontinuous very low density urban fabric (S.L. : < 10%)",
                "Isolated structures",
                "Mineral extraction and dump sites",
                "Pastures",
            }
        keep_mask = valid_mask & ~raw_y.isin(exclude)
        X_filt = X[keep_mask]
        y_filt = raw_y[keep_mask]

    # count and print number of samples per class
    class_counts = y_filt.value_counts()
    print(
        f"⇒ Coordinates has {len(class_counts)} classes: {class_counts.to_dict()}"
    )

    # encode string labels to integer codes
    y_codes = pd.Categorical(y_filt).codes
    print(
        f"⇒ Coordinates has {len(np.unique(y_codes))} classes after filtering: "
        f"{dict(zip(np.unique(y_codes), pd.Categorical(y_filt).categories))}"
    )

    return X_filt.astype(np.float32), y_codes.astype(np.int64)


def evaluate_one_setting(
    settings_name: str, settings_info: dict, dl_idx: int, dataset_name: str
) -> list:
    """
    Evaluate one configuration (one model).

    Parameters
    ----------
    settings_name : str
        The name of the settings.
    settings_info : dict
        The information of the settings.
    dl_idx : int
        The data loader index.
    dataset_name : str
        The name of the dataset.

    Returns
    -------
    list
        The evaluation results.
    """
    date = settings_info["date"]
    epoch = settings_info["epoch"]
    modalities = settings_info["modalities"]
    mod_names = settings_info["modalities_name"]

    if date is None:
        loc_strings = (
            locations_data.locations_dl_0
            if dl_idx == 0
            else locations_data.locations_dl_1
        )
    else:
        loc_path = os.path.join(
            log_dir,
            settings_name,
            "runs",
            date,
            "plots",
            f"modalities_test_dl_{dl_idx}.pt",
        )
        print(f"Loading locations from: {loc_path}")
        loc_strings = torch.load(loc_path, map_location="cpu")
        loc_strings = [s for sub in loc_strings for s in sub]
        loc_strings = torch.load(loc_path, map_location="cpu")
        loc_strings = [s for sub in loc_strings for s in sub]

    if not locations_data.locations_dl_0 and dl_idx == 0:
        locations_data.set_locations_dl_0(loc_strings)
    if not locations_data.locations_dl_1 and dl_idx == 1:
        locations_data.set_locations_dl_1(loc_strings)

    results = []
    for mod_idx in modalities:
        name = mod_names[modalities.index(mod_idx)]

        # Representation file name
        fname = (
            f"representations_test_epoch_{epoch}_"
            f"{'masked_' if 'UrbanFusion' in settings_name or 'Raw' in settings_name else ''}"
            f"modality_{mod_idx}_dl_{dl_idx}.pt"
        )

        if date is None:
            features = None  # => only coordinates as features
        else:
            features = torch.load(
                os.path.join(
                    log_dir, settings_name, "runs", date, "plots", fname
                ),
                map_location="cpu",
            )

        # Build dataset & evaluate
        X, y = make_X_y(
            loc_strings, features, dl_idx, fine_grained=FINE_GRAINED
        )

        uniq, cnts = np.unique(y, return_counts=True)
        print(
            f"⇒ {name} has {len(uniq)} classes " f"({dict(zip(uniq, cnts))})"
        )
        if MODEL == "mlp" or MODEL == "both":
            df_mlp = run_pipeline(
                X,
                y,
                task_type="classification",
                model_type="mlp",
                epochs=40,
                metric_name="log_loss",
                standardize=True,
                class_weights=None,
                alpha_range=(1e-4, 10000.0),
                lr_range=(1e-5, 1e-1),
                weight_decay_range=(1e-6, 1e-1),
                dataset_name=dataset_name,
            )
            df_mlp["settings_name"] = settings_name
            df_mlp["modality_name"] = name
            df_mlp["data_loader_name"] = DATA_LOADERS_NAME[dl_idx]
            df_mlp["model"] = "mlp"
            results.append(df_mlp)

        if MODEL == "ridge" or MODEL == "both":
            # Ridge
            df_ridge = run_pipeline(
                X,
                y,
                task_type="classification",
                model_type="ridge",
                metric_name="log_loss",
                standardize=True,
                class_weights=None,
                alpha_range=(1e-4, 10000.0),
                lr_range=(1e-5, 1e-1),
                weight_decay_range=(1e-6, 1e-1),
                dataset_name=dataset_name,
            )
            df_ridge["settings_name"] = settings_name
            df_ridge["modality_name"] = name
            df_ridge["data_loader_name"] = DATA_LOADERS_NAME[dl_idx]
            df_ridge["model"] = "ridge"
            results.append(df_ridge)

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    start = time.time()
    all_rows = []

    for setting, info in REPRESENTATIONS_TO_EVALUATE.items():
        for dl in info["data_loaders"]:
            print(dl, flush=True)
            print(FINE_GRAINED, flush=True)

            if dl == 0 and FINE_GRAINED == "F":
                dataset_name = "landuse_eu_f_in_region"
            else:
                dataset_name = None
            print(dataset_name, flush=True)
            all_rows.append(
                evaluate_one_setting(setting, info, dl, dataset_name)
            )
    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

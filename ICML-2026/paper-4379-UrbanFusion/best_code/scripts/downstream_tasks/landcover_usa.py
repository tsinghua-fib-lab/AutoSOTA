#!/usr/bin/env python3
"""
Description: Implementation of land cover usa task.

Usage:
-- settings [config json name] -- model [linear or mlp]

Example:
python python scripts/downstream_tasks/landuse_usa.py --settings landuse_usa_within_region_UrbanFusion_trained_1_ridge.json --model ridge
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
args = parser.parse_args()
SETTINGS_NAME = args.settings
MODEL = args.model
print(f"Using settings: {SETTINGS_NAME}", flush=True)
print(f"Using model: {MODEL}", flush=True)

# Configuration
CLUSTER = True
DATA_LOADERS_NAME = ["In-region", "Out-of-region"]
# Remove some samples, which are not supported by PDFM model
REMOVE_IN_REGION = [
    138,
    160,
    212,
    366,
    369,
    547,
    594,
    632,
    694,
    838,
    941,
    1018,
    1021,
    1160,
    1306,
    1468,
    1571,
    1953,
    2074,
    2174,
    2274,
    2299,
    2554,
    2567,
    2746,
    3019,
    3044,
    3051,
    3095,
    3096,
    3180,
    3347,
    3441,
    3838,
    3864,
    3897,
    3998,
    4331,
    4385,
    4520,
    4626,
    4630,
    4758,
    4849,
]
REMOVE_OUT_OF_REGION = []
US_CITIES_IN_REGION = [
    "Philadelphia",
    "Denver",
    "Atlanta",
    "Portland",
    "Houston",
    "Minneapolis",
    "Chicago",
    "Seattle",
    "WashingtonDC",
    "Boston",
    "SanFrancisco",
    "LosAngeles",
]
US_CIITES_OUT_OF_REGION = ["NewYork"]
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if CLUSTER:
    data_dir = "/svi_data"
    log_dir = "/logs"
    output_dir = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "landuse_usa",
    )
    settings_path = os.path.join(output_dir, SETTINGS_NAME)
    output_path = os.path.join(
        output_dir, SETTINGS_NAME[:-5] + f"_{MODEL}.csv"
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
        "landuse_usa",
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


class DownstreamLocations:
    """
    Cache the location lists.
    """

    def __init__(self) -> None:
        self.locations_list_dl_0: List[str] = []
        self.locations_list_dl_1: List[str] = []


locations_data = DownstreamLocations()


def parse_locations(loc_strings: list) -> gpd.GeoDataFrame:
    """
    Convert a list ['lat_lon_id_city', …] into a GeoDataFrame.

    Parameters
    ----------
    loc_strings : list
        A list of location strings in the format 'lat_lon_id_city'.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the parsed location data.
    """
    rows = []
    for i, s in enumerate(loc_strings):
        lat, lon, loc_id, city = s.split("_")
        rows.append(
            dict(
                idx=i,
                lat=float(lat),
                lon=float(lon),
                loc_id=loc_id,
                city=city,
                geometry=Point(float(lon), float(lat)),
            )
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def make_X_y(
    pts: gpd.GeoDataFrame,
    repr_tensor: torch.Tensor,
    dataloader_idx: int,
) -> tuple:
    """
    Create feature matrix X and target vector y from GeoDataFrame and
    representation tensor.

    Parameters
    ----------
    pts : gpd.GeoDataFrame
        A GeoDataFrame containing the input points.
    repr_tensor : torch.Tensor
        A tensor containing the representation features.
    dataloader_idx : int
        The index of the dataloader (0 for in-region, 1 for out-of-region).

    Returns
    -------
    tuple
        A tuple containing the feature matrix X and the target vector y.
    """

    lat_col = "lat" if "lat" in pts.columns else "latitude"
    lon_col = "lon" if "lon" in pts.columns else "longitude"
    y = pts["land_use"].to_numpy()
    print("Columns in GeoDataFrame:", list(pts.columns))
    coords = np.column_stack(
        [pts[lat_col].to_numpy(), pts[lon_col].to_numpy()]
    )

    if repr_tensor is not None:
        X = np.column_stack([repr_tensor.numpy(), coords])
    else:
        X = coords

    print("Initial shapes  X:", X.shape, " y:", y.shape)
    # Combine and remove some classes
    if dataloader_idx == 0:  # in-region
        mask = ~pts["city"].isin(US_CITIES_IN_REGION)
        print("Cities in region:", pts["city"][~mask].unique())
        y[mask] = 250
    else:  # out-of-region
        mask = ~pts["city"].isin(US_CIITES_OUT_OF_REGION)
        y[mask] = 250

    y = np.where(y == 250, np.nan, y)

    if dataloader_idx == 0:
        y = np.where(y == 95, np.nan, y)
        y = np.where(np.isin(y, [42, 43]), 41, y)
    else:  # dl-1
        y = np.where(np.isin(y, [11, 41]), np.nan, y)
    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    print(f"⇒ kept {len(y):,} samples (after filtering)")
    if dataloader_idx == 0:
        drop_idx = REMOVE_IN_REGION
    else:
        drop_idx = REMOVE_OUT_OF_REGION
    keep = ~np.isin(np.arange(len(X)), drop_idx)
    X, y = X[keep], y[keep]
    print(f"⇒ kept {len(y):,} samples (after removing indices)")
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        print("Found NaNs in final features! → rows:", np.where(nan_mask)[0])

    return X.astype(np.float32), y.astype(np.int64)


def evaluate_one_setting(
    settings_name: str, settings_info: dict, dl_idx: int
) -> list:
    """
    Evaluate a single setting.

    Parameters
    ----------
    settings_name : str
        The name of the settings.
    settings_info : dict
        A dictionary containing information about the settings.
    dl_idx : int
        The index of the data loader (0 for in-region, 1 for out-of-region).

    Returns
    -------
    list
        A list of evaluation results for the setting.
    """
    date = settings_info["date"]
    epoch = settings_info["epoch"]
    modalities = settings_info["modalities"]
    mod_names = settings_info["modalities_name"]

    if date is None:
        loc_strings = (
            locations_data.locations_list_dl_0
            if dl_idx == 0
            else locations_data.locations_list_dl_1
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
        loc_strings = torch.load(loc_path, map_location="cpu")
        loc_strings = [s for sub in loc_strings for s in sub]

    if not locations_data.locations_list_dl_0 and dl_idx == 0:
        locations_data.locations_list_dl_0 = loc_strings
    if not locations_data.locations_list_dl_1 and dl_idx == 1:
        locations_data.locations_list_dl_1 = loc_strings

    gdf_all = parse_locations(loc_strings)

    geotiff = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "landuse_usa",
        "Annual_NLCD_LndCov_2023_CU_C1V0.tif",
    )
    with rasterio.open(geotiff) as src:
        if gdf_all.crs != src.crs:
            gdf_all = gdf_all.to_crs(src.crs)
        gdf_all["land_use"] = [
            v[0] if v else None
            for v in src.sample([(p.x, p.y) for p in gdf_all.geometry])
        ]

    results = []

    for mod_idx in modalities:
        name = mod_names[modalities.index(mod_idx)]

        # Representation file name
        fname = (
            f"representations_test_epoch_{epoch}_"
            f"{'masked_' if 'UrbanFusion' in settings_name or 'Raw' in settings_name else ''}"
            f"modality_{mod_idx}_dl_{dl_idx}.pt"
        )

        features: torch.Tensor | None
        if date is None:
            features = None  # only coordinates as features
        else:
            features = torch.load(
                os.path.join(
                    log_dir, settings_name, "runs", date, "plots", fname
                ),
                map_location="cpu",
            )

        # ––– build dataset & evaluate –––
        X, y = make_X_y(gdf_all, features, dl_idx)

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
            all_rows.append(evaluate_one_setting(setting, info, dl))
    results_df = pd.concat(all_rows, ignore_index=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

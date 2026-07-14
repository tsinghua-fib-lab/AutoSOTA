#!/usr/bin/env python3
"""
Description: Implementation of housing prices task.

Usage:
-- settings [config json name] -- model [linear or mlp]

Example:
python scripts/downstream_tasks/London_housing.py --settings London_housing_within_region_trained_UrbanFusionV1_3.json
"""
import os

print("Current working directory:", os.getcwd(), flush=True)
import argparse
import datetime
import json
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
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
    default="London_housing_within_region_pretrained_baselines_old.json",
    help="Path to the settings JSON file",
)
parser.add_argument(
    "-m", "--model", default="both", help="Path to the settings JSON file"
)
args = parser.parse_args()
SETTINGS_NAME = args.settings
MODEL = args.model

# Configuration
CLUSTER = True
DATA_LOADERS_NAME = ["In-region", "Out-of-region"]
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if CLUSTER:
    data_dir = "/svi_data"
    log_dir = "/logs"
    output_dir = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "London_housing",
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
        "London_housing",
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


def main():
    """
    Train and evaluate downstream learner.
    """
    # Load data
    df_housing = pd.read_csv(
        os.path.join(
            data_dir,
            "place-pulse-2.0/downstreamtask_data/London_Housing_Prices/London_Housing_Prices_locations.csv",
        )
    )
    geometry = [
        Point(xy) for xy in zip(df_housing.longitude, df_housing.latitude)
    ]
    gdf = gpd.GeoDataFrame(df_housing, geometry=geometry, crs="EPSG:4326")

    # Additional features, preprocessing
    selected = [
        "history_price_log",
        "bathrooms",
        "bedrooms",
        "floorAreaSqM",
        "livingRooms",
        "tenure",
        "propertyType",
        "currentEnergyRating",
        "longitude",
        "latitude",
    ]
    mask = gdf[selected].notna().all(axis=1)
    gdf = gdf[mask].copy()
    gdf["floorAreaSqM_log"] = np.log(gdf.floorAreaSqM)
    cats = ["tenure", "propertyType", "currentEnergyRating"]
    gdf = pd.get_dummies(gdf, columns=cats, drop_first=False)
    base = [
        "bathrooms",
        "bedrooms",
        "livingRooms",
        "floorAreaSqM_log",
        "longitude",
        "latitude",
    ]
    onehot = [
        c
        for c in gdf.columns
        if any(
            pref in c
            for pref in ["tenure_", "propertyType_", "currentEnergyRating_"]
        )
    ]
    X_onehot = gdf[onehot].astype(int).values
    X_base = gdf[base].values
    X_tabular = np.hstack([X_base, X_onehot])
    print("shape one-hot features:", X_onehot.shape, flush=True)
    y = gdf["history_price_log"].values

    def load_rep(name: str, info: dict) -> np.ndarray:
        """
        Load representation for a specific name and info.

        Parameters
        ----------
        name : str
            The name of the representation.
        info : dict
            Information about the representation, including 'date' and 'epoch'.

        Returns
        -------
        np.ndarray
            The loaded representation as a NumPy array, or None if not found.
        """
        date, epoch = info["date"], info["epoch"]
        if date is None:
            return None
        fname = (
            f"representations_test_epoch_{epoch}_masked_modality_1_2_3_4_dl_London_Housing_Prices_locations.pt"
            if "UrbanFusion" in name
            else f"representations_test_epoch_{epoch}_modality_0_dl_London_Housing_Prices_locations.pt"
        )
        path = os.path.join(log_dir, name, "runs", date, "plots", fname)
        feats = torch.load(path, map_location="cpu")
        mask_t = torch.tensor(mask.values, dtype=torch.bool)
        return feats[mask_t].numpy()

    all_runs = []
    for name, info in REPRESENTATIONS_TO_EVALUATE.items():
        feats = load_rep(name, info)
        if feats is not None:
            X = np.hstack([feats, X_tabular])
        else:
            X = X_tabular
        print(X.shape, flush=True)

        print(f"=== Evaluation for {name} ===")
        if MODEL == "mlp" or MODEL == "both":
            df_mlp = run_pipeline(
                X,
                y,
                task_type="regression",
                model_type="mlp",
                epochs=40,
                standardize=29,
                alpha_range=(1e-4, 1000.0),
                lr_range=(1e-5, 1e-1),
                weight_decay_range=(1e-6, 1e-1),
            )
            df_mlp["representation"] = name
            df_mlp["model"] = "mlp"
            all_runs.append(df_mlp)
        if MODEL == "ridge" or MODEL == "both":
            df_ridge = run_pipeline(
                X,
                y,
                task_type="regression",
                model_type="ridge",
                standardize=29,
                alpha_range=(1e-4, 10000.0),
                lr_range=(1e-5, 1e-1),
                weight_decay_range=(1e-6, 1e-1),
            )
            df_ridge["representation"] = name
            df_ridge["model"] = "ridge"
            all_runs.append(df_ridge)
    results_df = pd.concat(all_runs, ignore_index=True)
    return results_df


if __name__ == "__main__":
    start = time.time()
    df_results = main()
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

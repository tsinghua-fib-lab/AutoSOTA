#!/usr/bin/env python3
"""
Description: Implementation of Place Pulse 2.0 perception task.

Usage:
-- settings [config json name] -- model [linear or mlp]
Example:
python scripts/downstream_tasks/PP2_perception.py --settings PP2_perception_within_region_pretrained_baselines.json
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
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if CLUSTER:
    data_dir = "/svi_data"
    log_dir = "/logs"
    output_dir = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "PP2_perception",
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
        "PP2_perception",
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


def train_evaluate_pipeline(
    locations_list: list,
    input_features: np.ndarray,
    gdf_task: gpd.GeoDataFrame,
    settings_name: str,
    modality_name: str,
    data_loader_name: str,
    downstream_task: str,
) -> pd.DataFrame:
    """
    Preprocess and run regression pipeline (MLP & Ridge)

    Parameters
    ----------
    locations_list : list
        List of location strings in the format "lat_lon_location_id_city".
    input_features : np.ndarray
        Input features for the regression model, if any.
    gdf_task : gpd.GeoDataFrame
        GeoDataFrame containing the task-specific data.
    settings_name : str
        Name of the settings used for this evaluation.
    modality_name : str
        Name of the modality being evaluated.
    data_loader_name : str
        Name of the data loader used for this evaluation.
    downstream_task : str
        Name of the downstream task being evaluated.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the regression evaluation.
    """
    # Parse and filter locations
    parsed = []
    for idx, item in enumerate(locations_list):
        lat, lon, loc_id, city = item.split("_")
        parsed.append(
            {
                "original_index": idx,
                "latitude": float(lat),
                "longitude": float(lon),
                "location_id": loc_id,
                "city_name": city,
            }
        )
    loc_df = pd.DataFrame(parsed)

    # Merge predicted locations with task data
    gdf = pd.merge(loc_df, gdf_task, on="location_id", how="left")

    # Add coordinates as features
    if input_features is None:
        X = gdf[["longitude", "latitude"]].values
    else:
        X = np.hstack([input_features, gdf[["longitude", "latitude"]].values])

    # Add target variable
    y = gdf["trueskill.score"].values

    # Run regression pipelines
    print("Number of samples:", len(y), flush=True)
    results = []
    if MODEL == "mlp" or MODEL == "both":
        # MLP
        df_mlp = run_pipeline(
            X,
            y,
            task_type="regression",
            model_type="mlp",
            epochs=40,
            standardize=True,
            alpha_range=(1e-4, 10000.0),
            lr_range=(1e-5, 1e-1),
            weight_decay_range=(1e-6, 1e-1),
        )
        df_mlp["settings_name"] = settings_name
        df_mlp["modality_name"] = modality_name
        df_mlp["data_loader_name"] = data_loader_name
        df_mlp["model"] = "mlp"
        df_mlp["task"] = downstream_task
        results.append(df_mlp)
    if MODEL == "ridge" or MODEL == "both":
        # Ridge
        df_ridge = run_pipeline(
            X,
            y,
            task_type="regression",
            model_type="ridge",
            standardize=True,  # Last two features
            alpha_range=(1e-4, 10000.0),
            lr_range=(1e-5, 1e-1),
            weight_decay_range=(1e-6, 1e-1),
        )
        df_ridge["settings_name"] = settings_name
        df_ridge["modality_name"] = modality_name
        df_ridge["data_loader_name"] = data_loader_name
        df_ridge["model"] = "ridge"
        df_ridge["task"] = downstream_task
        results.append(df_ridge)

    return pd.concat(results, ignore_index=True)


def main():
    # Load data
    df_locations = pd.read_csv(
        os.path.join(data_dir, "place-pulse-2.0", "locations.tsv"), sep="\t"
    )
    df_locations = df_locations.rename(columns={"_id": "location_id"})
    df_places = pd.read_csv(
        os.path.join(data_dir, "place-pulse-2.0", "places.tsv"), sep="\t"
    )
    df_places = df_places.rename(columns={"_id": "place_id"})
    df_scores = pd.read_csv(
        os.path.join(data_dir, "place-pulse-2.0", "qscores.tsv"), sep="\t"
    )
    df_study = pd.read_csv(
        os.path.join(data_dir, "place-pulse-2.0", "studies.tsv"), sep="\t"
    )
    df_study = df_study.rename(columns={"_id": "study_id"})
    df_all = pd.merge(df_locations, df_places, on="place_id", how="left")
    df_all = pd.merge(df_all, df_scores, on="location_id", how="left")
    df_all = pd.merge(df_all, df_study, on="study_id", how="left")
    gdf = gpd.GeoDataFrame(
        df_all,
        geometry=gpd.points_from_xy(df_all["loc.1"], df_all["loc.0"]),
        crs="EPSG:4326",
    )

    all_runs = []
    in_region_locs = out_region_locs = None
    for downstream_task in [
        "cleaner_multicity",
        "depressing_multicity",
        "more beautiful",
        "safer_multicity",
        "livelier_multicity",
        "wealthy_multicity",
    ]:
        gdf_task = gdf[gdf["study_name"] == downstream_task].copy()

        for name, info in REPRESENTATIONS_TO_EVALUATE.items():
            date = info["date"]
            modalities = info["modalities"]
            mod_names = info["modalities_name"]
            epoch = info["epoch"]
            dls = info["data_loaders"]

            if name == "Coordinates":
                for dl in dls:
                    if dl == 0:
                        locs = in_region_locs
                    else:
                        locs = out_region_locs
                    res_df = train_evaluate_pipeline(
                        locs,
                        None,
                        gdf_task,
                        name,
                        "Coords",
                        DATA_LOADERS_NAME[dl],
                        downstream_task,
                    )
                    all_runs.append(res_df)
                continue

            for dl in dls:
                dl_name = DATA_LOADERS_NAME[dl]
                path = os.path.join(
                    log_dir,
                    name,
                    "runs",
                    date,
                    f"plots/modalities_test_dl_{dl}.pt",
                )
                loc_list = torch.load(path, map_location="cpu")
                loc_list = [i for sub in loc_list for i in sub]
                if dl == 0:
                    in_region_locs = loc_list
                else:
                    out_region_locs = loc_list

                for m_idx in modalities:
                    m_name = mod_names[modalities.index(m_idx)]
                    feat_file = (
                        (
                            f"representations_test_epoch_{epoch}"
                            f"_masked_modality_{m_idx}_dl_{dl}.pt"
                        )
                        if "UrbanFusion" in name or "Raw" in name
                        else (
                            f"representations_test_epoch_{epoch}_"
                            f"modality_{m_idx}_dl_{dl}.pt"
                        )
                    )
                    feats = torch.load(
                        os.path.join(
                            log_dir, name, "runs", date, "plots", feat_file
                        ),
                        map_location="cpu",
                    )
                    all_runs.append(
                        train_evaluate_pipeline(
                            loc_list,
                            feats,
                            gdf_task,
                            name,
                            m_name,
                            dl_name,
                            downstream_task,
                        )
                    )

    results_df = pd.concat(all_runs, ignore_index=True)
    return results_df


if __name__ == "__main__":
    start = time.time()
    df_results = main()
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

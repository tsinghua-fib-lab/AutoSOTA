#!/usr/bin/env python3
"""
Description: Implementation of London_energy task.

Usage:
-- settings [config json name] -- model [linear or mlp]

Example:
python scripts/downstream_tasks/London_energy.py --settings London_energy_within_region_pretrained_baselines.json
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
    description="Run London_energy with a specified settings JSON"
)
parser.add_argument(
    "-s",
    "--settings",
    default="London_energy_within_region_pretrained_baselines_old.json",
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
        "London_energy",
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
        "London_energy",
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
    Train and evaluate downstream learners.
    """
    df_energy = pd.read_csv(
        os.path.join(
            data_dir,
            "place-pulse-2.0/downstreamtask_data/London_Energy_Usage/London_Energy_Usage_locations.csv",
        )
    )
    grouped = (
        df_energy.groupby(["latitude", "longitude"])
        .agg({"Num_meters": "sum", "Total_cons_kwh": "sum"})
        .reset_index()
    )
    grouped["Mean_cons_kwh"] = (
        grouped["Total_cons_kwh"] / grouped["Num_meters"]
    )
    unique_locs = df_energy.drop_duplicates(subset=["latitude", "longitude"])[
        ["latitude", "longitude"]
    ]
    X_tabular = unique_locs[["longitude", "latitude"]].values
    merged = pd.merge(
        unique_locs, grouped, on=["latitude", "longitude"], how="left"
    )
    y = merged["Mean_cons_kwh"].values

    # Loop through representations
    def load_rep(name, info):
        date, epoch = info["date"], info["epoch"]
        if date is None:
            return None
        fname = (
            f"representations_test_epoch_{epoch}_masked_modality_1_2_3_4_dl_London_Energy_Usage_locations.pt"
            if "UrbanFusion" in name
            else f"representations_test_epoch_{epoch}_modality_0_dl_London_Energy_Usage_locations.pt"
        )
        path = os.path.join(log_dir, name, "runs", date, "plots", fname)
        feats = torch.load(path, map_location="cpu")
        return feats.numpy()

    all_runs = []
    for name, info in REPRESENTATIONS_TO_EVALUATE.items():
        feats = load_rep(name, info)
        if feats is not None:
            # Only keep one representation per unique location
            feats = feats[
                ~df_energy.duplicated(subset=["latitude", "longitude"])
            ]

            # Double-check alignment
            assert (
                feats.shape[0] == X_tabular.shape[0]
            ), "Mismatch in feature and location dimensions"

            X = np.hstack([feats, X_tabular])
        else:
            X = X_tabular

        print(f"=== Evaluation for {name} ===")
        if MODEL == "mlp" or MODEL == "both":
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
            df_mlp["representation"] = name
            df_mlp["model"] = "mlp"
            all_runs.append(df_mlp)
        if MODEL == "ridge" or MODEL == "both":
            df_ridge = run_pipeline(
                X,
                y,
                task_type="regression",
                model_type="ridge",
                standardize=True,
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

#!/usr/bin/env python3
"""
Description: Implementation of crime incidence tasks.

Usage:
-- settings [config json name] -- model [linear or mlp]
Example:
python scripts/downstream_tasks/crime_usa.py --settings crime_usa_within_region_UrbanFusion_trained_1_ridge.json
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
    description="Run crime usa with a specified settings JSON"
)
parser.add_argument(
    "-s",
    "--settings",
    default="crime_usa_within_region_pretrained_baselines_old.json",
    help="Path to the settings JSON file",
)
parser.add_argument(
    "-m", "--model", default="both", help="Path to the settings JSON file"
)
args = parser.parse_args()
SETTINGS_NAME = args.settings
MODEL = args.model

# Configuration
BUFFER = 500
CLUSTER = False
US_CITIES = [
    "Boston",
    "Chicago",
    "Houston",
    "LosAngeles",
    "Minneapolis",
    "NewYork",
    "SanFrancisco",
    "Seattle",
]
US_CITIES_CORRECTED = {
    "Boston": "Boston",
    "Chicago": "Chicago",
    "Houston": "Houston",
    "LosAngeles": "Los Angeles",
    "Minneapolis": "Minneapolis",
    "NewYork": "New York",
    "SanFrancisco": "San Francisco",
    "Seattle": "Seattle",
}
# Remove samples (ensure compatibility with PDFM)
SAMPLES_REMOVE = [
    11,
    118,
    599,
    833,
    840,
    874,
    1395,
    1520,
    1529,
    1554,
    1585,
    1601,
    1739,
    1772,
    1837,
    1865,
    2082,
]
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
        "crime_usa",
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
        "crime_usa",
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


def train_evaluate_crime_with_pipeline(
    locations_list: list,
    input_features: np.ndarray,
    gdf_crimes: gpd.GeoDataFrame,
    us_cities: list,
    us_cities_corrected: dict,
    settings_name: str,
    modality_name: str,
    data_loader_name: str,
    buffer_val: float,
) -> pd.DataFrame:
    """
    Preprocess crime counts and run regression pipeline (MLP & Ridge)

    Parameters
    ----------
    locations_list : list
        List of location strings in the format "lat_lon_loc_id_city"
    input_features : np.ndarray
        Array of input features for the model
    gdf_crimes : gpd.GeoDataFrame
        GeoDataFrame containing crime data
    us_cities : list
        List of US cities to consider
    us_cities_corrected : dict
        Mapping of city names to their corrected versions
    settings_name : str
        Name of the settings file
    modality_name : str
        Name of the modality (e.g., "In-region" or "Out-of-region")
    data_loader_name : str
        Name of the data loader
    buffer_val : float
        Buffer value for spatial queries

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results of the evaluation
    """
    # Parse and filter locations ---
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
    loc_df = loc_df[loc_df["city_name"].isin(us_cities)].reset_index(drop=True)

    # UTM EPSG calculation
    def utm_epsg(lon: float, lat: float) -> int:
        """
        Calculate the UTM EPSG code for a given longitude and latitude.

        Parameters
        ----------
        lon : float
            Longitude of the location
        lat : float
            Latitude of the location

        Returns
        -------
        int
            UTM EPSG code for the location
        """
        zone = int((lon + 180) / 6) + 1
        return 32600 + zone if lat >= 0 else 32700 + zone

    loc_df["utm_epsg"] = loc_df.apply(
        lambda r: utm_epsg(r.longitude, r.latitude), axis=1
    )

    # GeoDataFrame of locations
    locs_gdf = gpd.GeoDataFrame(
        loc_df,
        geometry=gpd.points_from_xy(loc_df.longitude, loc_df.latitude),
        crs="EPSG:4326",
    )

    # Spatial join to count crimes in buffer
    all_counts = []
    for city in us_cities:
        city_locs = locs_gdf[locs_gdf.city_name == city]
        city_crimes = gdf_crimes[
            gdf_crimes.city_name == us_cities_corrected.get(city, city)
        ]
        if city_locs.empty or city_crimes.empty:
            continue
        utm = city_locs.iloc[0].utm_epsg
        cl = city_locs.to_crs(epsg=utm).copy()
        cc = city_crimes.to_crs(epsg=utm)
        cl.geometry = cl.buffer(buffer_val)
        joined = gpd.sjoin(cl, cc, predicate="contains", how="left")
        cnts = (
            joined.groupby("original_index")
            .size()
            .reset_index(name="crime_count")
        )
        res = (
            loc_df.loc[loc_df.city_name == city][
                [
                    "original_index",
                    "location_id",
                    "city_name",
                    "latitude",
                    "longitude",
                ]
            ]
            .merge(cnts, on="original_index", how="left")
            .fillna(0)
        )
        all_counts.append(res)

    if not all_counts:
        return pd.DataFrame([])

    df_counts = pd.concat(all_counts, ignore_index=True)
    df_counts["crime_count"] = df_counts.crime_count.astype(int)

    # Build feature matrix X and target y
    idxs = df_counts.original_index.values
    if input_features is None:
        X = df_counts[["longitude", "latitude"]].values
    else:
        feats = input_features[idxs]
        X = np.hstack([feats, df_counts[["longitude", "latitude"]].values])
    y = np.log1p(df_counts["crime_count"].values)
    print(len(X), len(y), flush=True)

    # Drop samples for In-region
    if data_loader_name == DATA_LOADERS_NAME[0]:
        mask = np.ones(len(y), dtype=bool)
        mask[SAMPLES_REMOVE] = False
        X, y = X[mask], y[mask]

    # Run regression pipelines
    print("Number of samples:", len(y), flush=True)
    results = []
    # MLP
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
        df_mlp["settings_name"] = settings_name
        df_mlp["modality_name"] = modality_name
        df_mlp["data_loader_name"] = data_loader_name
        df_mlp["model"] = "mlp"
        results.append(df_mlp)

    # Ridge
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
        df_ridge["settings_name"] = settings_name
        df_ridge["modality_name"] = modality_name
        df_ridge["data_loader_name"] = data_loader_name
        df_ridge["model"] = "ridge"
        results.append(df_ridge)

    # Kernel Ridge (Nystroem + Ridge)
    if MODEL == "kernel_ridge" or MODEL == "both":
        df_kr = run_pipeline(
            X,
            y,
            task_type="regression",
            model_type="kernel_ridge",
            standardize=True,
            alpha_range=(1e-4, 10000.0),
        )
        df_kr["settings_name"] = settings_name
        df_kr["modality_name"] = modality_name
        df_kr["data_loader_name"] = data_loader_name
        df_kr["model"] = "kernel_ridge"
        results.append(df_kr)

    return pd.concat(results, ignore_index=True)


def main():
    # Load crimes
    df_cr = pd.read_csv(
        os.path.join(
            data_dir,
            "place-pulse-2.0",
            "downstreamtask_data",
            "crime_usa",
            "crime_open_database_core_2021.csv",
        )
    )
    gdf_cr = gpd.GeoDataFrame(
        df_cr,
        geometry=gpd.points_from_xy(df_cr.longitude, df_cr.latitude),
        crs="EPSG:4326",
    )

    all_runs = []
    in_region_locs = out_region_locs = None

    for name, info in REPRESENTATIONS_TO_EVALUATE.items():
        date = info["date"]
        modalities = info["modalities"]
        mod_names = info["modalities_name"]
        epoch = info["epoch"]
        dls = info["data_loaders"]

        # Coordinates-only
        if name == "Coordinates":
            for dl in dls:
                if dl == 0:
                    locs = in_region_locs
                else:
                    locs = out_region_locs
                res_df = train_evaluate_crime_with_pipeline(
                    locs,
                    None,
                    gdf_cr,
                    US_CITIES,
                    US_CITIES_CORRECTED,
                    name,
                    "Coordinates",
                    DATA_LOADERS_NAME[dl],
                    BUFFER,
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
                    f"representations_test_epoch_{epoch}_masked_modality_{m_idx}_dl_{dl}.pt"
                    if "UrbanFusion" in name or "Raw" in name
                    else f"representations_test_epoch_{epoch}_modality_{m_idx}_dl_{dl}.pt"
                )
                feats = torch.load(
                    os.path.join(
                        log_dir, name, "runs", date, "plots", feat_file
                    ),
                    map_location="cpu",
                )
                all_runs.append(
                    train_evaluate_crime_with_pipeline(
                        loc_list,
                        feats,
                        gdf_cr,
                        US_CITIES,
                        US_CITIES_CORRECTED,
                        name,
                        m_name,
                        dl_name,
                        BUFFER,
                    )
                )

    results_df = pd.concat(all_runs, ignore_index=True)
    print(results_df.head())
    return results_df


if __name__ == "__main__":
    start = time.time()
    df_results = main()
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

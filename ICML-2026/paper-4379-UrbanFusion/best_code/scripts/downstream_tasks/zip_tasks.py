#!/usr/bin/env python3
"""
Description: Implementation of zip code downstream tasks.
Predictions are averaged across all locations within the same ZIP code during
inference.

Usage:
-- settings [config json name] -- model [linear or mlp]

Example:
python scripts/downstream_tasks/zip_tasks.py --settings zip_tasks_within_region_UrbanFusion_trained_1_MLP.json --model mlp
"""
import os

print("Current working directory:", os.getcwd(), flush=True)
import argparse
import datetime
import json
import os
import random
import time
import warnings

import geopandas as gpd
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.samplers import TPESampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from scripts.downstream_tasks.downstream_eval import (
    MLP,
    SEED,
    EarlyStopping,
    cosine_scheduler,
    evaluate_mlp,
    standardize_all,
    standardize_and_rescale_last_n,
    train_mlp,
)

warnings.filterwarnings("ignore")

# Set deterministic behavior for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Input parser
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

# Configurations
CLUSTER = True
DATA_LOADERS_NAME = ["In-region", "Out-of-region"]
# Remove samples (PDFM does not provide embeddings for those samples)
SAMPLES_REMOVE_DL0 = [
    1,
    9,
    65,
    77,
    138,
    160,
    212,
    282,
    366,
    369,
    532,
    549,
    596,
    627,
    634,
    656,
    679,
    696,
    766,
    841,
    944,
    1021,
    1024,
    1121,
    1163,
    1180,
    1259,
    1307,
    1310,
    1332,
    1472,
    1568,
    1575,
    1854,
    1957,
    1968,
    2023,
    2078,
    2179,
    2187,
    2216,
    2280,
    2305,
    2342,
    2344,
    2421,
    2449,
    2560,
    2573,
    2754,
    3029,
    3054,
    3061,
    3105,
    3106,
    3190,
    3359,
    3454,
    3853,
    3879,
    3880,
    3895,
    3912,
    4013,
    4347,
    4402,
    4537,
    4643,
    4647,
    4775,
    4866,
]
SAMPLES_REMOVE_DL1 = [
    17,
    25,
    43,
    53,
    71,
    82,
    83,
    107,
    146,
    205,
    285,
    304,
    331,
    373,
    414,
    500,
    518,
    527,
    528,
    531,
    537,
    550,
    667,
    668,
    702,
    713,
    747,
    760,
    776,
    794,
    838,
    854,
    856,
    952,
    967,
    1099,
    1106,
    1121,
    1123,
    1131,
    1134,
    1173,
    1207,
    1217,
    1236,
    1245,
    1394,
    1402,
    1409,
    1433,
    1492,
    1538,
    1559,
    1599,
    1659,
    1666,
    1681,
    1686,
    1716,
    1730,
    1731,
    1753,
    1757,
    1772,
    1777,
    1794,
    1824,
    1857,
    1858,
    1869,
    1923,
    1954,
    1956,
    1994,
    2057,
    2098,
    2171,
    2300,
    2373,
    2374,
    2394,
    2398,
    2475,
    2566,
    2574,
    2575,
    2604,
    2631,
    2653,
    2679,
    2681,
    2724,
    2817,
    2830,
    2831,
    2834,
    2838,
    2872,
    2902,
    2997,
    3022,
    3032,
    3059,
    3063,
    3109,
    3131,
    3200,
    3223,
    3224,
    3231,
    3260,
    3276,
    3292,
    3306,
    3316,
    3332,
]
# Downstream tasks
DOWNSTREAM_TASKS = [
    "Percent_Person_WithHighCholesterol",
    "Percent_Person_WithPhysicalHealthNotGood",
    "Percent_Person_WithStroke",
    "Percent_Person_BingeDrinking",
    "Percent_Person_PhysicalInactivity",
    "Percent_Person_ReceivedAnnualCheckup",
    "Percent_Person_WithCancerExcludingSkinCancer",
    "Percent_Person_WithDiabetes",
    "Percent_Person_WithMentalHealthNotGood",
    "Percent_Person_WithCoronaryHeartDisease",
    "Percent_Person_WithHighBloodPressure",
    "Percent_Person_ReceivedCholesterolScreening",
    "Percent_Person_ReceivedDentalVisit",
    "Percent_Person_WithAsthma",
    "Percent_Person_WithChronicKidneyDisease",
    "Percent_Person_WithArthritis",
    "Percent_Person_WithChronicObstructivePulmonaryDisease",
    "Percent_Person_18OrMoreYears_WithHighBloodPressure_ReceivedTakingBloodPressureMedication",
    "Percent_Person_Obesity",
    "Percent_Person_SleepLessThan7Hours",
    "Percent_Person_Smoking",
    "Median_Income_Household",
    "Median_HomeValue_HousingUnit_OccupiedHousingUnit_OwnerOccupied",
    "night_lights_log10",
    "population_density_log10",
    "poverty_rate",
    "tree_cover",
    "elevation",
]

# US Cities to consider
US_CITIES = [
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
    "NewYork",
]

# Get current date and time
date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Paths for local machine and HPC cluster
if CLUSTER:
    data_dir = "/svi_data"
    log_dir = "/logs"
    output_dir = os.path.join(
        data_dir,
        "place-pulse-2.0",
        "downstreamtask_data",
        "results",
        "zip_tasks",
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
        "zip_tasks",
    )
    settings_path = os.path.join(output_dir, SETTINGS_NAME)
    output_path = os.path.join(
        output_dir, SETTINGS_NAME[:-5] + f"_{MODEL}.csv"
    )

# Read config json
if not os.path.isfile(settings_path):
    raise FileNotFoundError(f"Settings file not found: {settings_path}")
with open(settings_path) as f:
    REPRESENTATIONS_TO_EVALUATE = json.load(f)


def train_zip_mlp(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epochs: int,
    patience: int = 10,
    mode: str = "min",
    loss_function: callable = mean_squared_error,
    groups_val: np.ndarray = None,
) -> tuple:
    """
    Train the MLP model with early stopping using a selectable metric.

    Parameters
    ----------
    model : nn.Module
        The neural network model to train.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    criterion : nn.Module
        Loss function to use.
    optimizer : optim.Optimizer
        Optimizer for updating model weights.
    scheduler : optim.lr_scheduler._LRScheduler
        Learning rate scheduler.
    epochs : int
        Number of training epochs.
    patience : int, optional
        Patience for early stopping (default is 10).
    mode : str, optional
        Mode for early stopping (default is 'min').
    loss_function : callable, optional
        Loss function to use (default is mean_squared_error).
    groups_val : np.ndarray, optional
        Group labels for validation data (default is None).

    Returns
    -------
    tuple
        Trained model and best validation score.
    """
    model.to(device)
    early_stopping = EarlyStopping(patience=patience, mode=mode)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        # Validation
        if val_loader is not None:
            # grouped validation MSE
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    out = model(xb).cpu().numpy().squeeze()
                    preds.append(out)
                    truths.append(yb.numpy())
            preds = np.concatenate(preds)
            truths = np.concatenate(truths)
            val_metric, _ = loss_function(preds, truths, groups_val)
            early_stopping.step(val_metric, model)
            if early_stopping.early_stop:
                break
        else:
            val_metric = None

    if val_loader is not None:
        early_stopping.restore(model)
        return model, early_stopping.best_score
    else:
        return model, None


def train_evaluate_prediction(
    locations_list: list,
    input_features: np.ndarray,
    conus_df: gpd.GeoDataFrame,
    us_cities: list,
    settings_name: str,
    modality_name: str,
    data_loader_name: str,
    downstream_task: str,
    model_type: str = "ridge",
    n_trials: int = 20,
    random_seed: int = 42,
    standardize: int = None,
    epochs: int = 40,
):
    """
    Trains and evaluates a model using the specified parameters and data.

    Parameters
    ----------
    locations_list : list
        List of location identifiers.
    input_features : np.ndarray
        Array of input features for the model.
    conus_df : gpd.GeoDataFrame
        GeoDataFrame containing CONUS region data.
    us_cities : list
        List of US city names.
    settings_name : str
        Name of the settings to use.
    modality_name : str
        Name of the modality to use.
    data_loader_name : str
        Name of the data loader to use.
    downstream_task : str
        Name of the downstream task to perform.
    model_type : str, optional
        Type of model to use (default is "ridge").
    n_trials : int, optional
        Number of trials to run in optuna (default is 20).
    random_seed : int, optional
        Random seed for reproducibility (default is 42).
    standardize : int, optional
        Whether to standardize the data (default is None).
    epochs : int, optional
        Number of training epochs (default is 40).

    Returns
    -------
    dict
        Keys: Name, downstream_task, settings_name, modality_name,
              data_loader_name, model_type, best_params,
              val_mse, r2_val, test_mse, r2_test.
    """
    # Parse locations
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
    locations_df = pd.DataFrame(parsed)
    locations_df = locations_df[
        locations_df["city_name"].isin(us_cities)
    ].reset_index(drop=True)
    locations_gdf = gpd.GeoDataFrame(
        locations_df,
        geometry=gpd.points_from_xy(
            locations_df["longitude"], locations_df["latitude"]
        ),
        crs="EPSG:4326",
    )
    conus_df = conus_df.to_crs("EPSG:4326").drop(
        columns=["longitude", "latitude"], errors="ignore"
    )
    join_cols = ["place", "geometry", downstream_task]
    result_gdf = gpd.sjoin(
        locations_gdf, conus_df[join_cols], how="left", predicate="within"
    )
    result_gdf["original_index"] = result_gdf["original_index"].astype(int)
    fi = result_gdf["original_index"].values

    # Build feature matrix
    if input_features is None:
        X = result_gdf[["longitude", "latitude"]].values
    else:
        X = np.concatenate(
            [input_features[fi], result_gdf[["longitude", "latitude"]].values],
            axis=1,
        )
    # Get labels
    y = result_gdf[downstream_task].values.astype(float)

    # Filter NaNs
    nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
    if nan_mask.any():
        print(
            f"Warning: {nan_mask.sum()} NaNs in features or target "
            f"'{downstream_task}'"
        )
        print("Index of NaNs:", np.where(nan_mask)[0], flush=True)
        # Check if NaNs are in features or target
        if np.isnan(X).any(axis=1).any():
            print("NaNs found in features.")
        if np.isnan(y).any():
            print("NaNs found in target.")

    # In-region / out-of-region filter
    if data_loader_name == DATA_LOADERS_NAME[0]:  # In-region
        mask = np.ones(len(y), dtype=bool)
        mask[SAMPLES_REMOVE_DL0] = False
    else:  # Out-of-region
        mask = np.ones(len(y), dtype=bool)
        mask[SAMPLES_REMOVE_DL1] = False
    X, y, result_gdf = X[mask], y[mask], result_gdf.loc[mask]
    groups = result_gdf["place"].values

    assert np.allclose(
        X[:, -2:], result_gdf[["longitude", "latitude"]].to_numpy()
    )
    # Split into train/val/test, split by ZIP codes rather than locations
    gss = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=random_seed
    )
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=random_seed
    )
    tr_rel, val_rel = next(
        gss2.split(
            X[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]
        )
    )
    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[val_rel]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    groups_val = groups[val_idx]
    groups_test = groups[test_idx]

    if standardize is not None:
        X_train, X_test, X_val = standardize_all(X_train, X_test, X_val)

    # Calculate scores per ZIP code for inference
    def grouped_scores_array(preds, truths, grps):
        df = pd.DataFrame({"grp": grps, "y_true": truths, "y_pred": preds})
        agg = df.groupby("grp").mean()
        return (
            mean_squared_error(agg.y_true, agg.y_pred),
            r2_score(agg.y_true, agg.y_pred),
        )

    # Set random seeds for reproducibility
    input_dim = X.shape[1]
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Training ridge regression
    if MODEL == "ridge":

        def objective(trial):
            alpha = trial.suggest_float("alpha", 1e-4, 10000.0, log=True)
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse, _ = grouped_scores_array(preds, y_val, groups_val)
            return mse

    # Training MLP
    elif MODEL == "mlp":  # MLP regression
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # prepare tensors once
        train_ds = TensorDataset(
            torch.tensor(X_train).float(), torch.tensor(y_train).float()
        )
        val_ds = TensorDataset(
            torch.tensor(X_val).float(), torch.tensor(y_val).float()
        )
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            wd = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)

            model = MLP(
                input_dim=input_dim, output_dim=1, task_type="regression"
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd
            )
            scheduler = cosine_scheduler(optimizer, epochs)
            criterion = nn.MSELoss()
            g = torch.Generator().manual_seed(random_seed)
            train_loader = DataLoader(
                train_ds, batch_size=64, shuffle=True, generator=g
            )
            val_loader = DataLoader(
                val_ds, batch_size=64, shuffle=False, generator=g
            )

            # Train and validate
            _, mse = train_zip_mlp(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                loss_function=grouped_scores_array,
                groups_val=groups_val,
            )
            return mse

    # Hyperparameter tuning
    n_jobs = 1 if model_type == "mlp" else -1
    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(seed=random_seed)
    )
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False, n_jobs=n_jobs
    )
    best_params = study.best_params
    best_val = study.best_value
    seeds = [random_seed] if model_type == "ridge" else [0, 1, 2, 3, 4]
    results = []
    # Train a model for each random seed using best hyperparameters
    for seed in seeds:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if MODEL == "ridge":
            model = Ridge(alpha=best_params["alpha"])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse_test, r2_test = grouped_scores_array(
                preds, y_test, groups_test
            )
            results.append(
                {
                    "settings_name": settings_name,
                    "downstream_task": downstream_task,
                    "modality_name": modality_name,
                    "data_loader_name": data_loader_name,
                    "model": model_type,
                    "best_params": best_params,
                    "seed": seed,
                    "mse": mse_test,
                    "r2": r2_test,
                    "val_mse": best_val,
                }
            )

        elif MODEL == "mlp":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            # Prepare DataLoaders
            train_ds = TensorDataset(
                torch.tensor(X_train).float(), torch.tensor(y_train).float()
            )
            val_ds = TensorDataset(
                torch.tensor(X_val).float(), torch.tensor(y_val).float()
            )
            test_ds = TensorDataset(
                torch.tensor(X_test).float(), torch.tensor(y_test).float()
            )
            # Create a generator seeded to `seed`
            g = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_ds, batch_size=64, shuffle=True, generator=g
            )
            val_loader = DataLoader(
                val_ds, batch_size=64, shuffle=False, generator=g
            )
            test_loader = DataLoader(
                test_ds, batch_size=64, shuffle=False, generator=g
            )

            # initialize and train MLP
            mlp = MLP(
                input_dim=input_dim, output_dim=1, task_type="regression"
            ).to(device)
            optimizer = torch.optim.AdamW(
                mlp.parameters(),
                lr=best_params["lr"],
                weight_decay=best_params["weight_decay"],
            )
            scheduler = cosine_scheduler(optimizer, epochs)
            criterion = nn.MSELoss()

            mlp, _ = train_zip_mlp(
                mlp,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                epochs,
                loss_function=grouped_scores_array,
                groups_val=groups_val,
            )

            mlp.eval()
            preds, truths = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    out = mlp(xb).cpu().numpy().squeeze()
                    preds.append(out)
                    truths.append(yb.numpy())
            preds = np.concatenate(preds)
            truths = np.concatenate(truths)
            mse, r2 = grouped_scores_array(preds, truths, groups_test)
            results.append(
                {
                    "settings_name": settings_name,
                    "downstream_task": downstream_task,
                    "modality_name": modality_name,
                    "data_loader_name": data_loader_name,
                    "model": model_type,
                    "best_params": best_params,
                    "seed": seed,
                    "mse": mse,
                    "r2": r2,
                    "val_mse": best_val,
                }
            )

    results_df = pd.DataFrame(results)
    return results_df


def main():
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load downstream task data
    df_conus27 = pd.read_csv(
        os.path.join(
            data_dir, "place-pulse-2.0", "downstreamtask_data", "conus27.csv"
        )
    )

    # Load zip code geometries
    gdf_zip_codes = gpd.read_file(
        os.path.join(data_dir, "place-pulse-2.0", "pdfm", "zcta.geojson")
    )
    df_conus27 = pd.merge(
        df_conus27,
        gdf_zip_codes[["place", "geometry"]],
        on="place",
        how="left",
    )
    df_conus27 = gpd.GeoDataFrame(
        df_conus27, geometry="geometry", crs="EPSG:4326"
    )
    # Load poverty data
    df_poverty = pd.read_csv(
        os.path.join(
            data_dir,
            "place-pulse-2.0",
            "downstreamtask_data",
            "Poverty_US",
            "zcta_poverty.csv",
        )
    )
    gdf_zip_codes = gpd.read_file(
        os.path.join(data_dir, "place-pulse-2.0", "pdfm", "zcta.geojson")
    )
    df_poverty = pd.merge(
        df_poverty,
        gdf_zip_codes[["place", "geometry"]],
        on="place",
        how="left",
    )
    df_poverty.rename(columns={"2022": "poverty_rate"}, inplace=True)
    df_poverty = df_poverty[["place", "poverty_rate"]]
    df_conus27 = pd.merge(
        df_conus27,
        df_poverty[["place", "poverty_rate"]],
        on="place",
        how="left",
    )
    df_conus27 = gpd.GeoDataFrame(
        df_conus27, geometry="geometry", crs="EPSG:4326"
    )

    results_list = []
    location_list_in_region = None
    location_list_out_region = None

    for downstream_task in DOWNSTREAM_TASKS:
        print(
            f"\n=== Evaluating downstream task: {downstream_task} ===\n",
            flush=True,
        )
        for (
            settings_name,
            settings_info,
        ) in REPRESENTATIONS_TO_EVALUATE.items():
            # Check if modality_ids is a key in settings_info
            if "modality_ids" in settings_info:
                modalities = settings_info["modality_ids"][downstream_task]
                modalities_name_list = settings_info[
                    "modality_names_per_target"
                ][downstream_task]
                if not isinstance(modalities_name_list, list):
                    modalities_name_list = [modalities_name_list]
                if not isinstance(modalities, list):
                    modalities = [modalities]
            else:
                modalities = settings_info["modalities"]
                modalities_name_list = settings_info["modalities_name"]

            date_str = settings_info["date"]
            epoch = settings_info["epoch"]
            setting_data_loaders = settings_info["data_loaders"]

            if settings_name == "Coordinates":
                for dl_idx in setting_data_loaders:
                    data_loader_name = DATA_LOADERS_NAME[dl_idx]
                    print("", flush=True)
                    print(
                        (
                            f"Processing {downstream_task}, {settings_name}, "
                            f"{data_loader_name}, Coordinates ..."
                        ),
                        flush=True,
                    )
                    if dl_idx == 0:
                        location_input_coords = location_list_in_region
                    elif dl_idx == 1:
                        location_input_coords = location_list_out_region

                    row = train_evaluate_prediction(
                        locations_list=location_input_coords,
                        input_features=None,
                        conus_df=df_conus27,
                        us_cities=US_CITIES,
                        settings_name=settings_name,
                        modality_name="Coords",
                        data_loader_name=data_loader_name,
                        downstream_task=downstream_task,
                        model_type=MODEL,
                        n_trials=20,
                        random_seed=SEED,
                        standardize=True,
                        epochs=40,
                    )
                    results_list.append(row)
                continue

            for dl_idx in setting_data_loaders:
                data_loader_name = DATA_LOADERS_NAME[dl_idx]
                locations_path = os.path.join(
                    log_dir,
                    settings_name,
                    "runs",
                    date_str,
                    f"plots/modalities_test_dl_{dl_idx}.pt",
                )
                locations_list = torch.load(
                    locations_path, map_location=torch.device("cpu")
                )
                locations_list = [
                    item for sublist in locations_list for item in sublist
                ]
                if dl_idx == 0 and location_list_in_region is None:
                    location_list_in_region = locations_list
                elif dl_idx == 1 and location_list_out_region is None:
                    location_list_out_region = locations_list

                for modality_idx in modalities:
                    idx_in_list = modalities.index(modality_idx)
                    modality_name = modalities_name_list[idx_in_list]
                    print("", flush=True)
                    print(
                        f"Processing {downstream_task}, {settings_name}, {data_loader_name}, {modality_name} ...",
                        flush=True,
                    )

                    # 3) Determine the correct file name for input_features
                    if (
                        "UrbanFusion" in settings_name
                        or "Raw" in settings_name
                    ):
                        feat_filename = f"representations_test_epoch_{epoch}_masked_modality_{modality_idx}_dl_{dl_idx}.pt"
                    else:
                        feat_filename = f"representations_test_epoch_{epoch}_modality_{modality_idx}_dl_{dl_idx}.pt"

                    input_features_path = os.path.join(
                        log_dir,
                        settings_name,
                        "runs",
                        date_str,
                        "plots",
                        feat_filename,
                    )
                    input_features = torch.load(
                        input_features_path, map_location=torch.device("cpu")
                    )
                    results_list.append(
                        train_evaluate_prediction(
                            locations_list=locations_list,
                            input_features=input_features,
                            conus_df=df_conus27,
                            us_cities=US_CITIES,
                            settings_name=settings_name,
                            modality_name=modality_name,
                            data_loader_name=data_loader_name,
                            downstream_task=downstream_task,
                            model_type=MODEL,
                            n_trials=20,
                            random_seed=SEED,
                            standardize=True,
                            epochs=40,
                        )
                    )
    results_df = pd.concat(results_list, ignore_index=True)
    return results_df


if __name__ == "__main__":
    start = time.time()
    df_results = main()
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}", flush=True)
    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds", flush=True)

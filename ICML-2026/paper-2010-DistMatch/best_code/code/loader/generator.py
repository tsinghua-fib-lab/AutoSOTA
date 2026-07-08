from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch

from enbPI.wrappers.enbPI_loader import EnbPI_DATASETS, demo_code_data_load
from config import TSDataConfig, TaskConfig
from loader.dataset import ChronoSplittedTsDataset, SimpleTsDataset, HydroDataset
from loader.jena_airpress import get_jena_data, JENA_TYPE_PREFIX
from loader.sapflux import SAPFLUX_TYPE_PREFIX, get_sapflux_data
from loader.toy_loader import TOY_DATA_PREFIX, get_toy_data
from loader.hydrology import HYDRO_TYPE_PREFIX, get_hydro_data

NSDB_DATASET = [
    "nsdb-60m-2020",
    "nsdb-60m-2019",
    "nsdb-60m-2018-20",
    "nsdb-30m-2020",
    "nsdb-30m-2019",
    "nsdb-30m-2018-20",
    "nsdb-60m-2019-wCoord",
    "nsdb-60m-2018-20-wCoord",
]
AIR_QUALITY = ["air-25", "air-10", "air-25-half", "air-10-half"]
STOCK = ["stock"]
PEMS = ["pems"]
RAIN = ["rain"]


def get_ts_id(dataset_type, data_path):
    if dataset_type is None:
        return str(dataset_type)
    else:
        return str(dataset_type) + data_path.split("/")[-1].split(".")[0]


class DataGenerator:

    @staticmethod
    def get_data(
        data_config: TSDataConfig,
        task_config: TaskConfig,
        replace_base_dir,
        X_norm_param=None,
        Y_norm_param=None,
        hydro_static_norm_param=None,
    ) -> [ChronoSplittedTsDataset]:
        datasets = []
        if data_config.dataset_type.startswith(TOY_DATA_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_toy_data(
                data_config.dataset_type, p
            )  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(
                    DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config)
                )
        elif data_config.dataset_type.startswith(JENA_TYPE_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_jena_data(
                data_config.dataset_type, p
            )  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(
                    DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config)
                )
        elif data_config.dataset_type.startswith(SAPFLUX_TYPE_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            table_combos = get_sapflux_data(
                data_config.dataset_type, p
            )  # List of X, Y, id Tuples
            for X_full, Y_full, ts_id in table_combos:
                datasets.append(
                    DataGenerator._table_to_dataset(X_full, Y_full, ts_id, task_config)
                )
        elif data_config.dataset_type.startswith(HYDRO_TYPE_PREFIX):
            assert len(data_config.paths) == 1
            path = data_config.paths[0]
            path = path.replace("/some_base_dir/data", replace_base_dir)
            p = Path(path)
            # Returns a list of (X, Y, id, number of the FC model's train samples) Tuples and indices of the
            # static attributes.
            table_combos, attribute_indices, attribute_norm_params = get_hydro_data(
                data_config.dataset_type,
                p,
                hydro_static_norm_param=hydro_static_norm_param,
                **data_config.add_config,
            )
            for X_full, Y_full, ts_id, train_steps, calib_steps in table_combos:
                # Add the number of training steps to the task config so the dataset generator can do its split
                # consistent with the train/test split in the pretrained neuralhydrology model.
                task_config.add_config = (
                    dict(task_config.add_config) if task_config.add_config else {}
                ) | {"train_steps": int(train_steps)}
                if calib_steps is not None:
                    task_config.add_config = dict(task_config.add_config) | {
                        "calib_steps": int(calib_steps)
                    }
                # Pass the static attribute indices to the hydro dataset so it knows which variables should be ignored
                # during normalization.
                dataset_init = lambda **kwargs: HydroDataset(
                    **(
                        kwargs
                        | {
                            "static_attribute_indices": attribute_indices,
                            "static_attribute_norm_param": attribute_norm_params,
                        }
                    )
                )
                datasets.append(
                    DataGenerator._table_to_dataset(
                        X_full, Y_full, ts_id, task_config, dataset_init
                    )
                )
        else:
            dataset_type = data_config.dataset_type
            is_small = dataset_type.startswith("small_")
            dataset_type = (
                dataset_type.replace("small_", "") if is_small else dataset_type
            )
            for path in data_config.paths:
                path = path.replace("/some_base_dir/data", replace_base_dir)
                p = Path(path)
                if p.is_dir():
                    datasets.extend(
                        [
                            DataGenerator._get_data_single(
                                dataset_type,
                                str(sub_path.resolve()),
                                task_config,
                                is_small=is_small,
                            )
                            for sub_path in p.glob("*.csv")
                        ]
                    )
                else:
                    datasets.append(
                        DataGenerator._get_data_single(
                            dataset_type, path, task_config, is_small
                        )
                    )
        if hasattr(task_config, "global_norm") and task_config.global_norm:
            DataGenerator._global_normalize(
                datasets, X_norm_param=X_norm_param, Y_norm_param=Y_norm_param
            )
        return datasets

    @staticmethod
    def _get_data_single(
        dataset_type, data_path, task_config: TaskConfig, is_small: bool = False
    ) -> ChronoSplittedTsDataset:
        if dataset_type in EnbPI_DATASETS:
            X_full, Y_full = demo_code_data_load(dataset_type, data_path)
        elif dataset_type in NSDB_DATASET:
            X_full, Y_full = load_nsdb_data(dataset_type, data_path)
        elif dataset_type in AIR_QUALITY:
            X_full, Y_full = load_bejing_air_data(dataset_type, data_path)
        elif dataset_type in STOCK:
            X_full, Y_full = load_stock_data(dataset_type, data_path)
        elif dataset_type in PEMS:
            X_full, Y_full = load_pems_data(dataset_type, data_path)
        elif dataset_type in RAIN:
            X_full, Y_full = load_rain_data(dataset_type, data_path)
        else:
            raise ValueError(f"Dataset {dataset_type.dataset_type} not supported!")
        if is_small:
            X_full = X_full[:5000]
            Y_full = Y_full[:5000]
        return DataGenerator._table_to_dataset(
            X_full, Y_full, get_ts_id(dataset_type, data_path), task_config
        )

    @staticmethod
    def _global_normalize(datasets, X_norm_param=None, Y_norm_param=None):
        if X_norm_param is None:
            X_train_calib_all = torch.concat(
                ([d.X_train for d in datasets] + [d.X_calib for d in datasets])
            )
            X_mean = torch.mean(X_train_calib_all, dim=0)
            X_std = torch.std(X_train_calib_all, dim=0)
            del X_train_calib_all
        else:
            X_mean, X_std = X_norm_param
        if Y_norm_param is None:
            Y_train_calib_all = torch.concat(
                ([d.Y_train for d in datasets] + [d.Y_calib for d in datasets])
            )
            Y_mean = torch.mean(Y_train_calib_all, dim=0)
            Y_std = torch.std(Y_train_calib_all, dim=0)
            del Y_train_calib_all
        else:
            Y_mean, Y_std = Y_norm_param
        for data in datasets:
            data.global_normalize(
                X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std
            )

    @staticmethod
    def _table_to_dataset(
        X_full,
        Y_full,
        ts_id,
        task_config: TaskConfig,
        dataset_init=ChronoSplittedTsDataset,
    ):
        # If the absolute number of train_steps is known, ignore the split config and do a 50/50 split for cal/test.
        if task_config.add_config is not None and task_config.add_config.get(
            "train_steps"
        ):
            train_steps = task_config.add_config["train_steps"]
            if task_config.add_config.get("no_calib", False):
                cal_steps = 0
            elif "calib_steps" in task_config.add_config:
                cal_steps = task_config.add_config["calib_steps"]
            else:
                cal_steps = (X_full.shape[0] - train_steps) // 2
            split_points = np.array([train_steps, train_steps + cal_steps])
        else:
            split_def = DataGenerator._get_needed_splits(task_config)
            split_points, _ = DataGenerator._chronological_split(
                X_full.shape[0], split_def
            )
        return DataGenerator._create_dataset(
            ts_id,
            X_full,
            Y_full,
            split_points,
            norm=(not hasattr(task_config, "global_norm"))
            or (not task_config.global_norm),
            dataset_init=dataset_init,
        )

    @staticmethod
    def _chronological_split(full_len, split_def):
        split_points = (
            (np.array(list(split_def.values())) / sum(split_def.values())) * full_len
        ).astype(int)[:-1]
        split_points = np.cumsum(split_points)
        return split_points, split_def

    @staticmethod
    def _create_dataset(
        ts_id, X_full, Y_full, split_points, norm, dataset_init
    ) -> ChronoSplittedTsDataset:
        if len(split_points) > 1:
            assert len(split_points) == 2
            return dataset_init(
                ts_id=ts_id,
                X=X_full,
                Y=Y_full,
                test_step=split_points[1].item(),
                calib_step=split_points[0].item(),
                normalize=norm,
            )
        else:
            return dataset_init(
                ts_id=ts_id,
                X=X_full,
                Y=Y_full,
                test_step=split_points[0].item(),
                normalize=norm,
            )

    @staticmethod
    def _get_needed_splits(task_config: TaskConfig) -> Dict[str, float]:
        if len(task_config.data_splits) < 3:
            raise ValueError("3 splits needed!")
        if len(task_config.data_splits) > 3:
            raise ValueError("More than 3 splits!")
        _sum = sum(task_config.data_splits)
        splits = OrderedDict()
        splits["train"] = task_config.data_splits[0] / _sum
        splits["calib"] = task_config.data_splits[1] / _sum
        splits["test"] = task_config.data_splits[2] / _sum
        return splits

    #
    # Legacy Methods for original EnbPI Implementation
    #

    @staticmethod
    def get_data_legacy(
        data_config: TSDataConfig, task_config: TaskConfig
    ) -> List[Tuple[dict, dict]]:
        data = []
        for path in data_config.paths:
            if data_config.dataset_type in EnbPI_DATASETS:
                X_full, Y_full = demo_code_data_load(data_config.dataset_type, path)
                splits = DataGenerator._get_needed_splits(task_config)
                # In Legacy there is no calib
                legacy_splits = OrderedDict()
                legacy_splits["train"] = splits["train"] + splits["calib"]
                legacy_splits["test"] = splits["test"]
                split_points, split_def = DataGenerator._chronological_split(
                    X_full.shape[0], legacy_splits
                )
                data.append(
                    (
                        DataGenerator._create_dataset_legacy(
                            X_full, Y_full, split_points, split_def
                        ),
                        dict(
                            ts_id=get_ts_id(
                                dataset_type=data_config.dataset_type, data_path=path
                            )
                        ),
                    )
                )
            else:
                raise ValueError(f"Dataset {data_config.dataset_type} not supported!")
        return data

    @staticmethod
    def _create_dataset_legacy(X_full, Y_full, split_points, split_def):
        splits = dict()
        for idx, split_name in enumerate(split_def.keys()):
            start_idx = 0 if idx == 0 else split_points[idx - 1]
            if idx == len(split_points):  # Last Split
                splits[split_name] = SimpleTsDataset(
                    X_full[start_idx:], Y_full[start_idx:]
                )
            else:
                end_idx = split_points[idx]
                splits[split_name] = SimpleTsDataset(
                    X_full[start_idx:end_idx], Y_full[start_idx:end_idx]
                )
        return splits


def load_nsdb_data(dataset_type, data_path):
    data = pd.read_csv(data_path)
    if dataset_type.startswith("nsdb-60m"):
        drop_every_n = 2
    elif dataset_type.startswith("nsdb-30m"):
        drop_every_n = 1
    else:
        raise ValueError("Type not supported")
    append_coords = dataset_type.endswith("-wCoord")
    data = data.iloc[::drop_every_n, :]
    Y_full = data["dhi"]
    X_full = data.loc[:, data.columns != "dhi"]
    X_full.drop(columns=X_full.columns[0:6], inplace=True)  # Drop Date Stuff
    if append_coords:
        p = Path(data_path)
        coordinates = pd.read_csv(p.parent.parent / "solar_coordinates.csv")
        lat = coordinates[coordinates["location"] == p.stem]["latitude"].item()
        long = coordinates[coordinates["location"] == p.stem]["longitude"].item()
        X_full["latitude"] = lat
        X_full["longitude"] = long
    return (
        torch.from_numpy(X_full.to_numpy()).float(),
        torch.from_numpy(Y_full.to_numpy()).float(),
    )


def load_stock_data(dataset_type, data_path):
    data = pd.read_csv(data_path)
    data = data.dropna()
    Y_full = data["Close"]
    X_full = data.loc[:, data.columns != "Close"]
    return (
        torch.from_numpy(X_full.to_numpy()).float(),
        torch.from_numpy(Y_full.to_numpy()).float(),
    )


def load_pems_data(dataset_type, data_path):
    data = np.load(data_path)
    # Choose inspected data: 0 -- total flow prediction, 1 -- avg speed prediction
    data = data[..., 0]

    target_sensor = 0

    Y_full = data[:, target_sensor]
    X_full = np.delete(data, target_sensor, -1)
    X_full = X_full[:, :200]

    return (
        torch.from_numpy(X_full).float(),
        torch.from_numpy(Y_full).float(),
    )


def load_rain_data(dataset_type, data_path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = pd.read_csv(data_path)
    data = data.drop(columns=["ymd"])
    Y_full = data["precip_hrly_avg"]
    columns = list(data.columns)
    columns.remove("precip_hrly_min")
    columns.remove("precip_hrly_max")
    columns.remove("precip_hrly_avg")
    X_full = data[columns]

    return (
        torch.from_numpy(X_full.values).float(),
        torch.from_numpy(Y_full.values).float(),
    )


def load_bejing_air_data(dataset_type, data_path, wd_encode="encode", imputer=None):
    data = pd.read_csv(data_path)
    if "Unnamed: 0" in data:
        data.drop(columns=["Unnamed: 0"], inplace=True, axis=1)
    data.drop(columns=["No", "station"], inplace=True, axis=1)
    data.drop(columns=["year", "month", "day", "hour"], inplace=True, axis=1)
    data_wd = data["wd"].fillna(value="Unknown")
    if imputer is not None:
        data = pd.DataFrame(imputer.fit_transform(data.loc[:, data.columns != "wd"]))
    else:
        data = data.loc[:, data.columns != "wd"].fillna(method="ffill")
        data = data.fillna(method="bfill")
    data["wd"] = data_wd
    if dataset_type.startswith("air-25"):
        Y_full = data["PM2.5"]
    elif dataset_type.startswith("air-10"):
        Y_full = data["PM10"]
    data.drop(columns=["PM2.5", "PM10"], inplace=True, axis=1)
    if wd_encode == "drop":
        data.drop(columns=["wd"], inplace=True, axis=1)
    elif wd_encode == "one-hot":
        data = pd.get_dummies(data)
    elif wd_encode == "encode":
        data["wd_h"] = data["wd"].apply(lambda x: _encode_direction(x, True))
        data["wd_v"] = data["wd"].apply(lambda x: _encode_direction(x, False))
        data.drop(columns=["wd"], inplace=True, axis=1)
    X_full = data
    return (
        torch.from_numpy(X_full.to_numpy()).float(),
        torch.from_numpy(Y_full.to_numpy()).float(),
    )


def _encode_direction(direction, horizontal):
    if horizontal:
        if direction in ["N", "S"]:
            return 0
        elif direction in ["NNW", "SSW"]:
            return -0.5
        elif direction in ["NW", "SW"]:
            return -0.7
        elif direction in ["WNW", "WSW"]:
            return -0.86
        elif direction == "W":
            return -1
        elif direction in ["NNE", "SSE"]:
            return 0.5
        elif direction in ["NE", "SE"]:
            return 0.7
        elif direction in ["ENE", "ESE"]:
            return 0.86
        elif direction == "E":
            return 1
        elif direction == "Unknown":
            return 0
        else:
            raise ValueError("Invalid Dir")
    else:
        if direction in ["W", "E"]:
            return 0
        elif direction in ["WSW", "ESE"]:
            return -0.5
        elif direction in ["SW", "SE"]:
            return -0.7
        elif direction in ["SSW", "SSE"]:
            return -0.86
        elif direction == "S":
            return -1
        elif direction in ["WNW", "ENE"]:
            return 0.5
        elif direction in ["NW", "NE"]:
            return 0.7
        elif direction in ["NNW", "NNE"]:
            return 0.86
        elif direction == "N":
            return 1
        elif direction == "Unknown":
            return 0
        else:
            raise ValueError("Invalid Dir")

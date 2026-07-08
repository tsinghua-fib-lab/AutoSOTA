import pandas as pd
import numpy as np
import torch

SAPFLUX_TYPE_PREFIX = "sapflux"


def _additional_encoding(df, time_col=0):
    # DateTimeEnc
    return df


def get_prep_type(dataset_type):
    if dataset_type.startswith(f'{SAPFLUX_TYPE_PREFIX}-solo1-'):
        return "solo_1"
    elif dataset_type.startswith(f'{SAPFLUX_TYPE_PREFIX}-solo2-'):
        return "solo_2"
    elif dataset_type.startswith(f'{SAPFLUX_TYPE_PREFIX}-solo3-'):
        return "solo_3"
    else:
        raise ValueError("Not available!")


def get_size_range(dataset_type):
    if dataset_type.endswith(f'-large'):
        return 15000, 20000
    elif dataset_type.endswith(f'-small'):
        return 7000, 9000
    else:
        raise ValueError("Not available!")


def get_sapflux_data(dataset_type, path):
    assert path.is_dir()
    path = path / get_prep_type(dataset_type)
    assert path.is_dir()
    datasets = []
    min_size, max_size = get_size_range(dataset_type)
    for sub_path in path.glob("*.csv"):
        _id = sub_path.stem
        df = pd.read_csv(sub_path, parse_dates=["solar_TIMESTAMP"])
        if len(df) < min_size or len(df) > max_size:
            continue
        df = _additional_encoding(df)
        Y_full = df.iloc[:, 1]
        X_full = df.iloc[:, 2:]
        datasets.append((
            torch.from_numpy(X_full.to_numpy()).float(),
            torch.from_numpy(Y_full.to_numpy()).float(),
            _id
        ))
    return datasets


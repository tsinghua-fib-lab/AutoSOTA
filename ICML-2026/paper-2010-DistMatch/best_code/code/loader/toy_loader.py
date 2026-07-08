import numpy as np
import pandas as pd
import torch

TOY_DATA_PREFIX = "toy"


def get_toy_data(dataset_type, path):
    assert path.is_dir()
    datasets = []
    for sub_path in path.glob("*.csv"):
        df = pd.read_csv(sub_path)
        Y_full = df.iloc[:, 1]
        X_full = df.iloc[:, 2:]
        datasets.append((
            torch.from_numpy(X_full.to_numpy()).float(),
            torch.from_numpy(Y_full.to_numpy()).float(),
            sub_path.name[:-4]
        ))
    return datasets

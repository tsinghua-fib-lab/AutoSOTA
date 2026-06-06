# The semi-synthetic datasets based on Lalonde, created by Neal et al. [1].
#
# Sources:
# [1] Neal, Brady, Chin-Wei Huang, and Sunand Raghupathi. "Realcause: Realistic causal inference benchmarking."
# arXiv preprint arXiv:2011.15007 (2020).
# [2] https://github.com/bradyneal/realcause/tree/master/realcause_datasets

import os
from abc import ABC
from typing import Callable, Tuple

import numpy as np
import pandas as pd

from .base import ATE_Dataset, CATE_Dataset, EvalDatasetCatalog

script_dir = os.path.dirname(os.path.abspath(__file__))


class RealCauseDataset(EvalDatasetCatalog, ABC):
    def __init__(
        self,
        name: str,
        csv_path_fn: Callable[[int], str],  # Returns the URL of the CSV file for the dataset, given an index
        n_tables: int = 100,
        test_ratio: float = 0.1,
        seed: int = 42,
        **kwargs,
    ):
        self.csv_path_fn = csv_path_fn
        self.test_ratio = test_ratio
        self.rngs = [np.random.default_rng(seed + i) for i in range(n_tables)]
        self.datasets = [self._get_data(i) for i in range(n_tables)]
        super().__init__(n_tables, name=name)

    def _get_data(self, idx: int) -> Tuple[CATE_Dataset, ATE_Dataset]:
        csv_path = self.csv_path_fn(idx)
        # Read the dataset
        data = pd.read_csv(csv_path)

        covariates_size = data.shape[1] - 5  # everything except for `t`, `y`, `y0`, `y1`, and `ite`
        # Define column names
        col_names = [f"x{i}" for i in range(1, covariates_size + 1)] + ["t", "y", "y0", "y1", "ite"]
        data.columns = col_names
        data = data.astype({"t": "bool"}, copy=False)

        # Convert to PyTorch tensors
        covariates = data.iloc[:, :covariates_size].values.astype(np.float32)  # Features
        treatments = data["t"].values.astype(np.float32)  # Treatment
        outcomes = data["y"].values.astype(np.float32)  # Factual outcomes

        cate = data["ite"].values.astype(np.float32)
        ate = cate.mean()

        # Split the dataset into train and test sets
        indices = self.rngs[idx].permutation(covariates.shape[0])
        split_idx = int(len(indices) * (1 - self.test_ratio))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        cate_dataset = CATE_Dataset(
            X_train=covariates[train_indices],
            t_train=treatments[train_indices],
            y_train=outcomes[train_indices],
            X_test=covariates[test_indices],
            true_cate=cate[test_indices],
        )
        ate_dataset = ATE_Dataset(
            X=covariates,
            t=treatments,
            y=outcomes,
            true_ate=float(ate),
        )
        return cate_dataset, ate_dataset

    def __getitem__(self, index) -> Tuple[CATE_Dataset, ATE_Dataset]:
        return self.datasets[index]


class RealCauseLalondePSIDDataset(RealCauseDataset):

    def __init__(self, **kwargs):
        LALONDE_PSID_CSV_PATH = lambda i: os.path.join(script_dir, f"realcause_datasets/lalonde_psid_sample{i}.csv")
        super().__init__(name="LalondePSID", csv_path_fn=LALONDE_PSID_CSV_PATH, **kwargs)


class RealCauseLalondeCPSDataset(RealCauseDataset):

    def __init__(self, **kwargs):
        LALONDE_CPS_CSV_PATH = lambda i: os.path.join(script_dir, f"realcause_datasets/lalonde_cps_sample{i}.csv")
        super().__init__(name="LalondeCPS", csv_path_fn=LALONDE_CPS_CSV_PATH, **kwargs)

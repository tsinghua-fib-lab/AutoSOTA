# The ACIC 2016 challenge dataset
#
# Sources:
# [1] Dorie, Vincent, et al. "Automated versus do-it-yourself methods for causal inference: Lessons learned
# from a data analysis competition." (2019): 43-68.
# [2] https://github.com/BiomedSciAI/causallib/tree/master/causallib/datasets/data/acic_challenge_2016
#
# The challenge includes 10 different datasets.

from typing import Tuple

import numpy as np
import pandas as pd

from .base import ATE_Dataset, CATE_Dataset, EvalDatasetCatalog

X_CSV_URL = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv"
)

ZY_CSV_URL = (
    lambda i: f"https://raw.githubusercontent.com/BiomedSciAI/causallib/master/causallib/datasets/data/acic_challenge_2016/zymu_{i}.csv"
)


class ACIC2016Dataset(EvalDatasetCatalog):
    def __init__(self, test_ratio: float = 0.1, seed: int = 42, n_tables: int = 10):
        super().__init__(n_tables, name="ACIC2016")
        self.test_ratio = test_ratio
        self.x_data = pd.read_csv(X_CSV_URL)
        self.rngs = [np.random.default_rng(seed + i) for i in range(n_tables)]
        self.datasets = [self._get_data(i) for i in range(n_tables)]

    def _get_data(self, idx: int) -> Tuple[CATE_Dataset, ATE_Dataset]:
        """Loads and processes a single dataset split."""
        # Download file URLs
        simulation_url = ZY_CSV_URL(idx + 1)

        sim_data = pd.read_csv(simulation_url)

        # Define column names for x.csv and simulation data
        self.x_data.columns = [f"x_{i+1}" for i in range(self.x_data.shape[1])]
        sim_data.columns = ["z", "y0", "y1", "mu0", "mu1"]

        # Handle categorical variables
        categorical_columns = ["x_2", "x_21", "x_24"]
        numerical_columns = [f"x_{i+1}" for i in range(self.x_data.shape[1]) if f"x_{i+1}" not in categorical_columns]
        self.x_data["x_2_numeric"] = self.x_data["x_2"].astype("category").cat.codes
        self.x_data["x_21_numeric"] = self.x_data["x_21"].astype("category").cat.codes
        self.x_data["x_24_numeric"] = self.x_data["x_24"].astype("category").cat.codes
        numerical_columns = numerical_columns + ["x_2_numeric", "x_21_numeric", "x_24_numeric"]
        self.x_data = self.x_data.loc[:, numerical_columns]

        # Convert to tensors
        covariates = self.x_data.values.astype(np.float32)  # Covariates with encoded categorical variables
        treatments = sim_data["z"].values.astype(np.float32)  # Treatment

        y1 = sim_data["y1"].values.astype(np.float32)  # Potential outcomes under treatment
        y0 = sim_data["y0"].values.astype(np.float32)  # Potential outcomes under control
        outcomes = np.where(treatments == 1, y1, y0)

        mu0 = sim_data["mu0"].values.astype(np.float32)
        mu1 = sim_data["mu1"].values.astype(np.float32)
        cate = mu1 - mu0
        ate = cate.mean().item()

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
            true_ate=ate,
        )

        return cate_dataset, ate_dataset

    def __getitem__(self, index) -> Tuple[CATE_Dataset, ATE_Dataset]:
        return self.datasets[index]

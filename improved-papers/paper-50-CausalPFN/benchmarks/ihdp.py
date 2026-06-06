# The Infant Health and Development Program (IHDP) dataset
#
# Sources:
# [1] Hill, Jennifer L. Bayesian nonparametric modeling for causal inference. Journal of Computational
# and Graphical Statistics, 20(1), 2011.
# [2] Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect:
# generalization bounds and algorithms." International conference on machine learning. PMLR, 2017.
# [3] https://www.fredjo.com/
#
# This version of the dataset includes 100 different realisations.

import os
from typing import Tuple

import numpy as np

from .base import ATE_Dataset, CATE_Dataset, EvalDatasetCatalog

script_dir = os.path.dirname(os.path.abspath(__file__))


class IHDPDataset(EvalDatasetCatalog):
    def __init__(
        self,
        n_tables: int = 100,
        seed: int = 42,
    ):
        n_tables = min(n_tables, 100)
        self.rngs = [np.random.default_rng(seed + i) for i in range(n_tables)]
        self.train_data = np.load(os.path.join(script_dir, "IHDP/ihdp_npci_1-100.train.npz"))
        self.test_data = np.load(os.path.join(script_dir, "IHDP/ihdp_npci_1-100.test.npz"))
        super().__init__(n_tables, name="IHDP")

    def __getitem__(self, idx: int) -> Tuple[CATE_Dataset, ATE_Dataset]:
        train_covariates = self.train_data["x"][..., idx].astype(np.float32)  # Covariates
        train_treatments = self.train_data["t"][..., idx].astype(np.float32)  # Treatment
        train_outcomes = self.train_data["yf"][..., idx].astype(np.float32)  # Outcomes

        test_covariates = self.test_data["x"][..., idx].astype(np.float32)  # Covariates
        test_treatments = self.test_data["t"][..., idx].astype(np.float32)  # Treatment
        test_outcomes = self.test_data["yf"][..., idx].astype(np.float32)
        test_mu1 = self.test_data["mu1"][..., idx].astype(np.float32)
        test_mu0 = self.test_data["mu0"][..., idx].astype(np.float32)

        test_cate = test_mu1 - test_mu0
        ate = float(self.train_data["ate"].item())

        # combine test and train and permute for ATE
        all_covariates = np.concatenate([train_covariates, test_covariates], axis=0)
        all_treatments = np.concatenate([train_treatments, test_treatments], axis=0)
        all_outcomes = np.concatenate([train_outcomes, test_outcomes], axis=0)
        indices = self.rngs[idx].permutation(all_covariates.shape[0])

        cate_dataset = CATE_Dataset(
            X_train=train_covariates,
            t_train=train_treatments,
            y_train=train_outcomes,
            X_test=test_covariates,
            true_cate=test_cate,
        )

        ate_dataset = ATE_Dataset(
            X=all_covariates[indices],
            t=all_treatments[indices],
            y=all_outcomes[indices],
            true_ate=ate,
        )

        return cate_dataset, ate_dataset

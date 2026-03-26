from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


@dataclass
class CATE_Dataset:  # conditional average treatment effect
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    true_cate: np.ndarray
    propensities_train: np.ndarray | None = None
    E_y0_train: np.ndarray | None = None
    E_y1_train: np.ndarray | None = None


@dataclass
class ATE_Dataset:  # average treatment effect
    X: np.ndarray
    t: np.ndarray
    y: np.ndarray
    true_ate: float


@dataclass
class Qini_Dataset:
    X_train: np.ndarray
    t_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    t_test: np.ndarray
    y_test: np.ndarray


class EvalDatasetCatalog(ABC):
    """
    The dataset catalog is a collection of datasets used for evaluating the model.
    """

    def __init__(self, n_tables: int, name: str):
        self.n_tables = n_tables
        self.name = name

    def __len__(self):
        return self.n_tables

    def __str__(self):
        return self.name

    @abstractmethod
    def __getitem__(self, index) -> Any:
        raise NotImplementedError("This method should be implemented by the subclass")


class RCTUpliftDatasetCatalog(EvalDatasetCatalog):
    """
    A base class for RCT uplift datasets.
    Here, we get all of the unit covariates, treatments, and outcomes, alongside the number of folds
    for honest spliiting evaluation. The class also optionally takes a subsample_max_rows parameter
    which when specified, does a stratified sub-sampling of the dataset to reduce the size of the dataset.
    This is done to speed up the evaluation process sometimes; however, it is generally not recommended.
    """

    def __init__(
        self,
        name: str,
        n_folds: int,
        subsample_max_rows: int | None,
        all_X: np.ndarray,
        all_y: np.ndarray,
        all_t: np.ndarray,
        base_rng: np.random.Generator,
    ):
        """
        Args:
            n_folds (int):
                The number of folds for honest splitting. This will result in each
                index holding one fold of the dataset.
            test_n_rows (int): The number of rows in the test set.
            train_n_rows (int): The number of rows in the train set.
            all_X (np.ndarray): All of the covariates for the dataset.
            all_y (np.ndarray): All of the outcomes for the dataset.
            all_t (np.ndarray): All of the treatments of the dataset.
            base_rng (np.random.Generator): A random number generator for reproducibility.
        """
        super().__init__(n_tables=n_folds, name=name)

        self.all_X = all_X
        self.all_y = all_y
        self.all_t = all_t

        # strata takes four values:
        # 0: control and low outcome
        # 1: control and high outcome
        # 2: treatment and low outcome
        # 3: treatment and high outcome
        if len(np.unique(self.all_y)) == 2:
            strata = np.where(self.all_y == np.min(self.all_y), 0, 1) * 2 + self.all_t
        else:
            strata = self.all_t

        if subsample_max_rows is not None and subsample_max_rows < len(self.all_X):
            fraction = subsample_max_rows / strata.shape[0]
            sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=base_rng.integers(0, 2**32 - 1))
            # Stratified sub-sampling
            idx, _ = next(sss.split(self.all_X, strata))
            self.all_X = self.all_X[idx]
            self.all_y = self.all_y[idx]
            self.all_t = self.all_t[idx]
            strata = strata[idx]

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_rng.integers(0, 2**32 - 1))

        self.all_fold_train_idx = []
        self.all_fold_test_idx = []
        for train_idx, test_idx in cv.split(self.all_X, strata):
            self.all_fold_train_idx.append(train_idx)
            self.all_fold_test_idx.append(test_idx)

        self.rng_seeds = []
        for _ in range(n_folds):
            self.rng_seeds.append(base_rng.integers(0, 2**32 - 1))

    def __getitem__(self, index) -> Qini_Dataset:
        train_idx = self.all_fold_train_idx[index]
        test_idx = self.all_fold_test_idx[index]

        X_train = self.all_X[train_idx]
        t_train = self.all_t[train_idx]
        y_train = self.all_y[train_idx]
        X_test = self.all_X[test_idx]
        t_test = self.all_t[test_idx]
        y_test = self.all_y[test_idx]

        return Qini_Dataset(
            X_train=X_train,
            t_train=t_train,
            y_train=y_train,
            X_test=X_test,
            t_test=t_test,
            y_test=y_test,
        )

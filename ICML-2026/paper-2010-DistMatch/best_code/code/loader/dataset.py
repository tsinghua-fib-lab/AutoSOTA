from abc import abstractmethod, ABC
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import stats
import warnings


class TsDataset(ABC):
    def _normalize(self, data_full):
        mean = torch.mean(data_full[: self.test_step], dim=0)
        std = torch.std(data_full[: self.test_step], dim=0)
        std = std.where(std != 0, torch.ones_like(std))
        return (data_full - mean) / std, mean, std

    @property
    @abstractmethod
    def X_normalize_props(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def Y_normalize_props(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def ts_id(self):
        pass

    @property
    @abstractmethod
    def X_full(self):
        pass

    @property
    @abstractmethod
    def Y_full(self):
        pass

    @property
    @abstractmethod
    def X_train(self):
        pass

    @property
    @abstractmethod
    def Y_train(self):
        pass

    @property
    @abstractmethod
    def X_calib(self):
        pass

    @property
    @abstractmethod
    def Y_calib(self):
        pass

    @property
    @abstractmethod
    def X_test(self):
        pass

    @property
    @abstractmethod
    def Y_test(self):
        pass

    @property
    @abstractmethod
    def no_test_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def no_of_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def first_prediction_step(self) -> int:
        pass

    @property
    @abstractmethod
    def no_calib_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def has_calib_set(self):
        pass

    @property
    @abstractmethod
    def calib_step(self):
        pass

    @property
    @abstractmethod
    def test_step(self):
        pass

    @property
    @abstractmethod
    def no_x_features(self):
        pass


class ChronoSplittedTsDataset(TsDataset):
    def __init__(
        self,
        ts_id: str,
        X,
        Y,
        test_step: int,
        calib_step: Optional[int] = None,
        normalize=True,
    ):
        super().__init__()
        self._ts_id = ts_id
        self._X = X
        self._Y = Y.view(Y.shape[0], -1)

        # LAG
        lag = 3
        self._X = self._X[:-lag]
        self._Y = self._Y[lag:]

        self._test_step = test_step
        self._calib_step = calib_step
        if calib_step is not None and test_step <= calib_step:
            raise ValueError("Calibration must be before training!")
        if test_step is None or test_step >= X.shape[0]:
            raise ValueError("Test Step must be defnied smaller than actual datset!")
        if normalize:
            self._X, self._X_means, self._X_stds = self._normalize(self._X)
            self._Y, self._Y_means, self._Y_stds = self._normalize(self._Y)
        else:
            self._X_means, self._X_stds = 0, 1.0
            self._Y_means, self._Y_stds = 0, 1.0

    def global_normalize(self, X_mean, X_std, Y_mean, Y_std):
        self._X = (self._X - X_mean) / X_std
        self._Y = (self._Y - Y_mean) / Y_std
        self._X_means, self._X_stds = X_mean, X_std
        self._Y_means, self._Y_stds = Y_mean, Y_std

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def Y_normalize_props(self) -> Tuple[float, float]:
        return self._Y_means, self._Y_stds

    @property
    def Y_std(self):
        return torch.std(self.Y_train)

    @property
    def ts_id(self):
        return self._ts_id

    @property
    def X_full(self):
        return self._X

    @property
    def Y_full(self):
        return self._Y

    @property
    def X_train(self):
        return self._X[: self.calib_step if self.has_calib_set else self.test_step]

    @property
    def Y_train(self):
        return self._Y[: self.calib_step if self.has_calib_set else self.test_step]

    @property
    def has_calib_set(self):
        return self.calib_step is not None

    @property
    def X_calib(self):
        if not self.has_calib_set:
            raise ValueError("No calibration set in this data!")
        return self._X[self.calib_step : self.test_step]

    @property
    def Y_calib(self):
        if not self.has_calib_set:
            raise ValueError("No calibration set in this data!")
        return self._Y[self.calib_step : self.test_step]

    @property
    def X_test(self):
        return self._X[self.test_step :]

    @property
    def Y_test(self):
        return self._Y[self.test_step :]

    @property
    def no_train_steps(self) -> int:
        return self.Y_train.shape[0]

    @property
    def no_calib_steps(self) -> int:
        return self.Y_calib.shape[0] if self.has_calib_set else 0

    @property
    def no_test_steps(self) -> int:
        return self.Y_test.shape[0]

    @property
    def no_of_steps(self) -> int:
        return self.Y_full.shape[0]

    @property
    def first_prediction_step(self) -> int:
        return self.calib_step if self.has_calib_set else self.test_step

    @property
    def calib_step(self):
        return self._calib_step

    @property
    def test_step(self):
        return self._test_step

    @property
    def no_x_features(self):
        return self.X_full.shape[1]


class BoostrapEnsembleTsDataset(TsDataset):
    """
    Implemented for the boostrap version of Xu et al. 2022 where train and calib data is shared
    by using an ensemble and the "left out" points as calibration points
    """

    def __init__(self, ts_id: str, X, Y, test_step: int, normalize=True):
        super().__init__()
        self._ts_id = ts_id
        self._X = X
        self._Y = Y.view(Y.shape[0], -1)
        self._test_step = test_step
        if test_step is None or test_step >= X.shape[0]:
            raise ValueError("Test Step must be defnied smaller than actual dataset!")
        if normalize:
            self._X, self._X_means, self._X_stds = self._normalize(self._X)
            self._Y, self._Y_means, self._Y_stds = self._normalize(self._Y)
        else:
            self._X_means, self._X_stds = 0, 1.0
            self._Y_means, self._Y_stds = 0, 1.0

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def Y_normalize_props(self) -> Tuple[float, float]:
        return self._Y_means, self._Y_stds

    @property
    def ts_id(self):
        return self._ts_id

    @property
    def X_full(self):
        return self._X

    @property
    def Y_full(self):
        return self._Y

    @property
    def X_train(self):
        return self._X[: self.test_step]

    @property
    def Y_train(self):
        return self._Y[: self.test_step]

    @property
    def X_calib(self):
        return self.X_train

    @property
    def Y_calib(self):
        return self.Y_train

    @property
    def X_test(self):
        return self._X[self.test_step :]

    @property
    def Y_test(self):
        return self._Y[self.test_step :]

    @property
    def no_test_steps(self) -> int:
        return self.Y_test.shape[0]

    @property
    def no_of_steps(self) -> int:
        return self.Y_full.shape[0]

    @property
    def first_prediction_step(self) -> int:
        return 0

    @property
    def no_calib_steps(self) -> int:
        return self.Y_calib.shape[0]

    @property
    def has_calib_set(self):
        return True

    @property
    def calib_step(self):
        return 0

    @property
    def test_step(self):
        return self._test_step

    @property
    def no_x_features(self):
        return self.X_full.shape[1]


class SimpleTsDataset:
    """
    Deprecated
    """

    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y


class HydroDataset(ChronoSplittedTsDataset):
    """Dataset for the hydrology application."""

    def __init__(
        self,
        ts_id: str,
        X,
        Y,
        test_step: int,
        static_attribute_indices: list[int],
        static_attribute_norm_param,
        calib_step: Optional[int] = None,
        normalize=True,
    ):

        self._static_attribute_indices = static_attribute_indices
        self._static_attribute_norm_param = static_attribute_norm_param
        # If there are no static attributes, we can use the default normalization from super.
        default_normalization = normalize and not static_attribute_indices
        super().__init__(
            ts_id, X, Y, test_step, calib_step, normalize=default_normalization
        )

        if normalize and static_attribute_indices:
            # Possible but not needed.
            raise NotImplementedError()

    @property
    def X_normalize_props(self) -> Tuple[float, float]:
        return self._X_means, self._X_stds

    @property
    def static_normalize_props(self):
        return self._static_attribute_norm_param

    def global_normalize(self, X_mean, X_std, Y_mean, Y_std):
        self._Y = (self._Y - Y_mean) / Y_std
        self._Y_means, self._Y_stds = Y_mean, Y_std

        # Statics are normalized in data loading
        dynamic_indices = [
            i
            for i in range(self._X.shape[1])
            if i not in self._static_attribute_indices
        ]
        self._X[:, dynamic_indices] = (
            self._X[:, dynamic_indices] - X_mean[dynamic_indices]
        ) / X_std[dynamic_indices]

        self._X_means, self._X_stds = X_mean, X_std
        (
            self._X_means[self._static_attribute_indices],
            self._X_stds[self._static_attribute_indices],
        ) = (0.0, 1.0)


def are_from_diff_distros_ks(x1: np.ndarray, x2: np.ndarray, alpha: float) -> bool:
    _, p_value = stats.ks_2samp(x1, x2)
    return p_value < alpha


def are_from_diff_distros_permutation_spearman(
    x1: np.ndarray, x2: np.ndarray, alpha: float
) -> bool:
    dof = len(x1) - 2

    def statistic(x):  # explore all possible pairings by permuting `x`
        rs = stats.spearmanr(x, x2).statistic  # ignore pvalue
        transformed = rs * np.sqrt(dof / ((rs + 1.0) * (1.0 - rs)))
        return transformed

    ref = stats.permutation_test(
        (x1,), statistic, alternative="greater", permutation_type="pairings"
    )
    return ref.pvalue < alpha


def are_from_diff_distros_spearman(
    x1: np.ndarray, x2: np.ndarray, alpha: float
) -> bool:
    pvalue = stats.spearmanr(x1, x2).pvalue
    return pvalue < alpha


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_neg_samples: int,
        match_alpha: float = 0.05,
        match_method: (
            Callable[[np.ndarray, np.ndarray, float], np.ndarray] | None
        ) = None,
    ):
        super(ContrastiveDataset, self).__init__()

        self.n_neg_samples = n_neg_samples
        self.match_alpha = match_alpha
        self.match_method = (
            match_method if match_method is not None else are_from_diff_distros_ks
        )
        self.x = x
        self.y = y

    def sample(self):
        temp_id = np.random.randint(0, len(self))
        return self.x[temp_id]

    def are_from_diff_distros(self, sample1: np.ndarray, sample2: np.ndarray) -> bool:
        return self.match_method(sample1, sample2, self.match_alpha)

    def sample_matching(self, x: np.ndarray):
        x_pos = None
        xs_neg = []

        while x_pos is None or len(xs_neg) < self.n_neg_samples:
            sampled = self.sample()
            are_from_diff_distros = self.are_from_diff_distros(x, sampled)

            if are_from_diff_distros and len(xs_neg) < self.n_neg_samples:
                xs_neg.append(sampled)

            if not are_from_diff_distros and x_pos is None:
                x_pos = sampled

        return x_pos, xs_neg

    def sample_close(self, idx: int):
        n_samples = len(self)
        pos_dist = self.match_alpha * n_samples
        pos_start = max(0, idx - pos_dist // 2)
        pos_end = min(idx + pos_dist // 2, n_samples - 1)
        pos_idx = np.random.randint(pos_start, pos_end)
        x_pos = self.x[pos_idx]

        neg_dist = n_samples - 1 - pos_dist * 2
        neg_ids = np.random.random_integers(0, neg_dist, self.n_neg_samples)
        neg_ids = neg_ids + np.where(neg_ids > pos_start, pos_dist, 0)
        xs_neg = self.x[neg_ids.astype(int)]

        return x_pos, list(xs_neg)

    def sample_by_distance(self, idx: int):
        y = self.y[idx]
        n_samples = len(self)

        pool_len = int(self.match_alpha * n_samples)
        pool_len = max(pool_len, self.n_neg_samples)

        dist_ids_sorted = np.argsort(abs(self.y - y))
        # neg_ids = dist_ids_sorted[-pool_len:]
        # neg_ids = np.random.choice(neg_ids, self.n_neg_samples, replace=False)
        neg_ids = np.argsort(abs(self.y - y))[-self.n_neg_samples :]
        xs_neg = self.x[neg_ids]

        pos_id = dist_ids_sorted[0]
        x_pos = self.x[pos_id]

        return x_pos, list(xs_neg)

    # def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     x = self.data[idx]

    #     # x_pos, xs_neg = self.sample_matching(x)
    #     x_pos, xs_neg = self.sample_close(idx)

    #     n_aux = 1 + self.n_neg_samples
    #     labels = np.ones(n_aux)
    #     labels[-1] = 0

    #     x_aux = list(xs_neg)
    #     x_aux.append(x_pos)

    #     x_exp_shape = np.ones(x.ndim + 1)
    #     x_exp_shape[0] = n_aux
    #     x_exp = np.tile(x, x_exp_shape.astype(int))

    #     return x_exp, np.stack(x_aux), labels

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.x[idx]

        # _, xs_neg = self.sample_close(idx)
        _, xs_neg = self.sample_by_distance(idx)

        labels = np.ones(self.n_neg_samples).astype(int)

        x_exp_shape = np.ones(x.ndim + 1)
        x_exp_shape[0] = len(xs_neg)
        x_exp = np.tile(x, x_exp_shape.astype(int))

        return x_exp, np.stack(xs_neg), labels

    def __len__(self):
        return len(self.x)

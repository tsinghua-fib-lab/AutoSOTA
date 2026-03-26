import math
import os
from abc import ABC
from pathlib import Path
from typing import Dict, Literal

import faiss
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from tqdm import tqdm

from .models import InContextModel
from .utils import sample_confidence_interval


def is_hf_model_path(model_path: str) -> bool:
    """
    Check if a given path is a Hugging Face model path (repo_id/model_name).

    Args:
        model_path: The model path to check

    Returns:
        bool: True if it's a Hugging Face model path, False otherwise
    """
    # Check if path doesn't exist locally but follows the pattern of org/repo or user/repo
    if not os.path.exists(model_path) and "/" in model_path and model_path.count("/") == 1:
        return True
    return False


def download_from_hf_hub(model_path: str, cache_dir: str) -> str:
    """
    Download a model from the Hugging Face Hub.

    Args:
        model_path: The model path in format 'org/repo' or 'user/repo'
        cache_dir: Optional directory to cache the downloaded model

    Returns:
        str: Path to the downloaded model file
    """
    # Download the model
    local_path = hf_hub_download(
        repo_id=model_path,
        filename="causalpfn_v0.pt",
        cache_dir=cache_dir,
    )
    return local_path


class CausalEstimator(ABC):
    """
    Shared functionalities between CATEEstimator and ATEEstimator.

    Args:
        device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
        model_path (str): The path to the model checkpoint. Can be:
            - A local file path
            - A Hugging Face model path
        cache_dir (str, optional): Directory to cache downloaded models from Hugging Face.
            Defaults to ~/.cache/causalpfn.
        calibrate (bool): Whether to calibrate the model's temperature using n-fold cross-validation.
        n_folds (int): The number of folds used for cross-validation to find the best calibration.
        ICE_n_bins (int): The number of bins used for the integrated coveragre error (ICE) calculation.
        ICE_n_samples (int): The number of samples used for integrated coveragre error (ICE) calculation.
        calibrate_T_min (float): The minimum temperature to use in calibration.
        calibrate_T_max (float): The maximum temperature to use in calibration.
        calibrate_T_size (int): The number of different temperature values to search during calibration.
        calibrate_T_batch_size (int): The batch size to use for calibration (should be <= `calibrate_T_size`).
        verbose (bool): Whether to print progress messages for the `predict_cep` function.

        *NOTE*: all of the following parameters only make sense when the data is too large
                to fit inside the context length of the model.

        max_context_length (int):
            The maximum number of context examples to use for each query.
            Set to as large as your GPU can handle.
        max_query_length (int):
            The maximum number of query examples to process at once.
            Set to as large as your GPU can handle.
        num_neighbours (int):
            The number of neerest neighbours to use for the CEPO prediction
    """

    def __init__(
        self,
        device: str,
        model_path: str = "vdblm/causalpfn",
        max_context_length: int = 4096,
        max_query_length: int = 4096,
        num_neighbours: int = 1024,
        calibrate: bool = False,
        n_folds: int = 3,
        calibrate_T_min: float = 0.001,
        calibrate_T_max: float = 10,
        calibrate_T_size: int = 500,
        calibrate_T_batch_size: int = 50,
        ICE_n_bins: int = 10,
        ICE_n_samples: int = 1000,
        verbose: bool = False,
        cache_dir: str | None = None,
    ):
        self.model_path = model_path
        self.cache_dir = cache_dir if cache_dir is not None else os.path.join(Path.home(), ".cache", "causalpfn")
        self.icl_model: InContextModel = None

        self.device = device
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        self.num_neighbours = num_neighbours

        # The maximum number of features to use for the model. If the number of features are
        # larger than this value, the model will apply PCA to reduce the dimensionality.
        self.max_feature_size = None
        self.x_dim_transformer = FunctionTransformer()  # identity transformer by default

        self.n_folds = n_folds

        self.calibrate = calibrate
        self.calibrate_T_min = calibrate_T_min
        self.calibrate_T_max = calibrate_T_max
        self.calibrate_T_size = calibrate_T_size
        self.calibrate_T_batch_size = calibrate_T_batch_size

        self.ICE_n_bins = ICE_n_bins
        self.ICE_n_samples = ICE_n_samples

        self.X_train, self.t_train, self.y_train = None, None, None
        self.temperature = 1.0
        self.prediction_temperature = 1.0

        self.verbose = verbose

    def _check_fitted(self):
        if self.X_train is None or self.t_train is None or self.y_train is None or self.icl_model is None:
            raise ValueError("The estimator must be fitted before calling the estimate function.")

    def load_model(self):
        """
        Load the model from the specified path or download it from Hugging Face.
        """
        model_path = self.model_path

        # Check if the model path is a Hugging Face model path
        if is_hf_model_path(model_path):
            model_path = download_from_hf_hub(model_path, self.cache_dir)

        # Load the model from the local path
        ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
        model_state = ckpt["model_state_dict"]
        config = ckpt["model_config"]

        self.icl_model = InContextModel.load(model_state=model_state, model_config=config).to(self.device)

        if config["model_type"] == "tabdpt":
            self.max_feature_size = config["model"]["max_num_features"] - 1
            self.x_dim_transformer = TruncatedSVD(n_components=self.max_feature_size, algorithm="arpack")

    @torch.no_grad()
    def _predict_cepo(
        self,
        X_context: np.ndarray,
        t_context: np.ndarray,
        y_context: np.ndarray,
        X_query: np.ndarray,
        t_query: np.ndarray,
        temperature: float,
        n_samples: int | None = None,
    ) -> np.ndarray:
        if self.icl_model is None:
            raise ValueError("CausalEstimator must be fitted before calling _predict_cepo.")

        temperature = torch.tensor([temperature], device=self.device)
        self.icl_model: InContextModel
        self.icl_model.eval()

        # list all of the point estimates as well as distributional estimates
        # of the CEPO in all_cepo and all_samples, respectively
        all_cepo = np.zeros((X_query.shape[0],), dtype=X_query.dtype)
        if n_samples is not None:
            all_samples = np.zeros((X_query.shape[0], n_samples), dtype=X_query.dtype)

        context_effects = self._estimate_cate_weak_learner(X_test=X_context)
        treatmentgroup_2_context_idx = np.where(np.isclose(t_context, 1))[0]
        controlgroup_2_context_idx = np.where(np.isclose(t_context, 0))[0]
        context_treatment_group_effects = context_effects[treatmentgroup_2_context_idx]
        context_control_group_effects = context_effects[controlgroup_2_context_idx]

        query_effects = self._estimate_cate_weak_learner(X_test=X_query)
        query_indices_sorted = np.argsort(query_effects)

        index_treatment = faiss.IndexFlatL2(1)
        index_treatment.add(
            np.ascontiguousarray(context_treatment_group_effects.reshape(-1, 1).copy(), dtype=np.float32)
        )
        _, query_neighbour_indices_treatment = index_treatment.search(
            np.ascontiguousarray(query_effects.reshape(-1, 1).copy(), dtype=np.float32), k=self.num_neighbours
        )
        index_control = faiss.IndexFlatL2(1)
        index_control.add(np.ascontiguousarray(context_control_group_effects.reshape(-1, 1).copy(), dtype=np.float32))
        _, query_neighbour_indices_control = index_control.search(
            np.ascontiguousarray(query_effects.reshape(-1, 1).copy(), dtype=np.float32), k=self.num_neighbours
        )

        # if the query size is large, we split it into batches
        pbar = tqdm(range(X_query.shape[0]), desc="Predicting CEPO", total=X_query.shape[0], disable=not self.verbose)
        start_idx = 0
        while start_idx < X_query.shape[0]:

            def get_stratum(query_neighbour_indices):
                # binary search to find the maximum window size for the query
                # that has neighbours less than the maximum context length
                L = start_idx + 1
                R = min(start_idx + self.max_query_length, X_query.shape[0])
                while R > L + 1:
                    mid = (L + R) // 2
                    tmp_idx_q = query_indices_sorted[start_idx:mid]
                    unique_indices = np.unique(query_neighbour_indices[tmp_idx_q].flatten())
                    if len(unique_indices) > self.max_context_length // 2:
                        R = mid - 1
                    else:
                        L = mid
                return min(L, X_query.shape[0])

            control_end_idx = get_stratum(query_neighbour_indices_control)
            control_q_idx = query_indices_sorted[start_idx:control_end_idx]
            all_neighbours = query_neighbour_indices_control[control_q_idx].flatten()
            all_neighbours = np.unique(all_neighbours)
            real_control_idx = controlgroup_2_context_idx[all_neighbours]
            treatment_end_idx = get_stratum(query_neighbour_indices_treatment)
            treatment_q_idx = query_indices_sorted[start_idx:treatment_end_idx]
            all_neighbours = query_neighbour_indices_treatment[treatment_q_idx].flatten()
            all_neighbours = np.unique(all_neighbours)
            real_treatment_idx = treatmentgroup_2_context_idx[all_neighbours]

            # now use that window size to get the query and context
            end_idx = min(control_end_idx, treatment_end_idx)
            idx_q = query_indices_sorted[start_idx:end_idx]
            x_q = X_query[idx_q]
            t_q = t_query[idx_q]

            # find the closest neighbours in terms of the stratifying CATE
            all_neighbours = np.concatenate([real_control_idx, real_treatment_idx])
            assert (
                len(all_neighbours) <= self.max_context_length
            ), f"Number of neighbours {len(all_neighbours)} is larger than the maximum context length {self.max_context_length}."
            x_c = X_context[all_neighbours]
            t_c = t_context[all_neighbours]
            y_c = y_context[all_neighbours]

            res = self.icl_model.predict_cepo(
                # shape: (1, context_size, num_features)
                X_context=torch.from_numpy(x_c).to(self.device).unsqueeze(0).float(),
                # shape: (1, context_size)
                t_context=torch.from_numpy(t_c).to(self.device).unsqueeze(0).float(),
                y_context=torch.from_numpy(y_c).to(self.device).unsqueeze(0).float(),
                # shape: (1, query_size, num_features)
                X_query=torch.from_numpy(x_q).to(self.device).unsqueeze(0).float(),
                # shape: (1, query_size)
                t_query=torch.from_numpy(t_q).to(self.device).unsqueeze(0).float(),
                n_samples=n_samples,
                temperature=temperature,
            )
            if n_samples is None:
                cepo = res.squeeze(0).squeeze(0)  # shape: (query_size,)
            else:
                cepo, samples = (
                    res[0].squeeze(0).squeeze(0),
                    res[1].squeeze(0).squeeze(0),
                )  # shapes: (query_size,), (query_size, n_samples)

            all_cepo[idx_q] = cepo.cpu().numpy()
            if n_samples is not None:
                all_samples[idx_q] = samples.cpu().numpy()
            pbar.update(end_idx - start_idx)
            start_idx = end_idx

        if n_samples is not None:
            return all_cepo, all_samples

        return all_cepo

    @torch.no_grad()
    def _calculate_reg_ice(
        self,
        X: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:

        alpha_values = torch.linspace(0.01, 1.0, self.ICE_n_bins, device=self.device)  # significance levels
        ce_values = torch.zeros_like(temperature, device=self.device)

        fold_indices = np.array_split(np.arange(len(X)), self.n_folds)
        for val_idx in fold_indices:
            train_idx = np.setdiff1d(np.arange(len(X)), val_idx)

            X_context, t_context, y_context = X[train_idx], t[train_idx], y[train_idx]
            X_query, t_query, y_query = X[val_idx], t[val_idx], y[val_idx]

            b_size = math.ceil(X_query.shape[0] / self.max_query_length)
            batch_ice_values = torch.zeros_like(temperature, device=self.device)
            for b in range(b_size):
                batch_start = b * self.max_query_length
                batch_end = min((b + 1) * self.max_query_length, X_query.shape[0])
                if batch_start >= batch_end:
                    continue
                X_q = X_query[batch_start:batch_end]
                t_q = t_query[batch_start:batch_end]
                y_q = y_query[batch_start:batch_end]  # shape: (len(X_q), )

                if t_context.shape[0] > self.max_context_length:
                    # sample random set of context points
                    context_indices = np.random.choice(t_context.shape[0], self.max_context_length, replace=False)
                else:
                    context_indices = np.arange(t_context.shape[0])

                res = self.icl_model.predict_cepo(
                    # shape: (1, context_size, num_features)
                    X_context=X_context[context_indices].unsqueeze(0).float(),
                    # shape: (1, context_size)
                    t_context=t_context[context_indices].unsqueeze(0).float(),
                    y_context=y_context[context_indices].unsqueeze(0).float(),
                    # shape: (1, query_size, num_features)
                    X_query=X_q.unsqueeze(0).float(),
                    # shape: (1, query_size)
                    t_query=t_q.unsqueeze(0).float(),
                    n_samples=self.ICE_n_samples,
                    temperature=temperature,
                )
                samples = res[1].squeeze(0)  # shape: (len(temperature), len(X_q), n_samples)

                # shape: (len(alpha_values, len(temperature), len(X_q))
                lower_bound, upper_bound = sample_confidence_interval(samples, alphas=alpha_values)
                # shape: (len(alpha_values), len(temperature))
                coverage = (
                    ((y_q[None, None, :] >= lower_bound) & (y_q[None, None, :] <= upper_bound)).float().mean(axis=-1)
                )

                batch_ice_values += torch.abs(coverage - (1 - alpha_values[:, None])).mean(axis=0)

            ce_values += batch_ice_values / b_size

        return ce_values / self.n_folds

    def _calibrate_fn(self, X: np.ndarray, t: np.ndarray, y: np.ndarray):
        """
        Calibrate the temperature using n-fold cross-validation.
        """

        X = torch.from_numpy(X).to(self.device)
        t = torch.from_numpy(t).to(self.device)
        y = torch.from_numpy(y).to(self.device)

        temperatures = torch.logspace(
            start=np.log10(self.calibrate_T_min),
            end=np.log10(self.calibrate_T_max),
            steps=self.calibrate_T_size,
            device=self.device,
        )

        all_losses = []
        batch_size = self.calibrate_T_batch_size
        num_batches = math.ceil(len(temperatures) / batch_size)

        pbar = tqdm(range(num_batches), desc="Calibrating temperature", disable=not self.verbose)
        for i in pbar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(temperatures))
            batch_temperatures = temperatures[start_idx:end_idx]

            batch_losses = self._calculate_reg_ice(X, t, y, temperature=batch_temperatures)
            all_losses.append(batch_losses)
            pbar.update(i)

        losses = torch.cat(all_losses, dim=0)
        best_idx = torch.argmin(losses).item()
        self.temperature = temperatures[best_idx].item()

    def _train_weak_learner(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        self.t_transformer = OneHotEncoder(sparse_output=False, categories="auto", drop="first")
        T = self.t_transformer.fit_transform(t.reshape(-1, 1))
        self._d_t = (T.shape[1],)
        feat_arr = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
        self.stratifier = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=max(2, int(X.shape[0] / 100)),
            random_state=111,
        )
        self.stratifier.fit(feat_arr, y)

    def _estimate_cate_weak_learner(self, X_test: np.ndarray) -> np.ndarray:
        if self.stratifier is None or self.t_transformer is None:
            raise ValueError("Weak learner must be trained before estimating CATE.")

        t0 = np.zeros(X_test.shape[0], dtype=X_test.dtype)
        t1 = np.ones(X_test.shape[0], dtype=X_test.dtype)
        T0 = self.t_transformer.transform(t0.reshape(-1, 1))
        T1 = self.t_transformer.transform(t1.reshape(-1, 1))
        X_T0 = np.concatenate((X_test, 1 - np.sum(T0, axis=1).reshape(-1, 1), T0), axis=1)
        X_T1 = np.concatenate((X_test, 1 - np.sum(T1, axis=1).reshape(-1, 1), T1), axis=1)
        Y0 = self.stratifier.predict(X_T0)
        Y1 = self.stratifier.predict(X_T1)

        return Y1 - Y0

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> "CausalEstimator":
        """
        Fit the model using the provided data.

        Args:
            X (np.ndarray): The observational covariate data with shape [N, D].
            t (np.ndarray): The observational treatment data with shape [N].
            y (np.ndarray): The observational outcome data with shape [N].
        """
        self.temperature = 1.0

        # load the model
        self.load_model()

        # set the x_dim_transform and transform the data
        if self.max_feature_size is not None and X.shape[1] > self.max_feature_size:
            X = self.x_dim_transformer.fit_transform(X)

        self.X_train = X
        self.t_train = t
        self.y_train = y

        # train a lightweight stratification model to use whenever the context length is too large
        # this ensures that the data will be stratified and the context would only look at the most relevant samples
        # for the given query
        self._train_weak_learner(X, t, y)

        if self.calibrate:
            self._calibrate_fn(
                X=self.X_train,
                t=self.t_train,
                y=self.y_train,
            )
        return self


class CATEEstimator(CausalEstimator):
    def estimate_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the conditional average treatment effect (CATE) using the fitted model.
        Args:
            X (np.ndarray): The input data with shape [N', D].
        """
        self._check_fitted()

        X_context = self.X_train
        t_context = self.t_train
        y_context = self.y_train
        X_query = X
        if self.max_feature_size is not None and X_query.shape[1] > self.max_feature_size:
            X_query = self.x_dim_transformer.transform(X_query)

        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)

        mu_0_and_1 = self._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            temperature=self.prediction_temperature,
        )

        mu_0 = mu_0_and_1[: X_query.shape[0]]
        mu_1 = mu_0_and_1[X_query.shape[0] :]
        return mu_1 - mu_0

    def _estimate_ate_cate_CI(
        self, X: np.ndarray, alpha: float = 0.05, n_samples: int = 10_000
    ) -> Dict[str, np.ndarray]:
        self._check_fitted()

        X_context = self.X_train
        t_context = self.t_train
        y_context = self.y_train
        X_query = X
        if self.max_feature_size is not None and X_query.shape[1] > self.max_feature_size:
            X_query = self.x_dim_transformer.transform(X_query)

        t_all_ones = np.ones(X_query.shape[0], dtype=X_query.dtype)
        t_all_zeros = np.zeros(X_query.shape[0], dtype=X_query.dtype)

        _, samples = self._predict_cepo(
            X_context=X_context,
            t_context=t_context,
            y_context=y_context,
            X_query=np.concatenate([X_query, X_query], axis=0),
            t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
            n_samples=n_samples,
            temperature=self.temperature,
        )

        # calculate the confidence intervals
        samples_0 = samples[: X_query.shape[0]]
        samples_1 = samples[X_query.shape[0] :]

        cate_samples = samples_1 - samples_0

        lower_bound, upper_bound = sample_confidence_interval(
            torch.from_numpy(cate_samples).float(), alphas=torch.tensor([alpha]).float()
        )

        ate_samples = samples_1.mean(axis=0) - samples_0.mean(axis=0)
        ate_lower_bound, ate_upper_bound = sample_confidence_interval(
            torch.from_numpy(ate_samples).float(), alphas=torch.tensor([alpha]).float()
        )
        return {
            "cate_lower_bound": lower_bound.numpy(),
            "cate_upper_bound": upper_bound.numpy(),
            "ate_lower_bound": ate_lower_bound.numpy(),
            "ate_upper_bound": ate_upper_bound.numpy(),
        }

    def estimate_cate_CI(self, X: np.ndarray, alpha: float = 0.05, n_samples: int = 10_000) -> Dict[str, np.ndarray]:
        """
        Estimate the conditional average treatment effect (CATE) with confidence intervals using the fitted model.

        Args:
            X (np.ndarray): The input data with shape [N', D].
            alpha (float): The significance level for the confidence interval.
            n_samples (int): The number of samples to use for estimating the confidence interval.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the confidence intervals.
                - "lower_bound": The lower bound of the confidence interval.
                - "upper_bound": The upper bound of the confidence interval.
        """
        output = self._estimate_ate_cate_CI(X, alpha=alpha, n_samples=n_samples)
        return {
            "lower_bound": output["cate_lower_bound"],
            "upper_bound": output["cate_upper_bound"],
        }

    def estimate_ate(self, X: np.ndarray) -> float:
        """
        Estimate the average treatment effect (ATE) using the fitted model and CATE values.
        This is different from the ATEEstimator in that it just simply computes the CATE values and averages them.
        """
        cate_pred = self.estimate_cate(X)
        return cate_pred.mean()

    def estimate_ate_CI(self, X: np.ndarray, alpha: float = 0.05, n_samples: int = 10_000) -> Dict[str, np.ndarray]:
        """
        Estimate the average treatment effect (ATE) with confidence intervals using the fitted model and CATE values.
        This is different from the ATEEstimator in that it just simply computes the CATE values and averages them.

        Args:
            X (np.ndarray): The input data with shape [N', D].
            alpha (float): The significance level for the confidence interval.
            n_samples (int): The number of samples to use for estimating the confidence interval.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the ATE estimates and the confidence intervals.
                - "ate": The ATE estimates.
                - "lower_bound": The lower bound of the confidence interval.
                - "upper_bound": The upper bound of the confidence interval.
        """
        output = self._estimate_ate_cate_CI(X, alpha=alpha, n_samples=n_samples)
        return {
            "ate": output["ate"],
            "lower_bound": output["ate_lower_bound"],
            "upper_bound": output["ate_upper_bound"],
        }


class ATEEstimator(CausalEstimator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu_0 = None
        self.mu_1 = None

    def _estimate_ate_ra_learner(self) -> float:

        mu_1_final = np.where(np.isclose(self.t_train, 1), self.y_train, self.mu_1)
        mu_0_final = np.where(np.isclose(self.t_train, 0), self.y_train, self.mu_0)

        return (mu_1_final - mu_0_final).mean()

    def _estimate_ate_s_learner(self) -> float:
        return (self.mu_1 - self.mu_0).mean()

    def estimate_ate(self, algorithm: Literal["RA", "S"] = "S") -> float:
        """
        Estimate the average treatment effect (ATE) using the fitted model.
        Args:
            algorithm (str):
                When set to "S", it simply uses the model-output CEPO values,
                When set to "RA", it uses the model only to get access to the counterfactuals
        """
        self._check_fitted()
        if self.mu_0 is None or self.mu_1 is None:
            t_all_ones = np.ones(self.X_train.shape[0], dtype=self.X_train.dtype)
            t_all_zeros = np.zeros(self.X_train.shape[0], dtype=self.X_train.dtype)

            mu_0_and_1 = self._predict_cepo(
                X_context=self.X_train,
                t_context=self.t_train,
                y_context=self.y_train,
                X_query=np.concatenate([self.X_train, self.X_train], axis=0),
                t_query=np.concatenate([t_all_zeros, t_all_ones], axis=0),
                temperature=self.prediction_temperature,
            )
            self.mu_0 = mu_0_and_1[: self.X_train.shape[0]]
            self.mu_1 = mu_0_and_1[self.X_train.shape[0] :]

        if algorithm == "S":
            return self._estimate_ate_s_learner()
        elif algorithm == "RA":
            return self._estimate_ate_ra_learner()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Supported algorithms are 'RA', and 'S'.")

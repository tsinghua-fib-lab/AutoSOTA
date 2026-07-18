from __future__ import annotations

"""Abstract black‑box function class and various concrete implementations.

Every concrete black‑box must inherit :class:`BaseBlackboxFunction` and
implement :meth:`evaluate`.
"""

import copy
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ConfigSpace
import numpy as np
import pandas as pd
import pyarrow as pa
import shapiq
import sklearn
import torch
from botorch.test_functions import synthetic as botorch_synthetic
from pyarrow import parquet as pq
from shapiq.utils import transform_array_to_coalitions
from shapiq.utils.saving import safe_str_to_tuple, safe_tuple_to_str
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor
# from tabpfn_client import TabPFNClassifier
from tabpfn.constants import ModelVersion
from yahpo_gym import *

os.environ["TABPFN_DISABLE_PROGRESS"] = "1"

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class BaseBlackboxFunction(ABC):
    """Base class for deterministic or stochastic black‑box functions."""

    deterministic: bool = True
    is_pseudo_expensive: bool = False

    # ------------------------------------------------------------------
    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Evaluate at `X`.

        Parameters
        ----------
        X : (N × D) tensor
            Each row is one input point.

        Returns
        -------
        y : (N,) tensor or (N × K) tensor
            Function values per input row.
        """

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.evaluate(X)

    @property
    @abstractmethod
    def plot_name(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def cat_dims(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def log_trafo_dims(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def bounds(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def indep_attr_names(self) -> torch.Tensor:
        pass


# -----------------------------------------------------------------------------
# Blackbox functions: Synthetic benchmark functions from BoTorch
# -----------------------------------------------------------------------------

BotorchTestFunctionName = Literal[
    "Branin",
    "StyblinskiTang_3",
    "Hartmann_6",
    "Levy_10",
    "Ackley_10",
    "Ackley_20",
    "Ackley_30",
    "Ackley_40",
    "Ackley_50",
    "Ackley_100",
    "Ackley_8",
    "Ackley_16",
    "Ackley_32",
    "Ackley_64",
    "Ackley_128",
    "Ackley_256",
    "Ackley_512",
]


@dataclass(frozen=True)
class BotorchTestFunction(BaseBlackboxFunction):
    name: BotorchTestFunctionName = "Branin"
    seed: int = 42

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):
        botorch_name = self.name.split("_")[0]

        kwargs = {"noise_std": None}
        if len(self.name.split("_")) > 1:
            kwargs["dim"] = int(self.name.split("_")[1])

        botorch_fn = getattr(botorch_synthetic, botorch_name)(**kwargs)

        object.__setattr__(self, "botorch_fn", botorch_fn)
        object.__setattr__(self, "dim", botorch_fn.dim)
        object.__setattr__(self, "_bounds", botorch_fn.bounds)

    def get_bounds_for_dim(self, dim: int = 0):
        return self._bounds[:, dim]

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return (torch.unsqueeze(self.botorch_fn.evaluate_true(X=X), dim=1), None)

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]


# -----------------------------------------------------------------------------
# Blackbox functions: Yahpo benchmark surrogates
# -----------------------------------------------------------------------------

YahpoScenarioName = Literal[
    "lcbench_3945",
    "lcbench_168868",
    "lcbench_167200",
    "lcbench_168330",
    "lcbench_189862",  # "rbv2_ranger_41157", #Notimplemented yet #"rbv2_ranger_40984", #"rbv2_ranger_54",
    "rbv2_xgboost_3",
    "rbv2_xgboost_54",
    "rbv2_xgboost_38",
    "rbv2_aknn_469",
    "rbv2_aknn_181",
    "rbv2_aknn_40496",
]
# YahpoInstance = Literal["3945"]


@dataclass(frozen=True)
class YahpoSurrogate(BaseBlackboxFunction):
    name: YahpoScenarioName = "lcbench_3945"
    seed: int = 42

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):
        parts = self.name.split("_")
        yahpo_name = "_".join(parts[:-1]) if len(parts) > 2 else parts[0]
        instance = int(parts[-1])

        task_id_column_name = "OpenML_task_id" if yahpo_name == "lcbench" else "task_id"

        object.__setattr__(self, "yahpo_name", yahpo_name)
        object.__setattr__(self, "instance", instance)
        object.__setattr__(self, "task_id_column_name", task_id_column_name)

        from yahpo_gym import local_config

        local_config.init_config()
        local_config.set_data_path("data/yahpo_surrogates/yahpo_data")

        benchmark_set = BenchmarkSet(scenario=self.yahpo_name)
        benchmark_set.set_instance(str(self.instance))

        object.__setattr__(self, "benchmark_set", benchmark_set)
        object.__setattr__(
            self, "yahpo_opt_space", benchmark_set.get_opt_space(seed=self.seed)
        )

        object.__setattr__(
            self,
            "task_id_index",
            self.yahpo_opt_space._hyperparameter_idx[self.task_id_column_name],
        )

        object.__setattr__(
            self,
            "task_id_value",
            self.yahpo_opt_space.sample_configuration(1).get_array()[
                self.task_id_index
            ],
        )

        if self.yahpo_name == "rbv2_xgboost":
            object.__setattr__(
                self,
                "booster_index",
                self.yahpo_opt_space._hyperparameter_idx["booster"],
            )

            def get_xgboost_dart_config():
                temp_config = self.yahpo_opt_space.sample_configuration(1)

                while temp_config["booster"] != "dart":
                    temp_config = self.yahpo_opt_space.sample_configuration(1)

                return temp_config

            object.__setattr__(
                self,
                "booster_value",
                get_xgboost_dart_config().get_array()[self.booster_index],
            )

        assert int(
            self.yahpo_opt_space.sample_configuration(1)[self.task_id_column_name]
        ) == int(self.instance)

        if yahpo_name == "lcbench":
            object.__setattr__(self, "perf_metric_name", "val_accuracy")
            object.__setattr__(self, "cost_metric_name", "time")
        elif yahpo_name == "rbv2_aknn":
            object.__setattr__(self, "perf_metric_name", "acc")
            object.__setattr__(self, "cost_metric_name", "timetrain")
        elif yahpo_name == "rbv2_xgboost":
            object.__setattr__(self, "perf_metric_name", "acc")
            object.__setattr__(self, "cost_metric_name", "timetrain")
        else:
            raise NotImplementedError(
                f"Yahpo scenario {yahpo_name} not implemented in blackbox_functions.py"
            )

        object.__setattr__(self, "_bounds", None)

    def get_bounds_for_dim(self, dim: int = 0):
        return self._bounds[:, dim]

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        # expects X without openml id
        if self.yahpo_name == "rbv2_xgboost":
            task_id_index = self.task_id_index
            task_id_value = self.task_id_value

            booster_index = self.booster_index
            booster_value = self.booster_value

            if task_id_index < booster_index:
                X_list = [
                    ConfigSpace.Configuration(
                        configuration_space=self.yahpo_opt_space,
                        vector=np.concatenate(
                            (
                                [
                                    X[i, :task_id_index],
                                    np.array([task_id_value]),
                                    X[i, task_id_index:booster_index],
                                    np.array([booster_value]),
                                    X[i, booster_index:],
                                ]
                            )
                        ),
                    )
                    for i in range(X.shape[0])
                ]

            else:
                X_list = [
                    ConfigSpace.Configuration(
                        configuration_space=self.yahpo_opt_space,
                        vector=np.concatenate(
                            (
                                [
                                    X[i, :booster_index],
                                    np.array([booster_value]),
                                    X[i, booster_index:task_id_index],
                                    np.array([task_id_value]),
                                    X[i, task_id_index:],
                                ]
                            )
                        ),
                    )
                    for i in range(X.shape[0])
                ]

        else:
            X_list = [
                ConfigSpace.Configuration(
                    configuration_space=self.yahpo_opt_space,
                    vector=np.concatenate(
                        (
                            [
                                X[i, : self.task_id_index],
                                np.array([self.task_id_value]),
                                X[i, self.task_id_index :],
                            ]
                        )
                    ),
                )
                for i in range(X.shape[0])
            ]

        X_openml_ids = torch.unsqueeze(
            torch.tensor(
                [int(elem[self.task_id_column_name]) for elem in X_list],
                dtype=torch.float64,
            ),
            dim=1,
        )
        assert X_openml_ids.unique() == self.instance

        if self.yahpo_name == "rbv2_xgboost":
            # X_booster_values = torch.unsqueeze(
            #     torch.tensor(
            #         [elem['booster'] for elem in X_list],
            #         dtype=torch.float64,
            #     ),
            #     dim=1,
            # )

            X_booster_unique_values = set([elem["booster"] for elem in X_list])
            assert len(X_booster_unique_values) == 1
            assert "dart" in X_booster_unique_values

        # Same for booster

        # config space is not tied to instance, but opt space is
        # remove openml id  and filter configs such that all entries are different, then outside bbf there is no logic

        res = self.benchmark_set.objective_function(X_list)

        res_acc = torch.unsqueeze(
            torch.tensor(
                [row[self.perf_metric_name] for row in res], dtype=torch.float64
            ),
            dim=1,
        )
        res_time = torch.unsqueeze(
            torch.tensor(
                [row[self.cost_metric_name] for row in res], dtype=torch.float64
            ),
            dim=1,
        )

        return (res_acc, res_time)

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]


# -----------------------------------------------------------------------------
# Blackbox functions: TabRepo benchmark
# -----------------------------------------------------------------------------

TabRepoAlgorithm = Literal[
    "catboost",
    "xt",
    "knn",
    "lightgbm",
    "nn_torch",
    "rf",
    "xgboost",
]

TabRepoPerfMetric = Literal["metric_error", "metric_error_val"]
TabRepoCostMetric = Literal["time_train_s", "time_infer_s"]


@dataclass(frozen=True)
class TabRepoBenchmark(BaseBlackboxFunction):
    """TabRepo benchmark."""

    name: TabRepoAlgorithm = "catboost"
    perf_metric: TabRepoPerfMetric = "metric_error"
    cost_metric: TabRepoCostMetric = "time_train_s"
    include_cat: bool = False

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):

        # ------------------------------------------------------------------
        # 1. Load data
        # ------------------------------------------------------------------

        # 1.1 Performance data
        performance_data_path = "data/tabrepo/performance_data/configs.csv"

        if os.path.exists(performance_data_path):
            performance_data = pd.read_csv(performance_data_path)
        else:
            raise FileNotFoundError(
                f"CSV file not found at {performance_data_path}. Please download as explained in README.md."
            )

        # 1.2 Config data
        config_data_path = f"""data/tabrepo/config_data/configs_{self.name}.json"""

        if os.path.exists(config_data_path):
            with open(config_data_path, "r") as f:
                config_data = json.load(f)
        else:
            raise FileNotFoundError(
                f"JSON file not found at {config_data_path}. Please download as explained in README.md."
            )

        # 1.3 Task metadata
        tasks_metadata_path = "data/tabrepo/tasks_metadata/task_metadata_289.csv"
        # Consider joining other OpenML meta features

        if os.path.exists(tasks_metadata_path):
            tasks_metadata = pd.read_csv(tasks_metadata_path)
        else:
            raise FileNotFoundError(
                f"CSV file not found at {tasks_metadata_path}. Please download as explained in README.md."
            )

        # ------------------------------------------------------------------
        # 2. Identify suitable configs and filter data accordingly
        # ------------------------------------------------------------------

        # 2.1 Only consider non-default configs (as for default-configs the configurations are not completely defined)
        config_names = list(config_data.keys())
        default_config_names = [
            temp_name
            for temp_name in config_names
            if "_" in temp_name and "c" in temp_name.split("_", 1)[1]
        ]
        non_default_config_names = [
            temp_name
            for temp_name in config_names
            if "_" in temp_name and "r" in temp_name.split("_", 1)[1]
        ]

        algorithm_names = list(
            performance_data["framework"].str.split("_").str[0].unique()
        )
        algorithm_name = config_names[0].split("_")[
            0
        ]  # Equivalent term to name in self.name in performance_data

        # 2.2 Prepare associated config-values
        config_data = pd.DataFrame.from_dict(
            [
                config_data[temp_conf_name]["hyperparameters"]
                for temp_conf_name in non_default_config_names
            ]
        )
        config_data["name_suffix"] = config_data["ag_args"].apply(
            lambda d: d["name_suffix"]
        )
        config_data = config_data.drop(["ag_args"], axis=1)

        # 2.3 Filter configs to those where non-numeric columns take the most frequent value
        # Select all non-numeric columns (except name_suffix)
        non_num_cols = config_data.select_dtypes(exclude=[np.number]).columns.drop(
            "name_suffix", errors="ignore"
        )

        if len(non_num_cols) > 0:
            # For each column compute the most frequent value
            modes = {}
            for col in non_num_cols:
                m = config_data[col].mode(dropna=True)
                if not m.empty:
                    modes[col] = m.iat[0]

            # Each selected column must equal its mode
            mask = pd.concat(
                [config_data[col].eq(val) for col, val in modes.items()], axis=1
            ).all(axis=1)

            config_data = config_data.loc[mask].copy()
        else:
            config_data = config_data

        config_data = config_data.drop(non_num_cols, axis=1)

        assert (
            config_data.shape[0] > 30
        )  # Ensure that at least 30 admissible configs remain

        config_suffixes = list(config_data["name_suffix"])

        # 2.4: Filter dataset to admissible configs
        config_names = [
            algorithm_name + temp_config_suffix
            for temp_config_suffix in config_suffixes
        ]
        performance_data = performance_data[
            performance_data["framework"]
            .str.split("_", n=2)
            .str[:2]
            .str.join("_")
            .isin(config_names)
        ]

        # ------------------------------------------------------------------
        # 3. Filter dataset
        # ------------------------------------------------------------------
        # 3.1 Filter dataset to binary classification tasks (as ROC AUC is on same scale and we must not normalize targets)
        performance_data = performance_data[performance_data["metric"] == "roc_auc"]

        # 3.2 Group identical configs on different folds
        group_cols = ["dataset", "tid", "framework", "metric", "problem_type"]

        performance_data = (
            performance_data.groupby(group_cols)
            .agg(
                {
                    "fold": list,  # collect all folds into a list
                    "metric_error": "mean",
                    "metric_error_val": "mean",
                    "time_train_s": "mean",
                    "time_infer_s": "mean",
                }
            )
            .reset_index()
        )

        if not (performance_data["fold"].apply(len) == 3).all():
            log.info(
                "Warning! For some configs and datasets, not all three folds have been evaluated."
            )

        # ------------------------------------------------------------------
        # 3. Merge config-data and task meta-data
        # ------------------------------------------------------------------

        # 3.1 Merge config data
        # Add config sufix column
        performance_data["name_suffix"] = (
            "_" + performance_data["framework"].str.split("_").str[1]
        )

        performance_data = performance_data.merge(config_data, on=["name_suffix"])

        # Track names of independent attributes refering to the config
        config_indep_attr_names = list(config_data.columns)
        config_indep_attr_names.remove("name_suffix")

        # Hard code attributes to log transform (according to paper)
        config_overall_log_trafo_attr_names = ["learning_rate", "weight_decay"]

        # Reduce to those occuring in this specific config
        config_log_trafo_attr_names = [
            temp_attr
            for temp_attr in config_overall_log_trafo_attr_names
            if temp_attr in config_indep_attr_names
        ]

        # 3.2 Merge OpenML data
        # Merge OpenML ID
        performance_data = performance_data.merge(
            tasks_metadata[["tid", "did"]], on=["tid"]
        )

        import openml

        task_indep_attr_names = [
            "NumberOfFeatures",
            "NumberOfInstances",
            "Dimensionality",
            "ClassEntropy",
            "MinorityClassPercentage",
            "PercentageOfMissingValues",
            "MeanMutualInformation",
            "MeanSkewnessOfNumericAtts",
            "NaiveBayesErrRate",
            "kNN1NErrRate",
        ]  # must be all continuous

        openml_log_trafo_meta_feature_names = [
            "NumberOfFeatures",
            "NumberOfInstances",
            "Dimensionality",
        ]

        # Fetch OpenML metadata
        openml_ids = performance_data["did"].unique()

        openml_meta_features_collection = []
        for temp_openml_id in openml_ids:
            temp_meta_features = openml.datasets.get_dataset(
                int(temp_openml_id), download_data=False
            ).qualities
            temp_meta_features["did"] = int(temp_openml_id)
            openml_meta_features_collection.append(temp_meta_features)

        openml_meta_features_df = pd.DataFrame(openml_meta_features_collection)[
            ["did"] + task_indep_attr_names
        ]

        performance_data = performance_data.merge(openml_meta_features_df, on=["did"])
        # Caution: task_indep_attr_names Contains nans

        # 1.8 Filter constant columns
        const_cols = [
            col
            for col in performance_data[
                config_indep_attr_names + task_indep_attr_names
            ].columns
            if performance_data[col].nunique(dropna=True) <= 1
        ]

        performance_data = performance_data.drop(columns=const_cols)

        config_indep_attr_names = [
            col for col in config_indep_attr_names if col not in const_cols
        ]
        task_indep_attr_names = [
            col for col in task_indep_attr_names if col not in const_cols
        ]

        # 1.9 Infer config id
        performance_data["config_id"] = (
            performance_data["framework"].str.split("_", n=2).str[1].str[1:].astype(int)
        )

        assert (
            len(performance_data["config_id"].value_counts().unique()) == 1
        )  # Assert that all configs are evaluated on same amount of datasets

        # ------------------------------------------------------------------
        # 4. Map to tensor
        # ------------------------------------------------------------------

        dataset_attr_names = (
            config_indep_attr_names
            + task_indep_attr_names
            + [self.perf_metric]
            + [self.cost_metric]
            + ["config_id"]
            + ["did"]
        )
        dataset_tensor = torch.tensor(
            performance_data[dataset_attr_names].values, dtype=torch.float64
        )

        # ------------------------------------------------------------------
        # 5. Persist relevant objects
        # ------------------------------------------------------------------

        object.__setattr__(self, "dataset", dataset_tensor)
        object.__setattr__(self, "performance_data", performance_data)

        object.__setattr__(
            self, "_indep_attr_names", config_indep_attr_names + task_indep_attr_names
        )
        object.__setattr__(self, "config_indep_attr_names", config_indep_attr_names)
        object.__setattr__(self, "task_indep_attr_names", task_indep_attr_names)

        object.__setattr__(
            self, "cat_attr_names", []
        )  # Explicitly filtered cat attributes
        object.__setattr__(
            self,
            "log_trafo_attr_names",
            config_log_trafo_attr_names + openml_log_trafo_meta_feature_names,
        )

        # Indices
        indep_attr_idx = torch.tensor(
            [dataset_attr_names.index(name) for name in self._indep_attr_names]
        )
        config_indep_attr_idx = torch.tensor(
            [dataset_attr_names.index(name) for name in self.config_indep_attr_names]
        )
        task_indep_attr_idx = torch.tensor(
            [dataset_attr_names.index(name) for name in self.task_indep_attr_names]
        )

        cat_attr_idx = torch.tensor(
            [dataset_attr_names.index(name) for name in self.cat_attr_names]
        )
        log_trafo_attr_idx = torch.tensor(
            [dataset_attr_names.index(name) for name in self.log_trafo_attr_names]
        )

        perf_metric_idx = torch.tensor([dataset_attr_names.index(self.perf_metric)])
        cost_metric_idx = torch.tensor([dataset_attr_names.index(self.cost_metric)])

        object.__setattr__(self, "indep_attr_idx", indep_attr_idx)
        object.__setattr__(self, "config_indep_attr_idx", config_indep_attr_idx)
        object.__setattr__(self, "task_indep_attr_idx", task_indep_attr_idx)
        object.__setattr__(self, "cat_attr_idx", cat_attr_idx)
        object.__setattr__(self, "log_trafo_attr_idx", log_trafo_attr_idx)
        object.__setattr__(self, "perf_metric_idx", perf_metric_idx)
        object.__setattr__(self, "cost_metric_idx", cost_metric_idx)

        object.__setattr__(self, "config_id_idx", -2)
        object.__setattr__(self, "dataset_id_idx", -1)

        # Infer bounds
        _bounds = torch.tensor(
            [
                [
                    self.dataset[
                        ~torch.isnan(self.dataset[:, temp_col_idx]), temp_col_idx
                    ].min(),
                    self.dataset[
                        ~torch.isnan(self.dataset[:, temp_col_idx]), temp_col_idx
                    ].max(),
                ]
                for temp_col_idx in self.indep_attr_idx
            ]
        ).T
        object.__setattr__(self, "_bounds", _bounds)

    def evaluate(self, X):
        if X.ndim == 1:
            X = X.unsqueeze(0)

        meta_attr = self.dataset[:, self.indep_attr_idx]

        # Broadcast-compare each query row against all meta rows
        q = X.unsqueeze(1)  # (N, 1, H)
        m = meta_attr.unsqueeze(0)  # (1, M, H)

        matches = ((q == m) | (torch.isnan(q) & torch.isnan(m))).all(-1)

        # Ensure exactly one match per query
        if (matches.sum(dim=1) != 1).any():
            raise ValueError(f"No or multiple matches for rows.")

        # Get the row index of the match
        row_idx = matches.float().argmax(dim=1)  # (N,)

        perf = self.dataset[row_idx, self.perf_metric_idx]  # (N,)
        cost = self.dataset[row_idx, self.cost_metric_idx]  # (N,)
        return torch.unsqueeze(perf, dim=1), torch.unsqueeze(cost, dim=1)  # add .T

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return self.cat_attr_idx

    @property
    def log_trafo_dims(self) -> list:
        return self.log_trafo_attr_idx

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return self._indep_attr_names


class ShapIQGame(shapiq.Game):
    def __init__(
        self, seed: int = 42, store_times: bool = False, bbf: ShapIQGameBBF = None
    ):
        super().__init__(n_players=bbf.n_players)
        object.__setattr__(self, "bbf", bbf)

        # Load TabPFN and set seed
        # Use v2
        if not bbf.is_regression:
            self.model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2_5, random_state=seed, ignore_pretraining_limits=True
            )  # verbose= False

        else:
            self.model = TabPFNRegressor.create_default_for_version(
                ModelVersion.V2_5, random_state=seed, ignore_pretraining_limits=True
            )  # verbose= False

        if store_times:
            game_times = {}
            object.__setattr__(self, "game_times", game_times)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        outputs = np.zeros(coalitions.shape[0])
        times = np.zeros(coalitions.shape[0])

        for temp_coalition_idx in range(coalitions.shape[0]):
            if coalitions[temp_coalition_idx].astype(int).sum() == 0:
                # Use y_train mean as constant predictor baseline

                if hasattr(self, "game_times"):
                    start_time = time.time()

                if self.bbf.is_regression:
                    temp_constant_preds = np.array(
                        [self.bbf.y_train.mean()] * len(self.bbf.y_test)
                    )
                    outputs[temp_coalition_idx] = np.array(
                        mean_squared_error(self.bbf.y_test, temp_constant_preds)
                    )
                else:
                    temp_constant_preds = np.array(
                        [self.bbf.y_train.mode()[0]] * len(self.bbf.y_test)
                    )
                    outputs[temp_coalition_idx] = np.array(
                        accuracy_score(self.bbf.y_test, temp_constant_preds)
                    )

                if hasattr(self, "game_times"):
                    end_time = time.time()
                    temp_time = end_time - start_time

                # Only works for classification

            else:
                # Filter xtrain and ytrain based on coalition
                temp_X_train = copy.deepcopy(self.bbf.X_train).iloc[
                    :, coalitions[temp_coalition_idx].astype(bool)
                ]
                temp_X_test = copy.deepcopy(self.bbf.X_test).iloc[
                    :, coalitions[temp_coalition_idx].astype(bool)
                ]

                # #Disable warnings
                # import warnings
                # warnings.filterwarnings("ignore")

                if hasattr(self, "game_times"):
                    start_time = time.time()

                self.model.fit(temp_X_train, self.bbf.y_train)
                temp_preds = self.model.predict(temp_X_test)

                temp_perf = (
                    mean_squared_error(self.bbf.y_test, temp_preds)
                    if self.bbf.is_regression
                    else accuracy_score(self.bbf.y_test, temp_preds)
                )
                # temp_acc = accuracy_score(self.y_test, temp_preds)

                if hasattr(self, "game_times"):
                    end_time = time.time()
                    temp_time = end_time - start_time

                outputs[temp_coalition_idx] = temp_perf

            # Store times if required
            if hasattr(self, "game_times"):
                temp_index = transform_array_to_coalitions(
                    np.expand_dims(coalitions[temp_coalition_idx].astype(int), axis=0)
                )[0]
                self.game_times[temp_index] = temp_time

        # return outputs
        return np.array([outputs] if outputs.ndim == 0 else outputs)


# -----------------------------------------------------------------------------

# precompute different games for different seeds

ShapIQGameName = Literal[
    "tabpfn_37", "tabpfn_15", "tabpfn_470", "tabpfn_diabreg"  # "tabpfn_ch",
]


@dataclass(frozen=True)
class ShapIQGameBBF(BaseBlackboxFunction):
    # Change name accordingly - this is just for tabpfn
    name: ShapIQGameName = "tabpfn_37"
    seed: int = 42

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):
        parts = self.name.split("_")

        game_name = parts[0]
        object.__setattr__(
            self, "game_name", game_name
        )  # add this to diff between tabpfn or not later

        task_identifier = parts[1]
        if task_identifier == "ch" or task_identifier == "diabreg":
            object.__setattr__(self, "task_identifier", task_identifier)

        else:
            openml_id = int(parts[1])
            object.__setattr__(self, "task_identifier", openml_id)

        if task_identifier == "ch":
            data = sklearn.datasets.fetch_california_housing(
                as_frame=True, return_X_y=False
            )
            object.__setattr__(self, "is_regression", True)

        elif task_identifier == "diabreg":
            data = sklearn.datasets.load_diabetes(as_frame=True, return_X_y=False)
            object.__setattr__(self, "is_regression", True)

        else:
            data = sklearn.datasets.fetch_openml(
                data_id=self.task_identifier, as_frame=True, return_X_y=False
            )
            is_regression = (
                "Class" not in data.target_names and "class" not in data.target_names
            )
            # Manually overwrite for 470 (there should be a better way)
            if self.task_identifier == 470:
                is_regression = False

            object.__setattr__(self, "is_regression", is_regression)

        X = data.data
        y = data.target if not self.is_regression else data.target.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed
        )

        object.__setattr__(self, "X_train", X_train)
        object.__setattr__(self, "X_test", X_test)
        object.__setattr__(self, "y_train", y_train)
        object.__setattr__(self, "y_test", y_test)

        # Find out amount of players from openml dataset
        n_players = X.shape[1]
        object.__setattr__(self, "n_players", n_players)

        path_to_values = Path(
            f"data/shapiq_games/{self.game_name}/tid{self.task_identifier}/{self.game_name}_tid{self.task_identifier}_seed{self.seed}_values.npz"
        )
        path_to_times = Path(
            f"data/shapiq_games/{self.game_name}/tid{self.task_identifier}/{self.game_name}_tid{self.task_identifier}_seed{self.seed}_times.json"
        )

        # Set up Shapiq Game (with seed)
        shapiq_game = ShapIQGame(seed=self.seed, store_times=True, bbf=self)

        if path_to_values.exists():
            shapiq_game.load_values(path_to_values)
            # Load times
            with open(path_to_times, "r") as f:
                game_times = json.load(f)
            game_times = {
                safe_str_to_tuple(key): value for (key, value) in game_times.items()
            }
            object.__setattr__(shapiq_game, "game_times", game_times)

            # temp_game= shapiq.Game(n_players=16)
            # temp_game._load_npz_values(path=vit_path)

        else:
            shapiq_game.verbose = True  # to see progress
            shapiq_game.precompute()

            path_to_values.parent.mkdir(parents=True, exist_ok=True)
            path_to_times.parent.mkdir(parents=True, exist_ok=True)

            shapiq_game.save_values(path_to_values)
            with open(path_to_times, "w") as f:
                transformed_game_times = {
                    safe_tuple_to_str(key): value
                    for (key, value) in shapiq_game.game_times.items()
                }
                json.dump(transformed_game_times, f)

            print("wait")

        object.__setattr__(self, "dim", n_players)
        object.__setattr__(self, "shapiq_game", shapiq_game)
        object.__setattr__(self, "_bounds", None)

    def get_bounds_for_dim(self, dim: int = 0):
        return self._bounds[:, dim]

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return (
            torch.unsqueeze(
                torch.tensor(self.shapiq_game(np.array(X)), dtype=torch.float64), dim=1
            ),
            None,
        )
        # ---------------- required overrides -----------------------------

    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]


ShapIQPrecomputedGameName = Literal[
    "sentiment_14",  # SentimentAnalysis_Game
    "vit_9",  # ImageClassifier_Game
    "resnet_14",
    "vit_16",
    "dvacdt_10",
    "dvacgb_10",
    "dvacrf_10",
    "dvbsdt_10",
    "dvbsgb_10",
    "dvbsrf_10",
    "dvchdt_10",
    "dvchgb_10",
    "dvchrf_10",
    "fsacdt_14",
    "fsacgb_14",
    "fsacrf_14",
    "fsbsdt_12",
    "fsbsgb_12",
    "fsbsrf_12",
    "fschdt_8",
    "fschgb_8",
    "fschrf_8",
    # Feature selection retrain
]


@dataclass(frozen=True)
class ShapIQPrecomputedGameBBF(BaseBlackboxFunction):
    # Does not use ShapIQGame
    name: ShapIQPrecomputedGameName = "vit_9"
    seed: int = 0

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):
        parts = self.name.split("_")
        game_name = parts[0]
        n_players = int(parts[1])

        object.__setattr__(
            self, "game_name", game_name
        )  # add this to diff between tabpfn or not later
        object.__setattr__(self, "n_players", n_players)

        if game_name == "vit":
            file_name = (
                f"model_name={self.game_name}_{str(n_players)}_patches_{self.seed}"
            )

        elif game_name == "resnet":
            file_name = f"model_name=resnet_18_n_superpixel_resnet=14_{self.seed}"

        elif game_name == "sentiment":
            file_name = f"mask_strategy=mask_{self.seed}"
            # sentiment_14

        elif game_name.startswith("fs"):

            model_name_end = game_name[-2:]
            if model_name_end == "dt":
                model_name = "decision_tree"
            elif model_name_end == "gb":
                model_name = "gradient_boosting"
            elif model_name_end == "rf":
                model_name = "random_forest"

            file_name = f"model_name={model_name}_{self.seed}"

        elif game_name.startswith("dv"):
            model_name_end = game_name[-2:]
            if model_name_end == "dt":
                model_name = "decision_tree"
            elif model_name_end == "gb":
                model_name = "gradient_boosting"
            elif model_name_end == "rf":
                model_name = "random_forest"

            file_name = f"model_name={model_name}_player_sizes=increasing_n_players=10_{self.seed}"

        else:
            raise NotImplementedError(
                f"Game name {game_name} not implemented in ShapIQPrecomputedGameBBF"
            )

        path_to_values = Path(
            f"data/shapiq_games/{self.game_name}/{str(self.n_players)}/{file_name}.npz"
        )

        # Set up Shapiq Game (with seed)
        shapiq_game = shapiq.Game(n_players=self.n_players)
        shapiq_game._load_npz_values(path=path_to_values)

        object.__setattr__(self, "dim", n_players)
        object.__setattr__(self, "shapiq_game", shapiq_game)
        object.__setattr__(self, "_bounds", None)

    def get_bounds_for_dim(self, dim: int = 0):
        return self._bounds[:, dim]

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return (
            torch.unsqueeze(
                torch.tensor(self.shapiq_game(np.array(X)), dtype=torch.float64), dim=1
            ),
            None,
        )

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]


# add shapley dummy function
@dataclass(frozen=True)
class ShapleyDummyFunction(BaseBlackboxFunction):
    """A dummy black-box function for testing purposes in the Shapley value computations."""

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    dim: int = 3
    symmetric: bool = True

    def __post_init__(self):
        object.__setattr__(self, "_bounds", torch.tensor([[0.0, 1.0]] * self.dim))

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        if self.symmetric:
            return (torch.sum(X, dim=1, keepdim=True) ** 2, 
                    None)

        else:
            return (1 * X[:, 0:1] + 1 * X[:, 1:2] + 0.01 * X[:, 2:3] + 1 * X[:, 0:1] * X[:, 1:2], 
                    #4 * X[:, 0:1] + 1 * X[:, 1:2] + 0.01 * X[:, 2:3] + 2 * X[:, 0:1] * X[:, 1:2], 
                    None)


        #return (torch.sum(X, dim=1, keepdim=True), None)

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return "ShapleyDummyFunction"

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"Dim. {i}" for i in range(self.dim)]

ShapleyTreeGameName = Literal[
    "IndependentLinear60", "Corrgroups60", "CommunitiesAndCrime", "NHANESI"
]


@dataclass(frozen=True)
class ShapleyTreeGameBBF(BaseBlackboxFunction):
    name: ShapleyTreeGameName = "IndependentLinear60"
    seed: int = 42

    deterministic: bool = field(init=False, default=True)
    is_pseudo_expensive: bool = field(init=False, default=True)

    def __post_init__(self):
        # Initialize background dataset
        import shap

        if self.name == "NHANESI":
            dataset_X, dataset_y = shap.datasets.nhanesi()

        elif self.name == "CommunitiesAndCrime":
            dataset_X, dataset_y = shap.datasets.communitiesandcrime()

        elif self.name == "IndependentLinear60":
            dataset_X, dataset_y = shap.datasets.independentlinear60()

        elif self.name == "Corrgroups60":
            dataset_X, dataset_y = shap.datasets.corrgroups60()

        else:
            raise NotImplementedError(
                f"Game name {self.name} not implemented in ShapleyTreeGameBBF."
            )
            # For new games, ensure that they are regression datasets. Otherwise RandomForestRegressor must be replaced.

        dataset_y = pd.DataFrame(dataset_y, columns=["target"])

        dim = dataset_X.shape[1]

        # Set self.seed'th element as test instance and rest as background dataset
        assert self.seed < len(
            dataset_X
        ), f"Seed {self.seed} must be smaller than dataset size {len(dataset_X)}"

        X_train = dataset_X.drop(dataset_X.index[[self.seed]]).to_numpy()
        y_train = dataset_y.drop(dataset_y.index[[self.seed]]).to_numpy()

        # Define explanation instance
        x_explain = dataset_X.iloc[[self.seed]].to_numpy()
        y_explain = dataset_y.iloc[[self.seed]].to_numpy()

        # Train tree model. Some datasets can yield trees that are too deep for the
        # Vandermonde inversion inside shapiq's TreeSHAPIQ implementation.
        from sklearn.ensemble import RandomForestRegressor

        from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

        shapiq_game = None
        tree_model = None
        retry_max_depths = (32, 16, 8, 4)

        for max_depth in retry_max_depths:
            tree_model = RandomForestRegressor(
                n_estimators=10,
                random_state=self.seed,
                max_depth=max_depth,
            )
            tree_model.fit(X_train, y_train.ravel())

            try:
                shapiq_game = TreeSHAPIQXAI(
                    x=x_explain, tree_model=tree_model, normalize=False
                )
                if max_depth != retry_max_depths[0]:
                    log.warning(
                        "TreeSHAPIQXAI required fallback max_depth=%s for %s (seed=%s).",
                        max_depth,
                        self.name,
                        self.seed,
                    )
                break
            except np.linalg.LinAlgError:
                if max_depth == retry_max_depths[-1]:
                    raise

                log.warning(
                    "TreeSHAPIQXAI failed for max_depth=%s on %s (seed=%s); retrying with a smaller depth.",
                    max_depth,
                    self.name,
                    self.seed,
                )

        assert shapiq_game is not None
        assert tree_model is not None

        object.__setattr__(self, "shapiq_game", shapiq_game)
        object.__setattr__(self, "_bounds", None)
        # shapiq Game with SV ground truth and value function

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "n_players", dim)

    def get_bounds_for_dim(self, dim: int = 0):
        return self._bounds[:, dim]

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return (
            torch.unsqueeze(
                torch.tensor(
                    self.shapiq_game.value_function(np.array(X)), dtype=torch.float64
                ),
                dim=1,
            ),
            None,
        )

    # ---------------- required overrides -----------------------------
    @property
    def plot_name(self) -> str:
        return self.name

    @property
    def cat_dims(self) -> list:
        return []

    @property
    def log_trafo_dims(self) -> list:
        return []

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @property
    def indep_attr_names(self) -> list:
        return [f"""Dim. {str(i)}""" for i in range(self.dim)]

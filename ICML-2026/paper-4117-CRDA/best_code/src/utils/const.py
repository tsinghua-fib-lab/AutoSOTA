"""
Constants and configuration setting in the project.
Default hyperparameters and model registry mappings are defined here.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple, Type
import numpy as np
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS

# -----------------------------------------------------------------------------
# KernelRidge – defaults hyper-parameters
# -----------------------------------------------------------------------------
KR_PARAMS: Dict[str, Any] = {
    "kernel": "rbf",
    "alpha": 1.0,
}

KR_PARAM_SPACE: Dict[str, Any] = {
    "kernel": ["rbf", "linear", "poly", "sigmoid"],
    "alpha": loguniform(1e-3, 1e3),
}

KR_FIT_KWARGS: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# LinearRegression – defaults hyper-parameters
# -----------------------------------------------------------------------------
LINREG_PARAMS: Dict[str, Any] = {
    "fit_intercept": True,
    "copy_X": True,
    "n_jobs": None,
    "positive": False,
}

LINREG_PARAM_SPACE: Dict[str, Any] = {
    "fit_intercept": [True, False],
    "positive": [False, True],
}

LINREG_FIT_KWARGS: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# MLPRegressor default hyper‑parameters
# -----------------------------------------------------------------------------
MLP_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes": (64, 32, 16),
    "activation": "relu",
    "solver": "adam",
    "max_iter": 1000,
    "batch_size": 32,
    "learning_rate": "adaptive",
    "learning_rate_init": 0.001,
    "tol": 1e-4,
    "n_iter_no_change": 20,
    "alpha": 1e-4,
    "early_stopping": True,
    "validation_fraction": 0.1,
}

MLP_PARAM_SPACE = {
    "hidden_layer_sizes": [
        (128, 64, 32),  
        (128, 64),      
        (64, 32),       
        (64,),          
    ],
    "alpha": loguniform(1e-5, 1e-3),       # L2
    "learning_rate_init": loguniform(1e-3, 1e-2),
    "tol": loguniform(1e-5, 1e-4),
}

MLP_FIT_KWARGS = {}

# -----------------------------------------------------------------------------
# XGBRegressor default hyper‑parameters
# -----------------------------------------------------------------------------
XGB_PARAMS: Dict[str, Any] = {
    "tree_method": "hist",
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}

XGB_PARAM_SPACE: Dict[str, Any] = {
    "learning_rate":  np.logspace(-3, -1, 10),
    "max_depth":      [3, 4, 6],
    "min_child_weight":[1, 5],
    "subsample":      [0.7, 1.0],
    "colsample_bytree":[0.7, 1.0],
    "reg_lambda":     np.logspace(-3, 1, 6),
}

XGB_FIT_KWARGS = {
    "verbose": False,
}

# -----------------------------------------------------------------------------
# CatBoostRegressor default hyper‑parameters
# -----------------------------------------------------------------------------
CATBOOST_PARAMS: Dict[str, Any] = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 3,
    "l2_leaf_reg": 3.0,
    "border_count": 128,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "verbose": False,
    "allow_writing_files": False,
}

CATBOOST_PARAM_SPACE: Dict[str, Any] = {
    "learning_rate": np.logspace(-3, -1, 10),
    "depth": [3, 6, 9],
    "l2_leaf_reg": np.logspace(-1, 1, 6),
    "bagging_temperature": [0.0, 0.5, 1.0],
    "random_strength": [0.5, 1.0, 2.0],
    "border_count": [32, 128, 254],
}

CATBOOST_FIT_KWARGS = {
    "verbose": False,
}

# -----------------------------------------------------------------------------
# Registry - maps a human‑readable name to (estimator class, params) pair.
# -----------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, Tuple[Type, Dict[str, Any]]] = {
    "linreg": (LinearRegression, LINREG_PARAMS, LINREG_PARAM_SPACE, LINREG_FIT_KWARGS),
    "mlp": (MLPRegressor, MLP_PARAMS, MLP_PARAM_SPACE, MLP_FIT_KWARGS),
    "xgboost": (XGBRegressor, XGB_PARAMS, XGB_PARAM_SPACE, XGB_FIT_KWARGS),
    "catboost": (CatBoostRegressor, CATBOOST_PARAMS, CATBOOST_PARAM_SPACE, CATBOOST_FIT_KWARGS),
    "krr": (KernelRidge, KR_PARAMS, KR_PARAM_SPACE, KR_FIT_KWARGS),
}

__all__ = [
    "LINREG_PARAMS",
    "MLP_PARAMS",
    "XGB_PARAMS",
    "CATBOOST_PARAMS",
    "KR_PARAMS",
    "MODEL_REGISTRY",
    "MLP_PARAM_SPACE",
    "XGB_PARAM_SPACE",
    "CATBOOST_PARAM_SPACE",
    "LINREG_PARAM_SPACE",
    "KR_PARAM_SPACE",
]

"""Functions to validate input arguments of main class XGBTreeHFD."""


import numpy as np
import xgboost as xgb
from numpy.random import default_rng


def check_xgb_model_type(xgb_model: xgb.sklearn.XGBModel) -> None:
    """Check xgb_model type.

    Check that xgb_model is a xgboost model for regression or binary
    classification, of type xgboost.sklearn.XGBRegressor
    or xgboost.sklearn.XGBClassifier, built with the scikit-learn
    interface of the xgboost package.
    """
    type_check = isinstance(xgb_model, (xgb.sklearn.XGBRegressor,
                                         xgb.sklearn.XGBClassifier))
    if type_check:
        objective = str(xgb_model.objective).split(":")[0]
        type_check = objective in ("reg", "binary")
    if not type_check:
        error_msg = ("xgb_model must be a xgboost model for regression or "
                     "binary classification, respectively of type "
                     "xgboost.sklearn.XGBRegressor "
                     "or xgboost.sklearn.XGBClassifier, built with the "
                     "scikit-learn interface of the xgboost package.")
        raise ValueError(error_msg)
    if xgb_model.enable_categorical:
        error_msg = ("treehfd does not support categorical variables. "
                     "One-hot encoding should be used instead to fit xgboost.")
        raise ValueError(error_msg)


def check_xgb_model_learner(config: dict) -> None:
    """Check that xgb_model is built with tree learners."""
    booster_type = config["learner"]["learner_train_param"]["booster"]
    type_check = booster_type == "gbtree"
    if not type_check:
        error_msg = ("xgb_model must be a xgboost model built with the tree "
                     "learner gbtree.")
        raise ValueError(error_msg)


def check_xgb_params(max_depth: int, n_estimators: int,
                     num_parallel_tree: int, num_target: int) -> None:
    """Check values of xgboost model parameters."""
    depth_check = max_depth > 0
    n_estimators_check = n_estimators > 0
    num_parallel_tree_check = num_parallel_tree == 1
    num_target_check = num_target == 1
    error_msg = ""
    if not n_estimators_check:
        error_msg = ("n_estimators of xgboost model (number of trees) must "
                     "be a strictly positive integer.")
        raise ValueError(error_msg)
    if not depth_check:
        error_msg = "xgboost model must have a stricly positive tree depth."
        raise ValueError(error_msg)
    if not num_parallel_tree_check:
        error_msg = "num_parallel_tree must set to 1."
        raise ValueError(error_msg)
    if not num_target_check:
        error_msg = "num_target must be 1."
        raise ValueError(error_msg)


def check_data(X: np.ndarray, name: str, num_feature: int) -> None:
    """Check that input data X is a numpy array."""
    type_check = isinstance(X, np.ndarray)
    if type_check:
        dim_array = 2
        type_check = len(X.shape) == dim_array
        if type_check:
            type_check = X.shape[0] > 0 and X.shape[1] == num_feature
    if not type_check:
        error_msg = (f"{name} must be a non-empty numpy array containing the "
                     "input data, and the number of columns should match the "
                     "expected number of features in the xgboost model, that "
                     f"is {num_feature}.")
        raise ValueError(error_msg)


def check_interaction_order(interaction_order: int) -> None:
    """Check that interaction_order is 1 or 2."""
    type_check = isinstance(interaction_order, int)
    if type_check:
        type_check = interaction_order in (1, 2)
    if not type_check:
        error_msg = ("interaction_order must be 1 to fit only main effects, "
                     "or 2 to also include second-order interactions.")
        raise ValueError(error_msg)


def check_interaction_list(interaction_list: np.ndarray | None) -> None:
    """Check that interaction_list is a numpy array of integer pairs."""
    type_check = interaction_list is None or isinstance(interaction_list,
                                                        np.ndarray)
    if isinstance(interaction_list, np.ndarray):
        dim_ref = 2
        type_check = len(interaction_list.shape) == dim_ref
        if type_check:
            type_check = (interaction_list.shape[1] == dim_ref
                          and interaction_list.dtype == int)
    if not type_check:
        error_msg = ("interaction_list must be a numpy array, where each row "
                     "contains a pair of integers for the indices of a "
                     "variable  interaction.")
        raise ValueError(error_msg)


def check_depth_variable(depth_variable: int | None) -> None:
    """Check that depth_variable is None or positive integer."""
    type_check = depth_variable is None or isinstance(depth_variable, int)
    if isinstance(depth_variable, int):
        type_check = depth_variable > 0
    if not type_check:
        error_msg = ("depth_variable must be None or positive integer.")
        raise ValueError(error_msg)


def sample_data(nsample: int) -> tuple:
    """Sample data from an analytical model.

    Data is sampled from a specific example with Gaussian inputs and a
    regression function defined from a mix of linear an sinusoidal
    functions with second-order interactions.
    """
    p = 6
    mu = np.zeros(p)
    rho = 0.5
    cov = np.full((p, p), rho)
    np.fill_diagonal(cov, np.ones(p))
    X = default_rng().multivariate_normal(mean=mu, cov=cov, size=nsample)
    X = np.round(X, decimals=2)
    y = np.sin(2*np.pi*X[:, 0]) + X[:, 0] * X[:, 1]  + X[:, 2] * X[:, 3]
    y += default_rng().normal(loc=0.0, scale=0.5, size=nsample)
    y = np.round(y, decimals=2)
    return(X, y)


def eta_main(x: np.ndarray, rho: float) -> np.ndarray:
    """Compute main effect for the analytical example."""
    return(rho/(1 + rho**2)*(x**2 - 1))


def eta_order2(x: np.ndarray, z: np.ndarray, rho: float) -> np.ndarray:
    """Compute interactions for the analytical example."""
    return(rho*(1 - rho**2)/(1 + rho**2) - rho/(1 + rho**2)*(x**2 + z**2)
           + x*z)

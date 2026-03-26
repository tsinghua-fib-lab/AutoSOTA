"""Tests for validation functions of validation.py."""


import json
from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from treehfd.validation import (
    check_data,
    check_depth_variable,
    check_interaction_list,
    check_interaction_order,
    check_xgb_model_learner,
    check_xgb_model_type,
    check_xgb_params,
    eta_main,
    eta_order2,
    sample_data,
)

TESTS_DIR = Path(__file__).parent.resolve()


def test_check_xgb_model_type() -> None:
    """Test check_xgb_model_type function."""
    # Check fail.
    with pytest.raises(ValueError, match="xgb_model must be a xgboost model"):
        check_xgb_model_type("string")
    xgb_model = xgb.XGBRegressor(enable_categorical=True)
    with pytest.raises(ValueError, match="One-hot encoding should be used"):
        check_xgb_model_type(xgb_model)

    # Check pass.
    xgb_model = xgb.XGBRegressor()
    assert check_xgb_model_type(xgb_model) is None


def test_check_xgb_model_learner() -> None:
    """Test check_xgb_model_learner function."""
    np.random.default_rng(21)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed21.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n10_seed21.csv",
                      delimiter=",")

    # Check fail.
    xgb_model = xgb.XGBRegressor(booster="gblinear")
    xgb_model = xgb_model.fit(X, y)
    config = json.loads(xgb_model.get_booster().save_config())
    with pytest.raises(ValueError, match="xgb_model must be a xgboost model"):
        check_xgb_model_learner(config)

    # Check pass.
    xgb_model = xgb.XGBRegressor(booster="gbtree")
    xgb_model = xgb_model.fit(X, y)
    config = json.loads(xgb_model.get_booster().save_config())
    assert check_xgb_model_learner(config) is None


def test_check_xgb_params() -> None:
    """Test check_xgb_params function."""
    # Check fail for depth.
    max_depth = 0
    n_estimators = 100
    num_parallel_tree = 1
    num_target = 1
    with pytest.raises(ValueError, match="stricly positive tree depth"):
        check_xgb_params(max_depth, n_estimators, num_parallel_tree, num_target)

    # Check fail for n_estimators.
    max_depth = 3
    n_estimators = 0
    with pytest.raises(ValueError, match="n_estimators of xgboost model"):
        check_xgb_params(max_depth, n_estimators, num_parallel_tree, num_target)

    # Check fail for num_parallel_tree.
    n_estimators = 100
    num_parallel_tree = 2
    with pytest.raises(ValueError, match="num_parallel_tree must set to 1"):
        check_xgb_params(max_depth, n_estimators, num_parallel_tree, num_target)

    # Check fail for num_target.
    num_parallel_tree = 1
    num_target = 3
    with pytest.raises(ValueError, match="num_target must be 1"):
        check_xgb_params(max_depth, n_estimators, num_parallel_tree, num_target)

    # Check pass.
    max_depth = 3
    n_estimators = 100
    num_parallel_tree = 1
    num_target = 1
    assert check_xgb_params(max_depth, n_estimators, num_parallel_tree,
                            num_target) is None


def test_check_data() -> None:
    """Test check_data function."""
    # Check fail.
    with pytest.raises(ValueError, match="X must be a non-empty numpy array"):
        check_data(np.empty((0, 6)), "X", num_feature=6)
    with pytest.raises(ValueError, match="X must be a non-empty numpy array"):
        check_data(np.ones((10, 5)), "X", num_feature=6)

    # Check pass.
    np.random.default_rng(21)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed21.csv",
                      delimiter=",")
    assert check_data(X, "X_new", num_feature=6) is None


def test_check_interaction_order() -> None:
    """Test check_interaction_order function."""
    # Check fail.
    with pytest.raises(ValueError, match="interaction_order must be 1 to fit"):
        check_interaction_order(np.ones((2, 3)))
    with pytest.raises(ValueError, match="interaction_order must be 1 to fit"):
        check_interaction_order(0)

    # Check pass.
    assert check_interaction_order(1) is None
    assert check_interaction_order(2) is None


def test_check_interaction_list() -> None:
    """Test check_interaction_list function."""
    # Check fail.
    with pytest.raises(ValueError, match="interaction_list must be a numpy"):
        check_interaction_list(np.ones(2))
    with pytest.raises(ValueError, match="interaction_list must be a numpy"):
        check_interaction_list("string")

    # Check pass.
    assert check_interaction_list(None) is None
    assert check_interaction_list(np.array([[0, 1], [2, 3]])) is None


def test_check_depth_variable() -> None:
    """Test check_depth_variable function."""
    # Check fail.
    for depth_variable in ["ezar", np.ones(10), -3]:
        with pytest.raises(ValueError, match="depth_variable must be None or"):
            check_depth_variable(depth_variable)

    # Check pass.
    for depth_variable in [None, 1, 13]:
        assert check_depth_variable(depth_variable) is None


def test_sample_data() -> None:
    """Test sample_data fucntion."""
    nsample = 57
    X, y = sample_data(nsample)
    assert X.shape == (nsample, 6)
    assert y.shape[0] == nsample


def test_eta_main() -> None:
    """Test eta_main function."""
    nsample = 7
    x = np.full(nsample, 2)
    y = eta_main(x, rho=0.7)
    assert set(np.round(y, 3)) == {1.409}


def test_eta_order2() -> None:
    """Test eta_main function."""
    nsample = 7
    x, z = np.full(nsample, 2), np.full(nsample, 11)
    y = eta_order2(x, z, rho=0.3)
    assert set(np.round(y, 3)) == {-12.153}

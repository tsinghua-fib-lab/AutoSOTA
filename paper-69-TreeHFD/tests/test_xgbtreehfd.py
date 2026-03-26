"""Tests for XGBTreeHFD class defined in ensemble.py."""


from pathlib import Path

import numpy as np
import pytest
import xgboost as xgb

from tests.utils import eta_main, eta_order2
from treehfd import XGBTreeHFD
from treehfd.cartesian_partition import CartesianTreePartition
from treehfd.tree_structure import (
    extract_variable_paths,
    extract_variables,
)

TESTS_DIR = Path(__file__).parent.resolve()


def test_xgbtreehfd() -> None:
    """Test initialization of XGBTreeHFD class."""
    np.random.default_rng(11)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed11.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed11.csv",
                      delimiter=",")
    xgb_model = xgb.XGBRegressor(n_estimators=3, max_depth=2)
    xgb_model = xgb_model.fit(X, y)

    treehfd_model = XGBTreeHFD(xgb_model)
    assert np.round(treehfd_model.base_score, decimals=2) == 1.11
    assert isinstance(treehfd_model.config, dict)
    assert treehfd_model.eta0 == 0.0
    assert callable(treehfd_model.fit)
    assert np.array_equal(treehfd_model.interaction_list, np.empty((0, 0)))
    assert treehfd_model.max_depth == 2
    assert treehfd_model.n_estimators == 3
    assert treehfd_model.num_feature == 6
    assert callable(treehfd_model.predict)
    assert not treehfd_model.treehfd_list
    assert isinstance(treehfd_model.xgb_model, xgb.XGBRegressor)
    assert treehfd_model.xgb_table.equals(
                xgb_model.get_booster().trees_to_dataframe())


def test_xgbtreehfd_failures() -> None:
    """Test failed initializations of XGBTreeHFD class."""
    with pytest.raises(ValueError, match="xgb_model must be a xgboost model"):
        XGBTreeHFD("xgb_model")
    xgb_model = xgb.XGBRegressor()
    with pytest.raises(ValueError,
                       match="fit or load xgboost model first"):
        XGBTreeHFD(xgb_model)
    np.random.default_rng(11)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed11.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed11.csv",
                      delimiter=",")
    xgb_model = xgb.XGBRegressor(n_estimators=3, booster="gblinear")
    xgb_model = xgb_model.fit(X, y)
    with pytest.raises(ValueError,
                       match="model built with the tree learner gbtree"):
        XGBTreeHFD(xgb_model)
    xgb_model = xgb.XGBRegressor(n_estimators=0)
    xgb_model = xgb_model.fit(X, y)
    with pytest.raises(ValueError, match="must be a strictly positive integer"):
        XGBTreeHFD(xgb_model)


def test_xgbtreehfd_fit() -> None:
    """Test fit method of XGBTreeHFD class."""
    # Test fit with sample of 1 point.
    n, p = (1, 11)
    X = np.ones(p).reshape(n, p)
    y = np.ones(n)
    xgb_model = xgb.XGBRegressor(n_estimators=3, max_depth=2)
    xgb_model = xgb_model.fit(X, y)
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)
    assert treehfd_model.treehfd_list[0].hfd_coeffs.size == 0

    # Test fit with sample of 100 points.
    np.random.default_rng(11)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed11.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed11.csv",
                      delimiter=",")
    xgb_model = xgb.XGBRegressor(n_estimators=3, max_depth=2)
    xgb_model = xgb_model.fit(X, y)
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)
    assert np.round(treehfd_model.eta0, decimals=2) == 1.08
    assert np.array_equal(treehfd_model.interaction_list,
                          np.array([[0, 1], [0, 2], [1, 2]]))
    assert isinstance(treehfd_model.treehfd_list, list)
    assert len(treehfd_model.treehfd_list) == treehfd_model.n_estimators
    assert np.array_equal(np.round(treehfd_model.treehfd_list[0].eta0, 2),
                          0.35)
    variable_paths = extract_variable_paths(
        treehfd_model.treehfd_list[0].tree_structure, depth_variable=2)
    main_variables = extract_variables(variable_paths)
    cartesian_partition = CartesianTreePartition(main_variables)
    _ = cartesian_partition.compute_cartesian_partition(X,
            treehfd_model.treehfd_list[0].tree_structure,
            treehfd_model.treehfd_list[0].interaction_list)
    tree_partition = treehfd_model.treehfd_list[0].cartesian_partition
    assert np.array_equal(tree_partition.main_variables,
                          cartesian_partition.main_variables)
    assert np.array_equal(tree_partition.partition_index,
                          cartesian_partition.partition_index)
    assert np.array_equal(tree_partition.split_list,
                          cartesian_partition.split_list)
    assert np.array_equal(tree_partition.cell_list[0],
                          cartesian_partition.cell_list[0])
    assert np.array_equal(tree_partition.cell_list[1],
                          cartesian_partition.cell_list[1])
    assert np.array_equal(tree_partition.counts_list[0],
                          tree_partition.counts_list[0])
    assert np.array_equal(tree_partition.counts_list[1],
                          tree_partition.counts_list[1])
    assert np.array_equal(np.round(treehfd_model.treehfd_list[0].hfd_coeffs,
                                   decimals=2),
                          np.array([-0.02, 0.02, 1.01, -0.05, -0.05, 1.17,
                                    0.02, -1., -0.02, 0.33, 0., 0., 0.]))
    assert np.array_equal(treehfd_model.treehfd_list[0].interaction_list,
                          [[0, 2], [1, 2]])
    assert treehfd_model.treehfd_list[0].interaction_order == 2
    assert np.array_equal(np.round(treehfd_model.treehfd_list[1].hfd_coeffs,
                                   decimals=2),
                          np.array([-0.07, 0.9, 0.8, -0.07, -0.01, 0., 0.03,
                                    -0.03, -1.2, 0.17]))
    assert np.array_equal(np.round(treehfd_model.treehfd_list[2].hfd_coeffs,
                                   decimals=2),
                          np.array([0.18, -0.18, 0.49, -0.04, 0.06, 0.03,
                                    -0.07, 0.04, -0.05, -0.68, 0.23]))

    # Test wrong inputs.
    with pytest.raises(ValueError, match="the number of columns should match"):
        treehfd_model.fit(np.zeros((100, 4)))
    with pytest.raises(ValueError, match="interaction_order must be 1 to fit"):
        treehfd_model.fit(X, interaction_order=3)
    with pytest.raises(ValueError, match="must be None or positive integer"):
        treehfd_model.fit(X, depth_variable=-5)
    treehfd_model.fit(X, depth_variable=2)
    assert treehfd_model.depth_variable == 2


def test_xgbtreehfd_predict() -> None:
    """Test predict method of XGBTreeHFD class."""
    np.random.default_rng(11)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed11.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed11.csv",
                      delimiter=",")
    xgb_model = xgb.XGBRegressor(n_estimators=3, max_depth=2)
    xgb_model = xgb_model.fit(X, y)
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)
    X_new = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_Xnew_n3_seed11.csv",
                          delimiter=",")

    y_main, y_order2 = treehfd_model.predict(X_new)
    assert np.array_equal(np.round(y_main, decimals=2),
                          np.array([[0.44, -0.01, -0.05, 0., 0., 0.],
                                    [0.09, -0.01, -0.05, 0., 0., 0.],
                                    [0.09, -0.01, -0.05, 0., 0., 0.]]))
    assert np.array_equal(np.round(y_order2, decimals=2),
                          np.array([[ 0.2, -0.02, 0.],
                                    [-0.1, 0.02, 0.],
                                    [-0.1, 0.02, 0.]]))

    # Test failures.
    treehfd_model = XGBTreeHFD(xgb_model)
    with pytest.raises(ValueError, match="Fit TreeHFD before"):
        y_main, y_order2 = treehfd_model.predict(X_new)
    treehfd_model.fit(X)
    with pytest.raises(ValueError, match="the number of columns should match"):
        treehfd_model.predict(np.zeros((100, 4)))


def test_xgbtreehfd_accuracy() -> None:
    """Test accuracy of XGBTreeHFD."""
    DIM = 6
    RHO = 0.5
    np.random.default_rng(61)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n1000_seed61.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n1000_seed61.csv",
                      delimiter=",")

    # Test regression.
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3)
    xgb_model = xgb_model.fit(X, y)
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)
    np.random.default_rng(11)
    X_new = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed11.csv",
                          delimiter=",")
    y_main, y_order2 = treehfd_model.predict(X_new)
    hfd_pred = (treehfd_model.eta0 + np.sum(y_main, axis=1)
                + np.sum(y_order2, axis=1))
    xgb_pred = xgb_model.predict(X_new)
    mse_resid_reg = np.mean((xgb_pred - hfd_pred)**2)/np.var(xgb_pred)
    assert mse_resid_reg < 0.02
    y_exact = np.zeros((100, DIM))
    y_exact[:, 0] = np.sin(2*np.pi*X_new[:, 0]) + eta_main(X_new[:, 0], RHO)
    for j in range(1, DIM - 2):
        y_exact[:, j] = eta_main(X_new[:, j], RHO)
    mse_main = np.mean((y_exact - y_main)**2, axis=0)
    y_exact_01 = eta_order2(X_new[:, 0], X_new[:, 1], RHO)
    y_exact_23 = eta_order2(X_new[:, 2], X_new[:, 3], RHO)
    mse_interaction = [np.mean((y_exact_01 - np.squeeze(y_order2[:, 0]))**2),
                       np.mean((y_exact_23 - np.squeeze(y_order2[:, 9]))**2)]
    mse_null = np.mean(np.delete(y_order2, [0, 9], axis=1)**2, axis=0)
    cumulated_mse = (np.sum(mse_main) + np.sum(mse_interaction)
                     + np.sum(mse_null))
    assert cumulated_mse < 0.4

    # Test binary classification
    prob = 1/(1 + np.exp(-(y - 1.0)))
    y = [int(x) for x in prob > 0.5]
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                  n_estimators=100, max_depth=3)
    xgb_model = xgb_model.fit(X, y)
    treehfd_model = XGBTreeHFD(xgb_model)
    treehfd_model.fit(X)
    y_main, y_order2 = treehfd_model.predict(X_new)
    hfd_pred = (treehfd_model.eta0 + np.sum(y_main, axis=1)
                + np.sum(y_order2, axis=1))
    xgb_pred = xgb_model.predict(X_new, output_margin=True)
    mse_resid_classif = np.mean((xgb_pred - hfd_pred)**2)/np.var(xgb_pred)
    assert mse_resid_classif < 0.02

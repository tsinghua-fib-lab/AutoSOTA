"""Tests for tree structure extraction functions of tree_structure.py."""


import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from tests.utils import load_tree_table
from treehfd.tree_structure import (
    extract_child_ids,
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
    get_params,
    transform_variables,
)

TESTS_DIR = Path(__file__).parent.resolve()


def test_get_params() -> None:
    """Test get_params function."""
    np.random.default_rng(21)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed21.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n10_seed21.csv",
                      delimiter=",")
    xgb_model = xgb.XGBRegressor()
    xgb_model = xgb_model.fit(X, y)
    config = json.loads(xgb_model.get_booster().save_config())
    (max_depth, n_estimators, base_score, num_feature,
     num_parallel_tree, num_target) = get_params(config)
    assert max_depth == 6
    assert n_estimators == 100
    assert np.round(base_score, decimals=2) == 0.89
    assert num_feature == 6
    assert num_parallel_tree == 1
    assert num_target == 1


def test_extract_child_ids() -> None:
    """Test extract_child_ids function."""
    assert extract_child_ids(str(np.nan)) == 0
    assert extract_child_ids("0-1") == 1


def test_transform_variables() -> None:
    """Test transform_variables function."""
    assert transform_variables("Leaf") == -1
    assert transform_variables("f3") == 3
    assert transform_variables("f728") == 728


def test_extract_tree_structure() -> None:
    """Test extract_tree_structure function."""
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    assert np.array_equal(tree_structure[0],
                          np.array([ 3,  2,  0, -1,  0,  1,  1, -1, -1, -1,
                                    -1, -1, -1]))
    assert np.array_equal(tree_structure[1],
                          np.array([[1, 3, 5, 0, 7, 9, 11, 0, 0, 0, 0, 0, 0],
                                    [ 2, 4, 6, 0, 8, 10, 12, 0, 0, 0, 0, 0, 0],
                                   ]))
    assert np.array_equal(tree_structure[2][[0, 1, 2, 4, 5, 6]],
                          np.array([-1.74, -1.51, 0.99, -0.78, -1.61, 1.69]))
    assert np.all(np.isnan(tree_structure[2][[3, 7, 8, 9, 10, 11, 12]]))


def test_extract_variable_paths() -> None:
    """Test extract_variable_paths function."""
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    variable_paths = extract_variable_paths(tree_structure, depth_variable=0)
    assert variable_paths == []
    variable_paths = extract_variable_paths(tree_structure, depth_variable=1)
    assert variable_paths == [[3]]
    variable_paths = extract_variable_paths(tree_structure, depth_variable=2)
    assert variable_paths == [[2, 3], [0, 3]]
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    assert variable_paths == [[0, 2, 3], [0, 1, 3], [0, 1, 3]]
    variable_paths = extract_variable_paths(tree_structure, depth_variable=7)
    assert variable_paths == [[0, 2, 3], [0, 1, 3], [0, 1, 3]]


def test_extract_variables() -> None:
    """Test extract_variables function."""
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    variable_paths = extract_variable_paths(tree_structure, depth_variable=1)
    variables = extract_variables(variable_paths)
    assert np.array_equal(variables, np.array([3]))
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    variables = extract_variables(variable_paths)
    assert np.array_equal(variables, np.array([0, 1, 2, 3]))
    variable_paths = extract_variable_paths(tree_structure, depth_variable=6)
    variables = extract_variables(variable_paths)
    assert np.array_equal(variables, np.array([0, 1, 2, 3]))


def test_extract_interactions() -> None:
    """Test extract_interactions function."""
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    interactions_all = [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]]
    variable_paths = extract_variable_paths(tree_structure, depth_variable=1)
    interactions = extract_interactions(variable_paths)
    assert np.all(interactions == [])
    variable_paths = extract_variable_paths(tree_structure, depth_variable=2)
    interactions = extract_interactions(variable_paths)
    assert np.all(interactions == [[0, 3], [2, 3]])
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    interactions = extract_interactions(variable_paths)
    assert np.all(interactions == interactions_all)
    variable_paths = extract_variable_paths(tree_structure, depth_variable=6)
    interactions = extract_interactions(variable_paths)
    assert np.all(interactions == interactions_all)

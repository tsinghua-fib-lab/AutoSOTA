"""Tests for TreeHFD class defined in tree.py."""


from pathlib import Path

import numpy as np

from tests.utils import load_tree_table
from treehfd.tree import TreeHFD

TESTS_DIR = Path(__file__).parent.resolve()


def test_treehfd() -> None:
    """Test initialization of TreeHFD class."""
    tree_table = load_tree_table()
    tree = TreeHFD(tree_table, interaction_order=2, interaction_list=None,
                   depth_variable=6)
    assert np.array_equal(tree.tree_structure[0],
                          np.array([3, 2, 0, -1, 0, 1, 1, -1, -1, -1, -1, -1,
                                    -1]))
    assert np.array_equal(tree.tree_structure[1],
                          np.array([[1, 3, 5, 0, 7, 9, 11, 0, 0, 0, 0, 0, 0],
                                    [2, 4, 6, 0, 8, 10, 12, 0, 0, 0, 0, 0, 0]],
                                   ))
    assert np.array_equal(tree.tree_structure[2][[0, 1, 2, 4, 5, 6]],
                          np.array([-1.74, -1.51, 0.99, -0.78, -1.61,  1.69]))
    assert tree.interaction_order == 2
    assert np.array_equal(tree.interaction_list,
                          [[0, 1], [0, 2], [0, 3], [1, 3], [2, 3]])
    assert tree.eta0 == 0.0
    assert np.array_equal(tree.cartesian_partition.main_variables,
                          np.array([0, 1, 2, 3]))
    assert np.array_equal(tree.cartesian_partition.partition_index,
                          np.empty(0))
    assert tree.cartesian_partition.split_list == []
    assert tree.cartesian_partition.cell_list == []
    assert tree.cartesian_partition.counts_list == []
    assert np.array_equal(tree.hfd_coeffs, np.empty(0))

    tree = TreeHFD(tree_table, interaction_order=2,
                   interaction_list=np.array([[0, 1], [2, 3]]),
                   depth_variable=6)
    assert np.array_equal(tree.interaction_list, [[0, 1], [2, 3]])


def test_treehfd_fit() -> None:
    """Test fit method of TreeHFD class."""
    np.random.default_rng(41)
    tree_table = load_tree_table()
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed41.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed41.csv",
                      delimiter=",")
    tree = TreeHFD(tree_table, interaction_order=2, interaction_list=None,
                   depth_variable=6)

    tree.fit(X, y)
    hfd_coeffs = np.round(tree.hfd_coeffs, decimals=2)
    assert np.array_equal(hfd_coeffs,
                          np.array([0.44, -0.5 , 1.33, 1.87, -0.34, 2.56, 0.23,
                                    -0.08, 3.81, -0.4, 1.18, -0.22, -1.22,
                                    -0.07, 3.14, 0.61, -2.13, 1.64, -0.2,
                                    -1.68, 0.03, -0.02, -2.63, 0.25, 0.85,
                                    -0.11, -0.02, -2.42, 0.46, 0.32, -0.05,
                                    -0.02, -0.02, -0.02, -0.02]))


def test_treehfd_predict() -> None:
    """Test predict method of TreeHFD class."""
    np.random.default_rng(51)
    tree_table = load_tree_table()
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n100_seed51.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n100_seed51.csv",
                      delimiter=",")
    tree = TreeHFD(tree_table, interaction_order=2, interaction_list=None,
                   depth_variable=6)
    tree.fit(X, y)
    X_new = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_Xnew_n3_seed51.csv",
                          delimiter=",")

    y_main, y_order2 = tree.predict(X_new)
    assert np.array_equal(np.round(y_main, decimals=2),
                          np.array([[-0.57, -0.07,  2.22, -0.16],
                                    [ 2.57, -0.07,  0.01, -0.16],
                                    [-0.09, -0.07,  0.01, -0.16]]))
    assert np.array_equal(np.round(y_order2, decimals=2),
                          np.array([[ 0.04,  0.37, -0.02,  0.04, -0.39],
                                    [-1.44,  0.04,  0.04,  0.04,  0.05],
                                    [ 0.28,  0.06,  0.06,  0.04,  0.05]]))

"""Tests for optimization matrix functions of optimization_matrix.py."""


from pathlib import Path

import numpy as np

from tests.utils import load_tree_table
from treehfd.cartesian_partition import CartesianTreePartition
from treehfd.optimization_matrix import build_constr_mat
from treehfd.tree_structure import (
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
)

TESTS_DIR = Path(__file__).parent.resolve()


def test_build_constr_mat() -> None:
    """Test build_constr_mat function."""
    np.random.default_rng(11)
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed11.csv",
                      delimiter=",")
    y = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_y_n10_seed11.csv",
                      delimiter=",")
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    main_variables = extract_variables(variable_paths)
    interaction_list = extract_interactions(variable_paths)
    cartesian_partition = CartesianTreePartition(main_variables)
    X_bin = cartesian_partition.compute_cartesian_partition(X, tree_structure,
                                                            interaction_list)
    y_tree = y - np.mean(y)

    constr_mat, target = build_constr_mat(y_tree, interaction_list, X_bin,
                                          cartesian_partition.main_variables,
                                          cartesian_partition.partition_index)
    data = np.round(constr_mat.data, decimals=1)
    assert np.array_equal(data,
                          np.array([1.8, 3.7, 7.1, 4.5, 3.2, 2.1, 5.3, 2.1,
                                    5.5, 7.1, 4.5, 3., 5., 2., 5.5, 7.1, 4.5,
                                    3., 5., 2., 3.2, 9.5, 1., 9., 10., 10.,
                                    3., 5., 2., 1., 9., 10., 10., 1., 2., 5.,
                                    2., 3., 5., 2., 3., 5., 2., 1., 9., 10.,
                                    3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2,
                                    3.2, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5,
                                    4.5, 4.5, 7.1, 7.1, 7.1, 7.1, 7.1, 7.1,
                                    7.1, 7.1, 7.1, 4.5, 4.5, 4.5, 4.5, 4.5,
                                    4.5, 4.5, 4.5, 4.5]))
    indices = constr_mat.indices
    assert np.array_equal(indices,
                          np.array([7, 8, 9, 10, 7, 8, 9, 10, 11, 12, 13, 11,
                                    12, 13, 14, 15, 16, 14, 15, 16, 17, 18, 17,
                                    18, 19, 19,  0,  1,  2,  3,  4,  5,  6, 7,
                                    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    19, 0, 3, 5, 6, 7, 11, 14, 17, 19, 0, 4, 5,
                                    6, 8, 11, 14, 18, 19, 1, 4, 5, 6, 9, 12,
                                    15, 18, 19, 2, 4, 5, 6, 10, 13, 16, 18,
                                    19]))
    indptr = constr_mat.indptr
    assert np.array_equal(indptr,
                          np.array([0, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16,
                                    17, 20, 21, 22, 24, 25, 26, 29, 31, 32,
                                    33, 37, 40, 43, 45, 46, 55, 64, 73, 82]))
    assert np.unique(target[:27]) == 0
    target = np.round(target, decimals=2)
    assert np.array_equal(target[27:], np.array([7.39, 9.37, 3.72, -7.53]))

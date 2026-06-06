"""Tests for Cartesian partition functions of cartesian_partition.py."""


from pathlib import Path

import numpy as np

from tests.utils import load_tree_table
from treehfd.cartesian_partition import CartesianTreePartition
from treehfd.tree_structure import (
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
)

TESTS_DIR = Path(__file__).parent.resolve()


def test_compute_partition_main() -> None:
    """Test compute_partition_main function."""
    np.random.default_rng(11)
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    main_variables = extract_variables(variable_paths)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed11.csv",
                      delimiter=",")

    # Test all outputs of compute_partition_main with a dataset of 10 points.
    cartesian_partition = CartesianTreePartition(main_variables)
    X_bin_main = cartesian_partition.compute_partition_main(X,
                     tree_structure[0], tree_structure[2])
    assert np.array_equal(X_bin_main, np.array([[0, 3, 5, 6], [1, 4, 5, 6],
                                                [1, 4, 5, 6], [2, 4, 5, 6],
                                                [0, 4, 5, 6], [1, 4, 5, 6],
                                                [0, 4, 5, 6], [1, 4, 5, 6],
                                                [2, 4, 5, 6], [1, 4, 5, 6]]))
    assert np.all(cartesian_partition.main_variables == [0, 1, 2, 3])
    assert np.all(cartesian_partition.partition_index == [0, 3, 5, 6, 7])
    assert np.array_equal(cartesian_partition.split_list[0],
                          np.array([-0.78,  0.99]))
    assert np.array_equal(cartesian_partition.split_list[1], np.array([-1.61]))
    assert np.array_equal(cartesian_partition.split_list[2], np.array([]))
    assert np.array_equal(cartesian_partition.split_list[3], np.array([]))

    # Test the case where intermediate splits are removed (empty bins).
    cartesian_partition = CartesianTreePartition(main_variables)
    X_bin_main = cartesian_partition.compute_partition_main(
        X[[0, 3, 4, 6, 8], :], tree_structure[0], tree_structure[2])
    assert np.array_equal(X_bin_main[:, 0], np.array([0, 1, 0, 0, 1]))
    assert np.all(cartesian_partition.main_variables == [0, 1, 2, 3])
    assert np.all(cartesian_partition.partition_index == [0, 2, 4, 5, 6])
    assert cartesian_partition.split_list[0][0] == (-0.78 + 0.99)/2

    # Test case where input values equal splits.
    X[1, 0] = -0.78
    X[1, 1] = -1.61
    cartesian_partition = CartesianTreePartition(main_variables)
    X_bin_main = cartesian_partition.compute_partition_main(
        X, tree_structure[0], tree_structure[2])
    assert X_bin_main[1, 0] == 1
    assert X_bin_main[1, 1] == 4


def test_compute_partition_order2() -> None:
    """Test compute_partition_order2 function."""
    np.random.default_rng(11)
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed11.csv",
                      delimiter=",")
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    main_variables = extract_variables(variable_paths)
    cartesian_partition = CartesianTreePartition(main_variables)
    X_bin_main = cartesian_partition.compute_partition_main(X,
                     tree_structure[0], tree_structure[2])
    interaction_list = extract_interactions(variable_paths)

    X_bin_order2 = cartesian_partition.compute_partition_order2(X_bin_main,
                       interaction_list)
    assert np.array_equal(X_bin_order2, np.array([[ 7, 11, 14, 17, 19],
                                                  [ 9, 12, 15, 18, 19],
                                                  [ 9, 12, 15, 18, 19],
                                                  [10, 13, 16, 18, 19],
                                                  [ 8, 11, 14, 18, 19],
                                                  [ 9, 12, 15, 18, 19],
                                                  [ 8, 11, 14, 18, 19],
                                                  [ 9, 12, 15, 18, 19],
                                                  [10, 13, 16, 18, 19],
                                                  [ 9, 12, 15, 18, 19]]))
    assert np.array_equal(cartesian_partition.partition_index,
                           np.array([0, 3, 5, 6, 7, 11, 14, 17, 19, 20]))
    assert np.array_equal(cartesian_partition.cell_list[0],
                          np.array([[0, 3], [0, 4], [1, 4], [2, 4]]))
    assert np.array_equal(cartesian_partition.cell_list[1],
                          np.array([[0, 5], [1, 5], [2, 5]]))
    assert np.array_equal(cartesian_partition.cell_list[2],
                          np.array([[0, 6], [1, 6], [2, 6]]))
    assert np.array_equal(cartesian_partition.cell_list[3],
                          np.array([[3, 6], [4, 6]]))
    assert np.array_equal(cartesian_partition.cell_list[4],
                          np.array([[5, 6]]))
    assert np.array_equal(cartesian_partition.counts_list[0],
                          np.array([1, 2, 5, 2]))
    assert np.array_equal(cartesian_partition.counts_list[1],
                          np.array([3, 5, 2]))
    assert np.array_equal(cartesian_partition.counts_list[2],
                          np.array([3, 5, 2]))
    assert np.array_equal(cartesian_partition.counts_list[3], np.array([1, 9]))
    assert np.array_equal(cartesian_partition.counts_list[4], np.array([10]))


def test_predict_partition() -> None:
    """Test predict_partition function."""
    np.random.default_rng(11)
    tree_table = load_tree_table()
    tree_structure = extract_tree_structure(tree_table)
    variable_paths = extract_variable_paths(tree_structure, depth_variable=3)
    main_variables = extract_variables(variable_paths)
    interaction_list = extract_interactions(variable_paths)
    X = np.genfromtxt(f"{TESTS_DIR}/datasets/dataset_X_n10_seed11.csv",
                      delimiter=",")
    cartesian_partition = CartesianTreePartition(main_variables)
    _ = cartesian_partition.compute_cartesian_partition(X, tree_structure,
                                                            interaction_list)
    X_new = np.array([[0, 0, 0, 0, 0, 0], [0, -2, 0, 0, 0, 0],
                      [1, -2, 0, 0, 0, 0]])

    X_bin_main, X_bin_order2 = cartesian_partition.predict_partition(X_new,
                                   interaction_list)
    assert np.array_equal(X_bin_main, np.array([[1, 4, 5, 6], [1, 3, 5, 6],
                                                [2, 3, 5, 6]]))
    assert np.array_equal(X_bin_order2, np.array([[ 9, 12, 15, 18, 19],
                                                  [ 9, 12, 15, 17, 19],
                                                  [10, 13, 16, 17, 19]]))

    # Test empty partitions.
    cartesian_partition = CartesianTreePartition(np.empty(0))
    _ = cartesian_partition.compute_cartesian_partition(X, tree_structure,
                                                            interaction_list)
    X_bin_main, X_bin_order2 = cartesian_partition.predict_partition(X_new,
                                   interaction_list)
    assert np.array_equal(X_bin_main, np.empty((3, 0)))
    assert np.array_equal(X_bin_order2, np.empty((3, 0)))

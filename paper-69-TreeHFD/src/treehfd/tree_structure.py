"""Functions to extract tree structure."""

import ast
import itertools

import numpy as np
import pandas as pd


def get_params(config: dict) -> tuple:
    """Extract main parameters of tree ensemble from config."""
    max_depth = int(config["learner"]["gradient_booster"]
                    ["tree_train_param"]["max_depth"])
    n_estimators = int(config["learner"]["gradient_booster"]
                       ["gbtree_model_param"]["num_trees"])
    base_score_str = config["learner"]["learner_model_param"]["base_score"]
    base_score = float(ast.literal_eval(base_score_str)[0])
    num_feature = int(config["learner"]["learner_model_param"]["num_feature"])
    num_parallel_tree = int(config["learner"]["gradient_booster"]
                            ["gbtree_model_param"]["num_parallel_tree"])
    num_target = int(config["learner"]["learner_model_param"]["num_target"])

    return(max_depth, n_estimators, base_score, num_feature,
           num_parallel_tree, num_target)


def extract_child_ids(child_id: str) -> int:
    """Extract ids of children nodes from table of xgboost model."""
    tree_child_id = child_id.split("-")
    if len(tree_child_id) > 1:
        return int(tree_child_id[1])
    return 0


def transform_variables(name: str) -> int:
    """Transform variable name in index."""
    if name == "Leaf":
        return -1
    return int(name[1:])


def extract_tree_structure(tree_table: pd.DataFrame) -> tuple:
    """Extract the tree structure.

    Parameters
    ----------
    tree_table : pd.DataFrame
        Dataframe containing the tree structure, derived from the
        xgboost method trees_to_dataframe.

    Returns
    -------
    tuple
        variables : np.ndarray
            array with the variable index for each node split,
            where -1 indicates terminal leaves.
        child_ids : np.ndarray
            array with indices of children nodes, with first row
            for left nodes and second row for right nodes.
        split_values : np.ndarray
            array with the list of node splitting values.
    """
    if tree_table.shape[0] > 1:
        variables = list(tree_table["Feature"])
        variables =  np.array([transform_variables(x) for x in variables])

        child_ids_left = [extract_child_ids(str(x)) for x in tree_table["Yes"]]
        child_ids_right = [extract_child_ids(str(x)) for x in tree_table["No"]]
        child_ids = np.array((child_ids_left, child_ids_right))

        split_values = np.array(tree_table["Split"])
    else:
        variables =  np.empty(0)
        child_ids =  np.empty((0, 2))
        split_values =  np.empty(0)

    return(variables, child_ids, split_values)


def extract_variable_paths(tree_structure: tuple, depth_variable: int) -> list:
    """Extract the variable list of each tree path.

    Parameters
    ----------
    tree_structure : tuple
        Tuple containing the splitting variables, children node indices,
        and splitting node values of the tree.
    depth_variable : int
        Variables are selected at the first depth_variable levels of the tree
        for the components of the decomposition.

    Returns
    -------
    List
        List of variable list of each tree path.
    """
    variables, child_ids = tree_structure[:2]
    var_paths_final: list[list[int]] = []
    var_paths = [[int(variables[0])]]
    node_list = [0]
    if depth_variable == 1:
        return var_paths
    # Loop through the tree levels to retrieve the variable lists
    # of all tree paths.
    for depth in range(depth_variable - 1):
        var_paths_child: list[list[int]] = []
        node_list_child: list[int] = []
        # Loop through all nodes at the current tree level.
        for k, node in enumerate(node_list):
            # Retrieve variable list and children nodes of current path.
            variable_list = var_paths[k]
            left_child = int(child_ids[0][node])
            right_child = int(child_ids[1][node])
            # Append splitting variables of children nodes to variable
            # lists of the two generated paths.
            if variables[left_child] >= 0:
                variable_left = [*variable_list, int(variables[left_child])]
                var_paths_child.append(variable_left)
                node_list_child.append(left_child)
            if variables[right_child] >= 0:
                variable_right = [*variable_list, int(variables[right_child])]
                var_paths_child.append(variable_right)
                node_list_child.append(right_child)
            # If current node is a terminal leave, store variable list.
            if (variables[left_child] == -1
                    and variables[right_child] == -1):
                var_paths_final.append(variable_list)
        # Reset node list and variable lists for next level iteration.
        node_list = node_list_child
        var_paths = var_paths_child
        # Store variable lists when depth_variable smaller than tree depth.
        if depth == depth_variable - 2:
            var_paths_final += var_paths
    # Deduplication of variable lists.
    return [list(set(x)) for x in var_paths_final]


def extract_variables(variable_paths: list[list[int]]) -> np.ndarray:
    """Extract the unique and sorted list of variables from all tree paths.

    Parameters
    ----------
    variable_paths : list
        List of variable list of each tree path.

    Returns
    -------
    variable_list : list
        Unique and sorted list of variables of the tree.
    """
    variable_list = np.array(list(set(itertools.chain(*variable_paths))))
    variable_list.sort()

    return variable_list


def extract_interactions(variable_paths: list[list[int]]) -> list:
    """Extract the variable interactions of order two from all tree paths.

    Parameters
    ----------
    variable_paths : list
        List of variable list of each tree path.

    Returns
    -------
    interactions : list
        List of variable index pairs, with a pair for each interaction.
    """
    # Generate all interactions of order two from tree path lists.
    interactions: list[list[int]] = []
    for x in variable_paths:
        interactions += [[i, j] for i,j in itertools.combinations(x, 2)]
    interactions = [list(x) for x in
                        {tuple(x) for x in interactions}]
    interactions.sort()

    return interactions

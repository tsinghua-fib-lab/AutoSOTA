"""Class to define the TreeHFD decomposition of a single tree."""


import numpy as np
import pandas as pd
from numpy.linalg import lstsq as _np_lstsq

from treehfd.cartesian_partition import CartesianTreePartition
from treehfd.optimization_matrix import build_constr_mat
from treehfd.tree_structure import (
    extract_interactions,
    extract_tree_structure,
    extract_variable_paths,
    extract_variables,
)


class TreeHFD:
    """TreeHFD decomposition of a single tree.

    This class is the TreeHFD decomposition of a single tree of an
    ensemble. A least square problem is solved to get the coefficients
    defining the components of the decomposition. This class is called
    sequentially by XGBTreeHFD to get the TreeHFD of an xgboost tree
    ensemble.

    Parameters
    ----------
    tree_table : pd.DataFrame
        The table with the structure of the considered tree, obtained
        from xgb_model.get_booster().trees_to_dataframe().
    interaction_order : int, default=2
        Set to 1 to fit only main effects, or to 2 to also include
        second-order interactions in the TreeHFD decomposition.
    interaction_list: np.ndarray, default=None
        Predefined list of second-order interactions to be estimated in the
        decomposition. Each row defines an interaction with two integers
        for the variable indices. Default=None, and interactions are
        automatically extracted from tree paths.
    depth_variable : int
        Variables are selected at the first depth_variable levels of the tree
        for the components of the decomposition.

    Attributes
    ----------
    tree_structure : tuple
        Structure of the tree, i.e., the splitting variables, children
        node indices, and splitting values.
    interaction_order : int, default=2
        Set to 1 to fit only main effects, or to 2 to also include
        second-order interactions in the TreeHFD decomposition.
    interaction_list : list
        The list of interactions, defined as variable pairs, that occur
        in the tree paths.
    eta0 : float, default=0
        Intercept of the TreeHFD decomposition of the tree.
    cartesian_partition : CartesianTreePartition
        Cartesian tree partitions, i.e., variable indices for main
        effects, cell index of each component partition, list of splits
        for each variable, list of cells for each interaction, and
        size of these cells.
    hfd_coeffs : np.array
        Array with coefficients defining the values of the decomposition
        components in each cell of the Cartesian tree partitions.
    """

    def __init__(self, tree_table: pd.DataFrame,
                 interaction_order: int, interaction_list: np.ndarray | None,
                 depth_variable: int) -> None:
        """Initialize TreeHFD from tree structure."""
        self.tree_structure: tuple[np.ndarray, np.ndarray, np.ndarray,
                                  ] = extract_tree_structure(tree_table)
        self.interaction_order = interaction_order
        self.interaction_list: list[list[int]] = []
        main_variables = np.empty(0, dtype=int)
        if len(self.tree_structure[0]) > 0:
            variable_paths = extract_variable_paths(self.tree_structure,
                                                    depth_variable)
            main_variables = extract_variables(variable_paths)
            order_two: int = 2
            if interaction_order == order_two:
                self.interaction_list = extract_interactions(variable_paths)
                if interaction_list is not None:
                    self.interaction_list = [list(x) for x in
                        {tuple(x) for x in self.interaction_list}
                        & {tuple(x) for x in interaction_list}]
        self.eta0 = 0.0
        self.cartesian_partition = CartesianTreePartition(main_variables)
        self.hfd_coeffs: np.ndarray = np.empty(0)

    def fit(self, X: np.ndarray, y_tree: np.ndarray) -> None:
        """Fit TreeHFD decomposition of a single tree.

        Parameters
        ----------
        X : np.ndarray
            The input data used to train the xgboost model.
        y_tree : np.ndarray
            Output of the original tree for the training data.
        """
        # Compute intercept.
        self.eta0 = np.mean(y_tree)
        y_tree = y_tree - self.eta0

        # Compute Cartesian partitions.
        X_bin = self.cartesian_partition.compute_cartesian_partition(X,
                    self.tree_structure, self.interaction_list)

        # Build matrix and target for optimization.
        constr_mat, target = build_constr_mat(y_tree, self.interaction_list,
            X_bin, self.cartesian_partition.main_variables,
            self.cartesian_partition.partition_index)

        # Fit treehfd coefficients.
        self.hfd_coeffs = _np_lstsq(constr_mat.toarray(), target, rcond=None)[0]

    def predict(self, X_new: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict TreeHFD components of a single tree for new input data.

        Parameters
        ----------
        X_new : np.ndarray
            New input data where TreeHFD predictions are computed.

        Returns
        -------
        tuple
            y_main : np.ndarray
                array for the predictions of main effects
            y_order2 : np.ndarray
                array for predictions of second-order interactions
                (columns are ordered according to interaction_list).
        """
        # Drop new data in Cartesian partitions.
        X_bin_main, X_bin_order2 = self.cartesian_partition.predict_partition(
            X_new, self.interaction_list)

        # Compute predictions from cell coefficients.
        y_main =  self.hfd_coeffs[X_bin_main]
        y_order2 =  self.hfd_coeffs[X_bin_order2]

        return(y_main, y_order2)

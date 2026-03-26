"""Main class to define TreeHFD decomposition of xgboost model."""


import json

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import logit
from tqdm import tqdm

from treehfd.tree import TreeHFD
from treehfd.tree_structure import get_params
from treehfd.validation import (
    check_data,
    check_depth_variable,
    check_interaction_list,
    check_interaction_order,
    check_xgb_model_learner,
    check_xgb_model_type,
    check_xgb_params,
)


class XGBTreeHFD:
    """TreeHFD decomposition of a xgboost model.

    XGBTreeHFD is the TreeHFD decomposition of a xgboost tree ensemble,
    defined as the Hoeffing functional decomposition of the target tree
    ensemble, where the hierarchical orthogonality constraints are
    discretized over the Cartesian tree partitions. The TreeHFD
    algorithm solves a least square problem for each tree to find the
    coefficients defining the set of functional components of the
    decomposition, which are all piecewise constant on the Cartesian
    tree partitions.

    Parameters
    ----------
    xgb_model : xgb.sklearn.XGBModel
        The xgboost model for regression or binary classification to be
        decomposed with TreeHFD.

    Attributes
    ----------
    xgb_model : xgb.sklearn.XGBModel
        The input xgboost model.
    config : dict
        The config of the xgboost model with all settings and
        parameters, from xgb_model.get_booster().save_config().
    max_depth : int
        Tree depth parameter of xgb_model (must be greater than 0).
    n_estimators : int
        Number of trees of xgb_model (must be greater than 0).
    base_score : float
        Base score of xgb_model.
    num_feature : int
        The number of variables of the data used to fit xgb_model.
    xgb_table : pd.core.frame.DataFrame
        The table with the tree structures, obtained from
        xgb_model.get_booster().trees_to_dataframe().
    interaction_order : int, default=2
        Set to 1 to fit only main effects, or to 2 to also include
        second-order interactions in the TreeHFD decomposition.
    interaction_list : np.array, default=np.empty((0, 0))
        The list of interactions, defined as variable pairs.
    depth_variable : int, default=max_depth
        Variables are selected at the first depth_variable levels of the tree
        for the components of the decomposition. Set to max_depth by default.
    treehfd_list : list, default=[]
        The list of the TreeHFD decomposition for each tree.
    eta0 : float, default=0
        Intercept of the TreeHFD decomposition of xgb_model.

    Examples
    --------
    >>> import numpy as np
    >>> import xgboost as xgb
    >>> from treehfd import XGBTreeHFD
    >>> from treehfd.validation import sample_data
    >>> np.random.seed(11)
    >>> X, y = sample_data(nsample=100)
    >>> xgb_model = xgb.XGBRegressor()
    >>> xgb_model = xgb_model.fit(X, y)
    >>> treehfd_model = XGBTreeHFD(xgb_model)
    >>> treehfd_model.fit(X, interaction_order=2)
    >>> X_new, y_new = sample_data(nsample=3)
    >>> y_main, y_order2 = treehfd_model.predict(X_new)
    >>> print(f'TreeHFD intercept: {treehfd_model.eta0}')
    >>> print(f'TreeHFD main effect predictions: {y_main}')
    >>> print(f'TreeHFD interaction predictions: {y_order2}')
    >>> interactions = treehfd_model.interaction_list
    >>> print(f'TreeHFD interactions: {interactions}')
    """

    def __init__(self, xgb_model: xgb.sklearn.XGBModel) -> None:
        """Initialize XGBTreeHFD from xgboost model."""
        # Check input xgboost model.
        check_xgb_model_type(xgb_model)
        if xgb_model.__sklearn_is_fitted__():
            booster = xgb_model.get_booster()
        else:
            error_msg = "Need to fit or load xgboost model first."
            raise ValueError(error_msg)
        config = json.loads(booster.save_config())
        check_xgb_model_learner(config)
        (max_depth, n_estimators, base_score, num_feature,
         num_parallel_tree, num_target) = get_params(config)
        check_xgb_params(max_depth, n_estimators, num_parallel_tree, num_target)

        # Initialize TreeHFD.
        self.xgb_model = xgb_model
        self.config = config
        self.max_depth: int = max_depth
        self.n_estimators: int = n_estimators
        self.base_score: float = base_score
        self.num_feature: int = num_feature
        self.xgb_table = booster.trees_to_dataframe()
        self.interaction_order: int = 2
        self.interaction_list = np.empty((0, 0))
        self.depth_variable: int = max_depth
        self.treehfd_list: list[TreeHFD] = []
        self.eta0 = 0.0

    def fit(self, X: np.ndarray, interaction_order: int = 2,
            interaction_list: np.ndarray | None = None,
            depth_variable: int | None = None,
            verbose: bool = True) -> None:
        """Fit TreeHFD decomposition of the provided xgboost model.

        Parameters
        ----------
        X : np.ndarray
            The input data used to train the xgboost model.
        interaction_order : int, default=2
            Set to 1 to fit only main effects, or to 2 to also include
            second-order interactions in the TreeHFD decomposition.
        interaction_list: np.ndarray, default=None
            Predefined list of second-order interactions to be estimated in the
            decomposition. Each row defines an interaction with two integers
            for the variable indices. Default=None, and interactions are
            automatically extracted from tree paths.
        depth_variable : int, default=None
            Variables are selected at the first depth_variable levels of the
            tree for the components of the decomposition. Default is None,
            and all variables are selected.
        verbose : bool, default=True
            Set to False to deactivate the console display of computation
            progress (% of trees).
        """
        # Check inputs.
        check_data(X, "X", self.num_feature)
        check_interaction_order(interaction_order)
        self.interaction_order = interaction_order
        check_depth_variable(depth_variable)
        check_interaction_list(interaction_list)
        if depth_variable is not None:
            self.depth_variable = depth_variable

        # Compute original tree predictions.
        tree_predictions = np.zeros((X.shape[0], self.n_estimators))
        for tree_idx in range(self.n_estimators):
            tree_predictions[:, tree_idx] = self.xgb_model.predict(
                X, iteration_range=(tree_idx, tree_idx + 1), output_margin=True)
        ratio_score = (self.n_estimators - 1)/self.n_estimators
        if self.config["learner"]["objective"]["name"] != "binary:logistic":
            tree_predictions -= ratio_score * self.base_score
        else:
            tree_predictions -= ratio_score * logit(self.base_score)

        # Fit TreeHFD decomposition for each tree.
        self.treehfd_list = []
        self.interaction_list = np.empty((0, 0), dtype=int)
        self.eta0 = 0
        interaction_list_raw: list[list[list[int]]] = []
        for tree_idx in tqdm(range(self.n_estimators), disable=not verbose):
            tree_table = pd.DataFrame(
                self.xgb_table[self.xgb_table["Tree"] == tree_idx])
            y_tree = tree_predictions[:, tree_idx]
            tree = TreeHFD(tree_table, interaction_order, interaction_list,
                           self.depth_variable)
            tree.fit(X, y_tree)
            self.treehfd_list.append(tree)
            self.eta0 += tree.eta0
            interaction_list_raw.append(tree.interaction_list)
        interaction_list_raw = [x for x in interaction_list_raw if len(x) > 0]
        if len(interaction_list_raw) > 0:
            self.interaction_list = np.unique(np.concatenate(
                                        interaction_list_raw, axis=0), axis=0)

    def predict(self, X_new: np.ndarray, verbose: bool = True) -> tuple:
        """Predict TreeHFD components for new input data.

        Parameters
        ----------
        X_new : np.ndarray
            New input data where TreeHFD predictions are computed.
        verbose : bool, default=True
            Set to False to deactivate the console display of computation
            progress (% of trees).

        Returns
        -------
        tuple
            y_main : np.ndarray
                array for the predictions of main effects
            y_order2 : np.ndarray
                array for predictions of second-order interactions
                (columns are ordered according to interaction_list).
        """
        # Check inputs.
        if len(self.treehfd_list) == 0:
            error_msg = "Fit TreeHFD before computing predictions."
            raise ValueError(error_msg)
        check_data(X_new, "X_new", self.num_feature)

        # Compute and aggregate tree predictions.
        y_main = np.zeros_like(X_new)
        y_order2 = np.zeros((X_new.shape[0], self.interaction_list.shape[0]))
        for tree_idx in tqdm(range(self.n_estimators), disable=not verbose):
            tree = self.treehfd_list[tree_idx]
            y_main_tree, y_order2_tree = tree.predict(X_new)
            main_variables = tree.cartesian_partition.main_variables
            y_main[:, main_variables] += y_main_tree
            interaction_index = []
            for interaction in tree.interaction_list:
                idx = np.where(np.all(self.interaction_list == interaction,
                                      axis=1))[0].tolist()
                interaction_index += idx
            y_order2[:, interaction_index] += y_order2_tree

        return(y_main, y_order2)

"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

import json
import math
from time import perf_counter

import numpy as np
from scipy.special import binom, beta, factorial
from functools import partial
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from shapiq.explainer.tree.validation import validate_tree_model
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset
from .base import CoeffsTreeModel
from warnings import warn
import xgboost as xgb

from .cpp_implementation import convert_forest_to_matrix

INDICES_C_IMPLEMENTATION_CAPABLE = ["SII", "SV", "BII", "BV", "CHII", "CV"]
INDICES_CII_IMPLEMENTATION_CAPABLE = [
    "FSII",
    "FBII",
    "STII",
    "WBII",
] + INDICES_C_IMPLEMENTATION_CAPABLE


def get_leaf_matrix(tree, n_features, leaf_dict):
    leaf_matrix = []

    def recurse(node_id, current_path):
        if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
            path_to_add = current_path.copy()
            path_to_add.append(tree.values[node_id].item())
            leaf_matrix.append(path_to_add)
            try:
                leaf_dict[tuple(path_to_add[:-1])] += tree.values[node_id].item()
            except KeyError:
                leaf_dict[tuple(path_to_add[:-1])] = tree.values[node_id].item()
            return
        feature = tree.features[node_id]
        # Go left
        current_path[feature] = 0
        recurse(tree.children_left[node_id], current_path)
        # Go right
        current_path[feature] = 1
        recurse(tree.children_right[node_id], current_path)
        current_path[feature] = -1  # Reset for backtracking

    recurse(0, [-1] * (n_features))
    return np.array(leaf_matrix)


def get_A_B_intervals(leaf_matrix, n_features):
    A_B_NB_intervals = []
    for leaf in leaf_matrix:
        A_intervals = set()
        B_intervals = set(range(n_features))
        N_intervals = set(range(n_features))
        for feature_index, feature_value in enumerate(
            leaf[:-1]
        ):  # Exclude the last element (leaf value)
            if feature_value == 0:
                # We must remove the feature from B_intervals
                B_intervals.discard(feature_index)
            elif feature_value == 1:
                # We must add the feature to A_intervals
                A_intervals.add(feature_index)
        A_B_NB_intervals.append(
            (
                list(sorted(A_intervals)),
                list(sorted(B_intervals)),
                list(sorted(N_intervals.difference(B_intervals))),
                leaf[-1],
            )
        )
    return A_B_NB_intervals


class InterventionalTreeExplainer:
    """Interventional Tree Shap explainer for a single decision tree.

    This class implements the interventional TreeSHAP algorithm for a single
    sklearn DecisionTreeClassifier or DecisionTreeRegressor.

    Attributes.
    ----------
    tree : Any
        Validated tree structure returned by validate_tree_model for the fitted
        sklearn tree.
    data : np.ndarray
        Background dataset used to compute the reference point.
    reference_point : np.ndarray
        The baseline/reference feature values (mean over the background data).

    Methods.
    -------
    _compute_interventional_shapley_value(x, reference_points) -> np.ndarray
        Compute interventional Shapley values for one instance x given reference points.
    explain_function(x, **kwargs) -> InteractionValues
        Compute Shapley values for a single instance and return InteractionValues.
    """

    def __init__(
        self,
        model: (
            DecisionTreeClassifier
            | DecisionTreeRegressor
            | RandomForestRegressor
            | RandomForestClassifier
        ),
        data: np.ndarray,
        class_index: int | None = None,
        debug: bool = False,
        max_order: int = 1,
        index: str = "SII",
        index_func: callable | None = None,
        p: float = 0.5,
    ) -> None:
        """Implementation of Interventional Tree Shap.

        Args:
            model (DecisionTreeClassifier | DecisionTreeRegressor): Tree Model
            data (np.ndarray): Backgroud data. Reference point is the mean of all points.
            class_index (int | None): Class label to explain (only for classifiers).
            debug (bool): Whether to print debug information.
            max_order (int): Maximum order of interactions to compute.
            index (str): The interaction index to compute.
            index_func (callable | None): Custom index function if index is not recognized.
            p (float): Probability parameter for weighted Banzhaf index (WBII).
        Raises:
            NotImplementedError: Throws exception if using something else than sklearn trees.
        """
        # Additional initialization for interventional explainer
        # if isinstance(model, (XGBClassifier, XGBRegressor)):
        #     self.tree = _from_xgboost(model)
        # else:

        if isinstance(model, (GridSearchCV,)):
            model = model.best_estimator_

        if isinstance(model, (XGBClassifier, XGBRegressor)):
            self.decision_type = "<"
            self.decision_function = (
                lambda a, b, c: (np.isnan(a) and c) or a < b
            )  # XGBoost uses strict less only considering the first 6 decimals
        else:
            self.decision_type = "<="
            self.decision_function = lambda a, b, c: (np.isnan(a) and c) or (a <= b)
            # print("Using decision function with '<=' operator.")

        # If Classification model and class_index is None, set to 1
        if class_index is None and hasattr(model, "predict_proba"):
            class_index = 1
        a = perf_counter()
        self.tree = validate_tree_model(model, class_label=class_index)
        b = perf_counter()
        # print("Tree Validation Time: ", b - a)
        self.reference_data: np.ndarray = data.astype(np.float32)
        self.debug = debug
        self.max_order = max_order
        self.index = index
        self.n_players = data.shape[1]
        self.index_func = index_func
        self.p = p
        if class_index is not None:
            # If XGBoost model, use DMatrix to obtain logits
            if isinstance(model, XGBClassifier):
                dmatrix_data = xgb.DMatrix(self.reference_data)
                logits = model.get_booster().predict(dmatrix_data, output_margin=True)
                if logits.ndim == 1:
                    # Binary classification case
                    if class_index == 1:
                        self.baseline_value = np.mean(logits).astype(np.float64)
                    else:
                        self.baseline_value = np.mean(-logits).astype(np.float64)
                else:
                    self.baseline_value = np.mean(logits[:, class_index]).astype(
                        np.float64
                    )
            else:
                self.baseline_value = np.mean(
                    model.predict_proba(self.reference_data)[:, class_index]
                ).astype(np.float64)
        else:
            self.baseline_value = np.mean(model.predict(self.reference_data)).astype(
                np.float64
            )

        a = perf_counter()
        leaf_dict = {}
        self.leaf_matrix = [
            get_leaf_matrix(tree, data.shape[1], leaf_dict) for tree in self.tree
        ]
        self.leaf_matrix = [(*t, v) for t, v in leaf_dict.items()]
        b = perf_counter()
        # print("Python Conversion Time: ", b - a)
        # print("Child left arrays:")
        # print(np.ascontiguousarray(np.vstack([t.children_left for t in self.tree])))
        # print("Child right arrays:")
        # print(np.ascontiguousarray(np.vstack([t.children_right for t in self.tree])))
        # print("Feature arrays:")
        # print(np.ascontiguousarray(np.vstack([t.features for t in self.tree])))
        # print("Value arrays:")
        # print(np.ascontiguousarray(np.vstack([t.values for t in self.tree])))
        # a = perf_counter()
        # self.leaf_matrix2 = convert_forest_to_matrix(
        #     children_left=
        #     np.ascontiguousarray(np.vstack([t.children_left for t in self.tree])),
        #     children_right=
        #     np.ascontiguousarray(np.vstack([t.children_right for t in self.tree])),
        #     features=
        #     np.ascontiguousarray(np.vstack([t.features for t in self.tree])),
        #     values=np.ascontiguousarray(np.vstack([t.values for t in self.tree])),
        #     n_features=data.shape[1],
        # )
        # b = perf_counter()
        # print("C++ Conversion Time: ", b - a)
        ## Convert dict to leaf_matrix

        # print("LEAF MATRIX: ", self.leaf_matrix)
        # print("LEAF MATRIX C++: ", self.leaf_matrix2)

        self.A_B_NB_Intervals = get_A_B_intervals(self.leaf_matrix, data.shape[1])

    def interaction_weight_func(self, interaction_size, coalition_size, n_players):
        """The general API for Interaction weight functions

        Args:
            interaction_size (int): The coaltion to compute the effect for
            coalition_size (int): the coalition which is a superset of s
            n_players (int): The total number of players.
        """
        if self.index in ["SII", "SV"]:
            return 1 / (
                (n_players + interaction_size - 1)
                * binom(n_players - interaction_size, coalition_size)
            )
        if self.index in ["BII", "BV", "FBII"]:
            return 1 / (2 ** (n_players - interaction_size))
        if self.index in ["WBII"]:
            return (self.p) ** coalition_size * (1 - self.p) ** (
                n_players - interaction_size - coalition_size
            )
        if self.index in ["CHII", "CV"]:
            return interaction_size / (
                (interaction_size + coalition_size)
                * binom(n_players, coalition_size + interaction_size)
            )
        if self.index in ["FSII"]:
            return (
                factorial(2 * interaction_size - 1)
                / (factorial(interaction_size - 1)) ** 2
                * (
                    factorial(interaction_size + coalition_size - 1)
                    * factorial(n_players - coalition_size - 1)
                    / factorial(n_players + interaction_size - 1)
                )
            )
        if self.index in ["STII"]:
            return (
                1
                / (binom(n_players - 1, coalition_size))
                * interaction_size
                / n_players
            )
        warn(
            f"Index {self.index} not recognized. Checking if callable function was given."
        )
        if self.index_func is None:
            raise ValueError(
                f"Index function must be provided if index {self.index} is not recognized."
            )
        return self.index_func(interaction_size, coalition_size, n_players)

    def interaction_weight_to_moebius_weight(
        self,
        interaction_size: int,
        coalition_size: int,
    ):
        """Converts the Interaction Weight Representation to Möbius Representation.

        Args:
            interaction_size (int): The coalition to compute the effect for
            coalition_size (int): The coalition which is a superset of s.
            interaction_weight_func (N x N -> R): The interaction weight function.
        """
        return self.interaction_weight_func(
            interaction_size, coalition_size - interaction_size, coalition_size
        )

    def interaction_weight_to_moebius_weight_gv(
        self,
        interaction_size: int,
        coalition_size: int,
    ):
        return sum(
            [
                self.interaction_weight_func(
                    interaction_size=interaction_size,
                    coalition_size=l,
                    n_players=l + interaction_size,
                )
                for l in range(
                    self.n_players - coalition_size,
                    self.n_players - interaction_size + 1,
                )
            ]
        )

    def general_weight_function(self, A, B, N, U, moebius_weight_func):
        """Computes a general weight for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
            möbius_weight_func: Möbius weight function to use.
        Returns:
            The general weight.
        """
        u_0 = len(U.intersection(N.difference(B)))
        a = len(A)
        b = len(B)
        n = len(N)
        u = len(U)
        sign = (-1) ** u_0
        return sign * sum(
            [
                (-1) ** k
                * binom(n - b - u_0, k)
                * moebius_weight_func(coalition_size=k + u_0 + a, interaction_size=u)
                for k in range(n - b - u_0 + 1)
            ]
        )

    def moebius_value(self, S, A, B):
        return sum(
            [
                (-1) ** (len(S) - len(T))
                * (1 if A.issubset(set(T)) and set(T).issubset(B) else 0)
                for T in powerset(S)
            ]
        )

    def general_weight_fbii(self, A, B, N, U):
        """Computes the general weight for FBII for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
        Returns:
            The general weight for FBII.
        """
        # Make sure that A,B,N,U contain integers
        A = set(map(int, A))
        B = set(map(int, B))
        N = set(map(int, N))
        U = set(map(int, U))

        u_0 = len(U.intersection(N.difference(B)))
        a = len(A)
        b = len(B)
        n = len(N)
        u = len(U)
        # return (-1)**(len(U.difference(A))) + (-1)**(self.max_order - u) * (1/2)**(a+u_0-u)* sum(
        #     [
        #         (-1)**l * binom(n-b-u_0, l) * binom(a+l+u_0-u-1,self.max_order - u)
        #         for l in range(self.max_order + 1 - u_0 - a, n - b - u_0 + 1)
        #     ]
        # )

        w1 = (-1) ** (u_0) if A.issubset(U) else 0
        # print("----")
        # for T in powerset(N.difference(B)):
        #     print("T: ", T)
        #     print("A: ", A)
        #     print("A.union T: ", A.union(set(T)))
        #     print("U: ", U)
        #     print("Moebius Value: ", self.moebius_value(U, A.union(set(T)), N))
        #     w1 += (-1) ** (len(T)) * self.moebius_value(U, A.union(set(T)), N)
        # print("N\\B \\cup A", N.difference(B).union(A))
        # print("N \\B", N.difference(B))
        # print("W1: ", w1, (-1)**(u_0) if A.issubset(U) else 0)
        # print("----")

        w = 0
        # for T in powerset(N.difference(B)):
        #     T = set(T)
        #     for L in powerset(N):
        #         L = set(L)
        #         if U.issubset(L) and len(L) > self.max_order:
        #             w += (
        #                 (-1) ** (len(T))
        #                 * (-1) ** (self.max_order - u)
        #                 * (1 / 2) ** (len(L) - u)
        #                 * binom(len(L) - u - 1, self.max_order - u)
        #                 * self.moebius_value(L, A.union(T), B)
        #             )
        w = sum(
            [
                (-1) ** (u_0 + l + self.max_order - u)
                * (1 / 2) ** (a + l + u_0 - u)
                * binom(a + l + u_0 - u - 1, self.max_order - u)
                * binom(n - b - u_0, l)
                for l in range(self.max_order + 1 - u_0 - a, n - b - u_0 + 1)
            ]
        )

        return w1 + w

    def shapley_weight_function(self, a, b):
        """Computes the Shapley weight for given set sizes a and b.

        Args:
            a: Size of set A.
            b: Size of set B.
        Returns:
            The Shapley weight.
        """
        return 1.0 / ((a + b + 1) * math.comb(a + b, b))

    def shapley_based_weight_function(self, A, B, N, U):
        """Computes the Shapley based weight for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
        Returns:
            The Shapley based weight.
        """
        a = len(A) - len(B.intersection(U))
        b = len(N.difference(B.union(U)))
        sign = (-1) ** (len(U.intersection(N.difference(B))))
        return sign * 1.0 / ((a + b + 1) * math.comb(a + b, b))

    def banzhaf_weight_function(self, A, B, N, U):
        """Computes the Banzhaf based weight for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
        Returns:
            The Banzhaf based weight.
        """
        sign = (-1) ** (len(U.intersection(N.difference(B))))
        weight = 1.0 / (2 ** (len(N) + len(A) - len(B) - len(U)))
        return sign * weight

    def chaining_weight_function(self, A, B, N, U):
        """Computes the Chaining based weight for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
        Returns:
            The Chaining based weight.
        """
        u_0 = len(U.intersection(N.difference(B)))
        n = len(N)
        a = len(A)
        b = len(B)
        sign = (-1) ** (u_0)
        weight = len(U) * beta(u_0 + a, n - b - u_0 + 1)
        return sign * weight

    def update_values(
        self,
        interaction_to_values,
        const_prediction,
        A,
        B,
        NB,
        max_order,
        weight_func,
    ):
        """Updates the CII based on sets A and NB.

        Args:
            interaction_to_values: Mapping from interactions to their effects.
            const_prediction: Constant prediction value.
            A: Set A.
            B: Set B.
            NB: Set NB.
            max_order: Maximum order of interactions.
            weight_func: Weight function for the update.
        Returns:
            The updated Shapley values.
        """
        # Though A & NB there is already some filtering of which interactions could even be updated. Irrelevant features will not be part of A or NB.
        for U in powerset(A.union(NB), min_size=1, max_size=max_order):
            U = set(U)
            # Compute the weights
            weight = weight_func(A=A, B=B, N=NB.union(B), U=U)
            if self.debug:
                print("Updating interaction:", U)
                print("Weight:", weight)
                print("const_prediction:", const_prediction)
            # Update the values
            # Make U contain not numpy numbers
            U = set(map(int, U))
            U = tuple(sorted(tuple(U)))

            try:
                interaction_to_values[U] += (
                    np.array(weight * const_prediction).astype(np.float64).item()
                )
            except KeyError:
                interaction_to_values[U] = (
                    np.array(weight * const_prediction).astype(np.float64).item()
                )
        return interaction_to_values

    def _compute_interventional_shapley_value(
        self, x: np.ndarray, tree: TreeModel
    ) -> np.ndarray:
        """Computes the interventional Shapley value for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.
            reference_points: The reference points for the interventional Shapley value computation.

        Returns:
            The computed Shapley value.
        """
        shapley_values = np.zeros(x.shape)
        N = set(range(x.shape[0]))  # number of features
        D = self.reference_data.shape[0]  # number of reference points
        for r in self.reference_data:
            # Implement the interventional Shapley value computation here
            reference_point = r
            # Initialize Node Stack
            node_stack = [(0, (set(), N))]  # (node_id, (A, B))
            while node_stack:
                node_id, (A, B) = node_stack.pop()

                if self.debug:
                    print("Visiting node:", node_id)
                    print("Current A:", A)
                    print("Current B:", B)
                # Check if inner node
                is_inner_node = (
                    tree.children_left[node_id] != tree.children_right[node_id]
                )
                if is_inner_node:  # Inner Node
                    feature_index = tree.features[node_id]
                    child_node_x = (
                        tree.children_left[node_id]
                        if self.decision_function(
                            x[feature_index], tree.thresholds[node_id]
                        )
                        else tree.children_right[node_id]
                    )
                    child_node_ref = (
                        tree.children_left[node_id]
                        if self.decision_function(
                            reference_point[feature_index], tree.thresholds[node_id]
                        )
                        else tree.children_right[node_id]
                    )
                    if self.debug:
                        print(
                            f"Feature index: {feature_index}, x value: {x[feature_index]}, ref value: {reference_point[feature_index]}, threshold: {tree.thresholds[node_id]}"
                        )
                        print(
                            f"Child node x: {child_node_x}, Child node ref: {child_node_ref}"
                        )
                    # Update stack based on child nodes
                    if child_node_x == child_node_ref:
                        if self.debug:
                            print("Both go to the same child node.")
                            print("Adding to stack:", child_node_x, (A, B))
                        node_stack.append((child_node_x, (A, B)))
                    else:
                        if feature_index in B:  # Keeping Child of x
                            if self.debug:
                                print("Feature index in B, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_x,
                                    (A.union({feature_index}), B),
                                )
                            node_stack.append(
                                (child_node_x, (A.union({feature_index}), B))
                            )
                        if feature_index not in A:
                            if self.debug:
                                print("Feature index not in A, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_ref,
                                    (A, B.difference({feature_index})),
                                )
                            node_stack.append(
                                (child_node_ref, (A, B.difference({feature_index})))
                            )
                else:
                    # Update Shapley values based on A & B
                    NB = N.difference(B)
                    # Compute Coalition Values [Due to linearity one could also directly use the values at the leaf node and later divide by D]
                    const_coalition = tree.values[node_id] / D
                    if self.debug:
                        print("-----Updating at leaf node-----")
                        print("Node ID:", node_id)
                        print("A:", A)
                        print("B:", B)
                        print("NB: ", NB)
                        print("const_coalition:", const_coalition)

                    if len(A) > 0:  # Update the Shapley Value for all features in A
                        weight = self.shapley_weight_function(len(A) - 1, len(NB))
                        # weight = 1.0 / (
                        #     (len(A) + len(NB))
                        #     * math.comb(len(A) + len(NB) - 1, len(NB))
                        # )
                        for j in A:
                            shapley_values[j] += weight * const_coalition
                    if (len(NB)) > 0:  # Update the Shapley Value for all features in NB
                        weight = self.shapley_weight_function(len(A), len(NB) - 1)
                        # weight = 1.0 / (
                        #     (len(A) + len(NB)) * math.comb(len(A) + len(NB) - 1, len(A))
                        # )
                        for j in NB:
                            shapley_values[j] -= weight * const_coalition
                    if self.debug:
                        print("Updated Shapley values:", shapley_values)
                        print("-----Updating at leaf node-----")

        return shapley_values

    def _compute_interventional_cii_values(
        self,
        x: np.ndarray,
        interactions_dict: dict[tuple[int, ...], float],
        tree: TreeModel,
    ) -> dict[tuple[int, ...], float]:
        """Computes the interventional CII value for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.
            interactions_dict: Resulting interactions dictionary.
            tree: The decision tree to explain.
        Returns:
            interventional CII values in a dictionary.
        """
        N = set(range(x.shape[0]))  # number of features
        D = self.reference_data.shape[0]  # number of reference points
        for r in self.reference_data:
            # Implement the interventional CII value computation here
            reference_point = r
            # Initialize Node Stack
            node_stack = [(0, (set(), N))]  # (node_id, (A, B))
            while node_stack:
                node_id, (A, B) = node_stack.pop()

                if self.debug:
                    print("Visiting node:", node_id)
                    print("Current A:", A)
                    print("Current B:", B)
                # Check if inner node
                is_inner_node = (
                    tree.children_left[node_id] != tree.children_right[node_id]
                )
                if is_inner_node:  # Inner Node
                    feature_index = tree.features[node_id]
                    child_node_x = (
                        tree.children_left[node_id]
                        if self.decision_function(
                            x[feature_index],
                            tree.thresholds[node_id],
                            tree.children_left_default[node_id],
                        )
                        else tree.children_right[node_id]
                    )
                    child_node_ref = (
                        tree.children_left[node_id]
                        if self.decision_function(
                            reference_point[feature_index],
                            tree.thresholds[node_id],
                            tree.children_left_default[node_id],
                        )
                        else tree.children_right[node_id]
                    )
                    if self.debug:
                        print(
                            f"Feature index: {feature_index}, x value: {x[feature_index]}, ref value: {reference_point[feature_index]}, threshold: {tree.thresholds[node_id]}"
                        )
                        print(
                            f"Child node x: {child_node_x}, Child node ref: {child_node_ref}"
                        )
                    # Update stack based on child nodes
                    if child_node_x == child_node_ref:
                        if self.debug:
                            print("Both go to the same child node.")
                            print("Adding to stack:", child_node_x, (A, B))
                        node_stack.append((child_node_x, (A, B)))
                    else:
                        if feature_index in B:  # Keeping Child of x
                            if self.debug:
                                print("Feature index in B, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_x,
                                    (A.union({feature_index}), B),
                                )
                            node_stack.append(
                                (child_node_x, (A.union({feature_index}), B))
                            )
                        if feature_index not in A:
                            if self.debug:
                                print("Feature index not in A, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_ref,
                                    (A, B.difference({feature_index})),
                                )
                            node_stack.append(
                                (child_node_ref, (A, B.difference({feature_index})))
                            )
                else:
                    # Update Shapley values based on A & B
                    NB = N.difference(B)
                    D = self.reference_data.shape[0]
                    # Compute Coalition Values [Due to linearity one could also directly use the values at the leaf node and later divide by D]
                    const_coalition = tree.values[node_id]
                    if self.debug:
                        print("-----Updating at leaf node-----")
                        print("Node ID:", node_id)
                        print("A:", A)
                        print("B:", B)
                        print("NB: ", NB)
                        print("const_coalition:", const_coalition)
                    # const_coalition /= D

                    # Obtain the necessary weight function
                    if self.index in ["SII", "SV"]:
                        weight_function = self.shapley_based_weight_function
                    elif self.index in ["BII", "BV"]:
                        weight_function = self.banzhaf_weight_function
                    elif self.index in ["CHII", "CV"]:
                        weight_function = self.chaining_weight_function
                    elif self.index in ["GSII", "GSV"]:
                        weight_function = partial(
                            self.general_weight_function,
                            moebius_weight_func=self.interaction_weight_to_moebius_weight_gv,
                        )
                    elif self.index in ["FBII"]:
                        weight_function = self.general_weight_fbii
                    else:
                        weight_function = partial(
                            self.general_weight_function,
                            moebius_weight_func=self.interaction_weight_to_moebius_weight,
                        )
                    # Update the CII Values
                    self.update_values(
                        interaction_to_values=interactions_dict,
                        const_prediction=const_coalition,
                        A=A,
                        B=B,
                        NB=NB,
                        max_order=self.max_order,
                        weight_func=weight_function,
                    )
                    if self.debug:
                        print("Updated Shapley values:", interactions_dict)
                        print("-----Updating at leaf node-----")

        return interactions_dict

    def explain_function(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Computes the Shapley values for a single instance using interventional approach.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.
        """
        if self.index not in INDICES_CII_IMPLEMENTATION_CAPABLE:
            warn(
                f"Index {self.index} not recognized. Checking if callable function was given."
            )
            if self.index_func is None:
                raise ValueError(
                    f"Index function must be provided if index {self.index} is not recognized."
                )
            print("Using custom index function provided by user.")
            interaction = self.explain_function_cii(x, **_)
            # raise ValueError(f"Index {self.index} not supported in interventional explainer.")
        if self.index in ["FSII","FBII"]:
            print(
                f"Using CII computation for Faithful indices." if self.index=="FBII" else
                f"Using CII computation for FSII index. Only the order={self.max_order} values are valid FSII values."
            )
            interaction = self.explain_function_cii(x, **_)
        else:
            # Use C++ implementation for intervals-based approach. This is faster than the pure Python one, but currently only supports closed form indices.
            interaction = self.explain_function_intervals(x, **_)

        return interaction
    def explain_function_cii(
        self,
        x: np.ndarray,
        # use_cython: bool = False,
        **_: dict,
    ) -> InteractionValues:
        """Computes the CII values for a single instance using interventional approach.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.
        """
        # Assert that x is np.float32
        x = x.astype(np.float32).flatten()

        # Try to import Cython implementation
        from ._interventional import compute_interventional_cii_multi_tree

        interactions_dict: dict[tuple[int, ...], float] = {}
        for j, tree in enumerate(self.tree):
                if self.debug:
                    print(f"#####Computing CII values for tree {j}#####")
                interactions = {}
                interactions = self._compute_interventional_cii_values(
                    x, interactions, tree
                )
                for key, value in interactions.items():
                    try:
                        interactions_dict[key] += value
                    except KeyError:
                        interactions_dict[key] = value

                if self.debug:
                    print(f"#####Finished Computing CII values for tree #####")
        interactions_dict[()] = self.baseline_value
        interactions_dict = dict(
                    sorted(
                        interactions_dict.items(),
                        key=lambda item: (len(item[0]), item[0]),
                    )
                )

        # Divide by number of ref points to get average
        for key in interactions_dict.keys():
                if key != ():
                    interactions_dict[key] /= self.reference_data.shape[0]
        return InteractionValues(
                interactions_dict,
                max_order=self.max_order,
                min_order=1,
                index=self.index,
                n_players=self.n_players,
                baseline_value=interactions_dict[()],
            )

    def explain_function_intervals(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Computes the CII values for a single instance using interventional approach.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.
        """
        from .cpp_implementation import (
            interventional_update,
            interventional_update_batch,
        )

        if self.debug:
            print(
                "Assuming input first dimension is number of points to explain. If only 1D array we have extended it to 2D Using the shape:",
                x.shape,
            )
        interactions_to_values: dict[tuple[int, ...], float] = {}
        if self.debug:
            print(f"#####Computing CII values for tree #####")
        # print("A_B_NB_Intervals:", self.A_B_NB_Intervals)

        if self.index in ["SII", "SV"]:
            weight_type = "shapley"
        elif self.index in ["BII", "BV"]:
            weight_type = "banzhaf"
        elif self.index in ["CHII", "CV"]:
            weight_type = "chain"
        else:
            raise ValueError(f"Index {self.index} not supported in C++ implementation.")

        interventional_update_batch(
            interactions_to_values,
            self.A_B_NB_Intervals,
            self.max_order,
            weight_type,
        )

        if self.debug:
            print(f"#####Finished Computing CII values for tree #####")
        interactions_to_values[()] = self.baseline_value
        interactions_to_values = dict(
                sorted(
                    interactions_to_values.items(),
                    key=lambda item: (len(item[0]), item[0]),
                )
            )

        return InteractionValues(
            interactions_to_values,
            max_order=self.max_order,
            min_order=1,
            index=self.index,
            n_players=self.n_players,
            baseline_value=interactions_to_values[()],
        )

    # def explain_function_c_given_vector(
    #     self,
    #     x: np.ndarray,
    #     **_: dict,
    # ) -> list[InteractionValues]:
    #     from ._cext import interventional_iterative_bitmask

    #     interactions_list = []
    #     for i in range(x.shape[0]):
    #         values = np.zeros(x.shape[1], dtype=np.float64)

    #         # Prepare tree arrays as lists for batch processing
    #         children_left_list = [
    #             np.ascontiguousarray(tree.children_left, dtype=np.int32)
    #             for tree in self.tree
    #         ]
    #         children_right_list = [
    #             np.ascontiguousarray(tree.children_right, dtype=np.int32)
    #             for tree in self.tree
    #         ]
    #         features_list = [
    #             np.ascontiguousarray(tree.features, dtype=np.int32)
    #             for tree in self.tree
    #         ]
    #         thresholds_list = [
    #             np.ascontiguousarray(tree.thresholds, dtype=np.float64)
    #             for tree in self.tree
    #         ]
    #         values_list = [
    #             np.ascontiguousarray(tree.values, dtype=np.float64)
    #             for tree in self.tree
    #         ]

    #         # Process all trees in a single C++ call
    #         result = interventional_iterative_bitmask(
    #             x[i],
    #             np.ascontiguousarray(self.reference_data, dtype=np.float64),
    #             children_left_list,
    #             children_right_list,
    #             features_list,
    #             thresholds_list,
    #             values_list,
    #             values,
    #             max_order=self.max_order,
    #         )

    #         interactions_list.append(result)
    #     return interactions_list

    # def explain_function_c(
    #     self,
    #     x: np.ndarray,
    #     **_: dict,
    # ) -> list[InteractionValues]:
    #     """Computes the C values for a single instance using interventional approach.

    #     Args:
    #         x: The instance to explain as a 1-dimensional array.
    #         **kwargs: Additional keyword arguments are ignored.
    #     """
    #     from ._cext import interventional_iterative

    #     interactions_list = []

    #     # Prepare tree arrays as lists for batch processing
    #     children_left_list = [
    #         np.ascontiguousarray(tree.children_left, dtype=np.int32)
    #         for tree in self.tree
    #     ]
    #     children_right_list = [
    #         np.ascontiguousarray(tree.children_right, dtype=np.int32)
    #         for tree in self.tree
    #     ]
    #     features_list = [
    #         np.ascontiguousarray(tree.features, dtype=np.int32) for tree in self.tree
    #     ]
    #     thresholds_list = [
    #         np.ascontiguousarray(tree.thresholds, dtype=np.float64)
    #         for tree in self.tree
    #     ]
    #     values_list = [
    #         np.ascontiguousarray(tree.values, dtype=np.float64) for tree in self.tree
    #     ]

    #     for i in range(x.shape[0]):
    #         values = np.zeros(x.shape[1], dtype=np.float64)

    #         # Process all trees in a single C++ call
    #         result = interventional_iterative(
    #             x[i],
    #             np.ascontiguousarray(self.reference_data, dtype=np.float64),
    #             children_left_list,
    #             children_right_list,
    #             features_list,
    #             thresholds_list,
    #             values_list,
    #             values,
    #             max_order=self.max_order,
    #         )

    #         interactions_list.append(result)
    #     return interactions_list
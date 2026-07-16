import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
from typing import Any


def proxyspex(game, n, top_k: int, samples, grid_search, odd,sample_weights, random_state) -> dict[tuple[int, ...], float]:

    # Sample budget uniform coalitions (boolean lists) from game
    # if samples is of type int:
    if len(samples[1]) <= 10:
        return {}
    train_X = pd.DataFrame(
        samples[0],
        columns=np.array([f"f{i}" for i in range(n)]),
    )
    train_y = samples[1]


    if grid_search:
        base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=random_state)
        param_grid = {
            "num_leaves": [10, 40],
            "n_estimators": [500, 1000],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8],
            "min_child_samples": [10, 20],
            "max_depth": [10],
        }
        # Set up GridSearchCV with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="r2",
            cv=5,
            verbose=0,
            n_jobs=1,
        )
        grid_search.fit(train_X, train_y)
        best_model = grid_search.best_estimator_
    else:
        base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=random_state, max_depth=10)
        if sample_weights is None:
            best_model = base_model.fit(train_X, train_y)
        else:
            best_model = base_model.fit(train_X, train_y, sample_weight=sample_weights)

    ## print the number of trees in the best model
    print(f'Number of trees in best model: {best_model.booster_.num_trees()}')

    initial_transform = lgboost_to_fourier(best_model.booster_.dump_model())

    if odd:
        return top_k_interactions(initial_transform, top_k, odd=True)
    else:
        return top_k_interactions(initial_transform, top_k, odd=False)
        #return initial_transform


def lgboost_to_fourier(model_dict: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Extracts the aggregated Fourier coefficients from an LGBoost model dictionary.

    This method iterates over all trees in the LightGBM ensemble, computes the
    Fourier coefficients for each individual tree using the `_lgboost_tree_to_fourier`
    helper method, and then sums these coefficients to get the final Fourier
    representation of the complete model.

    Args:
    model_dict: A dictionary representing the trained LGBoost model, as
        produced by `model.booster_.dump_model()`.

    Returns:
        A dictionary that maps interaction tuples (representing Fourier frequencies)
        to their aggregated Fourier coefficients.
    """
    aggregated_coeffs = defaultdict(float)

    for tree_info in model_dict["tree_info"]:
        tree_coeffs = lgboost_tree_to_fourier(tree_info)
        for interaction, value in tree_coeffs.items():
            aggregated_coeffs[interaction] += value

    # Convert defaultdict to a standard dict, removing zero-valued coefficients
    return {k: v for k, v in aggregated_coeffs.items() if v != 0.0}

def lgboost_tree_to_fourier(tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Recursively strips the Fourier coefficients from a single LGBoost tree.

    This method traverses a tree's structure, as provided by LightGBM's `dump_model`
    method, and computes the Fourier representation of the piecewise-constant
    function that the tree defines. The logic is adapted from the work by Gorji et al. (2024).

    Args:
        tree_info: A dictionary representing a single decision tree from an LGBM model.

    Returns:
        A dictionary mapping interaction tuples to their corresponding coefficients for
        the single tree.

    References:
        Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
        "SHAP values via sparse Fourier representation"
        arXiv preprint arXiv:2410.06300 (2024).
    """

    def combine_coeffs(
        left_coeffs: dict[tuple[int, ...], float],
        right_coeffs: dict[tuple[int, ...], float],
        feature_idx: int,
    ) -> dict[tuple[int, ...], float]:
        """Combines Fourier coefficients from the left and right children of a split node."""
        combined_coeffs = {}
        all_interactions = set(left_coeffs.keys()) | set(right_coeffs.keys())

        for interaction in all_interactions:
            left_val = left_coeffs.get(interaction, 0.0)
            right_val = right_coeffs.get(interaction, 0.0)
            combined_coeffs[interaction] = (left_val + right_val) / 2

            new_interaction = tuple(sorted(set(interaction) | {feature_idx}))
            combined_coeffs[new_interaction] = (left_val - right_val) / 2
        return combined_coeffs

    def dfs_traverse(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
        """Performs a depth-first traversal of the tree to compute coefficients."""
        # Base case: if the node is a leaf, its function is a constant.
        if "leaf_value" in node:
            # The only non-zero coefficient is for the empty interaction (the bias term).
            return {(): node["leaf_value"]}
        # Recursive step: if the node is a split node.
        left_coeffs = dfs_traverse(node["left_child"])
        right_coeffs = dfs_traverse(node["right_child"])
        feature_idx = node["split_feature"]
        return combine_coeffs(left_coeffs, right_coeffs, feature_idx)

    return dfs_traverse(tree_info["tree_structure"])


def top_k_interactions(four_dict: dict[tuple[int, ...], float], k: int, odd: bool) -> dict[tuple[int, ...], float]:
    """Return the top-k Fourier coefficients whose interaction keys have an odd
    cardinality greater than 1.

    Parameters
    ----------
    four_dict
        Mapping from interaction tuples to Fourier coefficients.
    k
        Number of top interactions to return.

    Returns
    -------
    dict
        Dictionary of the selected interaction tuples mapped to their
        coefficients, ordered by descending magnitude.
    """
    # Sort by absolute coefficient magnitude descending
    items = sorted(four_dict.items(), key=lambda iv: abs(iv[1]), reverse=True)

    selected: list[tuple[tuple[int, ...], float]] = []
    for key, val in items:
        if len(key) > 1: 
            if (not odd) or (len(key) % 2 == 1):
                selected.append((key, val))
                if len(selected) >= k:
                    break

    return {k: v for k, v in selected}

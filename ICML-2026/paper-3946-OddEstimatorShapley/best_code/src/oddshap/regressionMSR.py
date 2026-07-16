from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import shap
from scipy.special import binom

# import random forest regressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# from pltreeshap import PLTreeExplainer

from shapiq import UnbiasedKernelSHAP
from shapiq import SHAPIQ
from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.game import Game
from shapiq.interaction_values import InteractionValues
from oddshap import InterventionalTreeExplainer
from sklearn.model_selection import GridSearchCV
from oddshap.abalation import (
    get_ablation_model,
    get_ablation_residual_approximator,
    validate_ablation_kwargs,
)


def pltree_array_to_interactions(pl_shap_values, num_features):
    # Create a dictionary to hold interaction values
    interactions_dict = {}
    for i in range(num_features):
        interactions_dict[(i,)] = 0.0
        for j in range(i + 1, num_features):
            interactions_dict[(i, j)] = 0.0

    # Now convert the array to the dictionary format. The Conversion is based on the comments in the .pyx file
    for i in range(num_features):
        for j in range(num_features):
            if i < j:
                interactions_dict[(i, j)] = pl_shap_values[i, j] + pl_shap_values[j, i]
    for i in range(num_features):
        interactions_dict[(i,)] = pl_shap_values[i, i] + 0.5 * sum(
            interactions_dict[(min(i, j), max(i, j))]
            for j in range(num_features)
            if j != i
        )

    return interactions_dict


class residualGame(Game):
    def __init__(self, n_players, game_values) -> None:
        super().__init__(n_players=n_players, normalize=False)
        self.vals = game_values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return self.vals


class RegressionMSR(Approximator):
    """ """

    def __init__(
        self,
        n: int,
        *,
        random_state: int | None = None,
        pairing_trick: bool = False,
        replacement: bool = True,
        sampling_weights: np.ndarray = None,
        regression_adjustment: bool = True,
        shapley_weighted_inputs: bool = False,
        residual_estimator: Approximator = None,
        index: str = "SV",
        max_order: int = 1,
        ablation_kwargs: dict = {
            "value_model": "xgb",
            "model_params": {},
            "sampling_weights": "default",
            "residual_approximator": "shapiq",
        },
    ) -> None:
        """Initialize the MonteCarlo approximator.

        Args:
            n: The number of players.

            random_state: The random state to use for the approximation. Defaults to ``None``.

            pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure.

            replacement: If ``True``, sampling is done with replacement. Defaults to ``True``.

            sampling_weights: An optional array of weights for the sampling procedure. The weights
                must be of shape ``(n + 1,)`` and are used to determine the probability of sampling
                a coalition of a certain size. Defaults to ``None``.
        """
        super().__init__(
            n,
            min_order=0,
            max_order=max_order,
            top_order=False,
            index=index,
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )
        validate_ablation_kwargs(ablation_kwargs)
        self.ablation_kwargs = ablation_kwargs
        # initialize sampler
        if ablation_kwargs["sampling_weights"] == "leverage":
            sampling_weights = np.ones(n + 1)

        if sampling_weights is None:  # init default sampling weights
            sampling_weights = self._init_sampling_weights()

        self._sampler = CoalitionSampler(
            n_players=self.n,
            sampling_weights=sampling_weights,
            pairing_trick=pairing_trick,
            replacement=replacement,
            random_state=self._random_state,
        )

        if residual_estimator is None:
            self.residual_estimator = get_ablation_residual_approximator(
                approximator_name=ablation_kwargs["residual_approximator"],
                n_players=n,
                max_order=max_order,
                index=index,
                pairing_trick=pairing_trick,
                replacement=replacement,
                sampling_weights=sampling_weights,
                random_state=random_state,
            )
        else:
            self.residual_estimator = residual_estimator

        self.value_model = get_ablation_model(
            model_name=self.ablation_kwargs["value_model"],
            model_params=self.ablation_kwargs["model_params"],
            random_state=self._random_state,
        )

        self.regression_adjustment = regression_adjustment
        self.shapley_weighted_inputs = shapley_weighted_inputs
        if shapley_weighted_inputs:
            self.shapley_weights = np.zeros(n + 1)
            for i in range(n + 1):
                if i == 0 or i == n:
                    self.shapley_weights[i] = 0
                else:
                    self.shapley_weights[i] = 1 / (binom(n - 2, i - 1))

        # init runtime dictionary of type float
        self.runtime_last_approximate_run: dict[str, float] = {}
        self.max_order = max_order

    def shapley_weight(self, coalition_size: int):
        return 1 / ((self.n) * binom(self.n - 1, coalition_size))

    def approximate(
        self, budget: int, game: Game | Callable[[np.ndarray], np.ndarray]
    ) -> InteractionValues:
        approximate_start_time = time.time()
        # sample with current budget
        self._sampler.sample(int(budget))
        coalitions_matrix = (
            self._sampler.coalitions_matrix
        )  # binary matrix of coalitions https://xgboost.readthedocs.io/en/stable/parameter.html
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = (
            sampling_end_time - approximate_start_time
        )
        # print("Runtime sampling: ", self.runtime_last_approximate_run["sampling"])

        # query the game for the current batch of coalitions
        game_values = game(coalitions_matrix)
        # print("GAME VALUES: ", game_values)
        baseline_value = game_values[0]
        game_values -= baseline_value

        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )
        # print("Runtime evaluations: ", self.runtime_last_approximate_run["evaluations"])
        # fit XGBoost regression model to the game values using coalition_matrix as input

        # Initialize the regressor
        # set weights for regression
        if self.shapley_weighted_inputs:
            coalition_weights = (
                self.shapley_weights[self._sampler.coalitions_size]
                * self._sampler.sampling_adjustment_weights
            )
            self.value_model.fit(
                coalitions_matrix, game_values, sample_weight=coalition_weights
            )
        else:
            self.value_model.fit(
                coalitions_matrix, game_values
            )
            #(self.value_model.predict(coalitions_matrix))

        # Fit the model
        # print("INPUTS TO REGRESSION: ", coalitions_matrix)
        # print("GAME VALUES TO REGRESSION: ", game_values)
        fit_time = time.time()
        self.runtime_last_approximate_run["proxy_fit"] = (
            fit_time - game_evaluation_end_time
        )
        # compute Shapley values of XGBoost
        explainer = InterventionalTreeExplainer(
                self.value_model,
                data=np.zeros((1, self.n)),
                class_index=None,
                debug=False,
                index=self.approximation_index,
                max_order=self.max_order,
        )
        interactions = explainer.explain_function_intervals(np.ones((1, self.n)))

        shapley_tree = InteractionValues(
            interactions.interactions,
            index=self.approximation_index,
            n_players=self.n,
            # interaction_lookup=self.interaction_lookup,
            min_order=self.min_order,
            max_order=self.max_order,
            baseline_value=baseline_value,
            estimated=not budget >= 2**self.n,
            estimation_budget=int(budget),
        )
        extraction_end_time = time.time()
        self.runtime_last_approximate_run["extraction"] = (
            extraction_end_time- fit_time
        )

        if self.regression_adjustment:
            # Predict on test set
            predicted_values = self.value_model.predict(coalitions_matrix)
            # compute the residual game values
            residual_values = game_values - predicted_values

            residual_game = residualGame(n_players=self.n, game_values=residual_values)
            shapley_residuals = self.residual_estimator.approximate(
                budget=budget, game=residual_game
            )
            # print("Runtime residual game: ", self.runtime_last_approximate_run["residual_game"])

            # reset empty set and baseline
            shapley_residuals.baseline_value = 0.0
            shapley_residuals[tuple()] = 0.0
            # print("SHAPLEY RESIDUALS: ", shapley_residuals.interactions)
            # print("> SUM TREE VALUES: ", shapley_tree.values.sum())
            # print("> SUM RESIDUAL VALUES: ", shapley_residuals.values.sum())
            shapley_value_estimates = shapley_tree + shapley_residuals
        else:
            shapley_value_estimates = shapley_tree

        residual_game_end_time = time.time()
        self.runtime_last_approximate_run["adjustment"] = (
                residual_game_end_time - extraction_end_time
        )

        regression_end_time = time.time()
        self.runtime_last_approximate_run["total"] = (
            regression_end_time - approximate_start_time
        )

        return shapley_value_estimates

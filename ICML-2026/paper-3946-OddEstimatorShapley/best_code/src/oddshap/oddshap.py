"""This module contains the Faithful Shapley-GAX approximator to compute the SV"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy.special import binom

from shapiq.approximator.base import Approximator
from shapiq.interaction_values import InteractionValues
from shapiq.utils.sets import powerset
from .proxyspex import proxyspex
from oddshap import InterventionalTreeExplainer
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from oddshap.proxyspex import lgboost_to_fourier, top_k_interactions

class ExplanationFrontierGenerator:
    def __init__(self, N: set):
        self.N = N
        self.n = len(N)

    def generate_kadd(self, max_order, sizes_to_exclude=None):
        explanation_basis = {}
        pos = 0
        for S in powerset(self.N, max_size=max_order):
            if sizes_to_exclude is None or len(S) not in sizes_to_exclude:
                explanation_basis[S] = pos
                pos += 1
        return explanation_basis

    def generate_prior(self, Q_prior):
        explanation_basis = {}
        pos = 0
        for S in Q_prior:
            explanation_basis[S] = pos
            pos += 1
        return explanation_basis

    def generate_partial(self, n_explanation_terms, sizes_to_exclude=None):
        explanation_basis = {}
        S_pos = 0
        # generate individual's basis
        for S in powerset(self.N, max_size=1):
            explanation_basis[S] = S_pos
            S_pos += 1

        # add interactions in random order
        perm = list(self.N)
        np.random.shuffle(perm)
        for S in powerset(self.N, min_size=2):
            if sizes_to_exclude is not None and len(S) in sizes_to_exclude:
                continue
            if S_pos < n_explanation_terms:
                explanation_basis[tuple(sorted([perm[i] for i in S]))] = S_pos
                S_pos += 1
            else:
                break
        return explanation_basis

def proxy_spex_top_fourier(game, n, top_k, samples, grid_search, odd, random_state, sample_weights=None):
    return [tuple(int(x) for x in k) for k in proxyspex(game, n, top_k=top_k, samples=samples, odd=odd, grid_search=grid_search, sample_weights=sample_weights, random_state=random_state).keys()]


class OddSHAP(Approximator):
    """Estimates the Shapley values using polynomial regression. Extends KernelSHAP.`_.

    Args:
        n: The number of players.
        explanation_frontier: A dictionary containing all interactions and their position.
        pairing_trick: If ``True``, the pairing trick is applied to the sampling procedure. Defaults
            to ``False``.
        sampling_weights: An optional array of weights for the sampling procedure. The weights must
            be of shape ``(n + 1,)`` and are used to determine the probability of sampling a coalition
            of a certain size. Defaults to ``None``.
        random_state: The random state of the estimator. Defaults to ``None``.

    Attributes:
        n: The number of players.
        N: The set of players (starting from ``0`` to ``n - 1``).
        max_order: The interaction order of the approximation.
        min_order: The minimum order of the approximation. For the regression estimator, min_order
            is equal to ``1``.
        iteration_cost: The cost of a single iteration of the regression SII.
    """

    def __init__(
        self,
        n: int,
        pairing_trick: bool = True,
        sampling_weights: np.ndarray | None = None,
        replacement: bool = False,
        random_state: int | None = None,
        regression_basis: str = "Fourier",
        interaction_detection: str = "ProxySPEX",
        odd_only: bool = True,
        grid_search: bool = False,
        interaction_factor: int = 10,
        shapley_weights: bool = False,
        tree_params: dict | None = None,
    ):
        super().__init__(
            n,
            min_order=0,
            max_order=1,
            index="SV",
            top_order=False,
            random_state=random_state,
            pairing_trick=pairing_trick,
            replacement=replacement,
            sampling_weights=sampling_weights,
        )
        self.regression_basis = regression_basis
        self.interaction_detection = interaction_detection
        self.odd_only = odd_only
        self.grid_search = grid_search
        self.shapley_weights = shapley_weights
        self.interaction_factor = interaction_factor
        self.random_state = random_state
        self.tree_params = tree_params
        # init runtime dictionary of type float
        self.runtime_last_approximate_run: dict[str, float] = {}

    def _init_kernel_weights(self) -> np.ndarray:
        """Initializes the kernel weights for the regression in KernelSHAP-IQ.

        The kernel weights are of size n + 1 and indexed by the size of the coalition. The kernel
        weights are set to a large number for the empty coalition and grand coalition.

        Returns:
            The weights for sampling subsets of size s in shape (n + 1,).
        """
        # vector that determines the kernel weights for KernelSHAPIQ
        weight_vector = np.zeros(shape=self.n + 1)
        for coalition_size in range(self.n + 1):
            if (coalition_size < 1) or (coalition_size > self.n - 1):
                # Constraints
                weight_vector[coalition_size] = 0
            else:
                weight_vector[coalition_size] = 1 / (
                    (self.n - 1) * binom(self.n - 2, coalition_size - 1)
                )
        kernel_weight = weight_vector
        return kernel_weight

    def approximate(
        self,
        budget: int,
        game: Callable[[np.ndarray], np.ndarray],
    ) -> InteractionValues:
        """The main approximation routine for the regression approximators.
        The regression estimators approximate Shapley Interactions based on a representation as a
        weighted least-square (WLSQ) problem. The regression approximator first approximates the
        objective of the WLSQ problem by Monte Carlo sampling and then computes an exact WLSQ
        solution based on the approximated objective. This approximation is an extension of
        KernelSHAP with different variants of kernel weights and regression settings.
        For details on KernelSHAP, refer to `Lundberg and Lee (2017) <https://doi.org/10.48550/arXiv.1705.07874>`_.

        Args:
            budget: The budget of the approximation.
            game: The game to be approximated.

        Returns:
            The `InteractionValues` object containing the estimated interaction values.
        """
        approximate_start_time = time.time()
        # initialize the kernel weights
        kernel_weights = self._init_kernel_weights()


        # get the coalitions
        self._sampler.sample(budget)
        sampling_end_time = time.time()
        self.runtime_last_approximate_run["sampling"] = (
            sampling_end_time - approximate_start_time
        )
        # query the game for the coalitions
        game_values = game(self._sampler.coalitions_matrix)
        game_evaluation_end_time = time.time()
        self.runtime_last_approximate_run["evaluations"] = (
            game_evaluation_end_time - sampling_end_time
        )

        # set number of interactions
        individuals = [()]
        for i in range(self.n):
            individuals.append((i,))
        n_interactions = max(0, budget // self.interaction_factor - self.n)

        sample_weights = np.sqrt(kernel_weights[self._sampler.coalitions_size]
            * self._sampler.sampling_adjustment_weights)
        sampling_normalization = sample_weights[2:]

        # normalize games
        empty_set_value = game_values[0]
        game_values -= empty_set_value
        full_set_value = game_values[1]

        if self.tree_params is not None:
            base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=self.random_state, **self.tree_params)
        else:
            # Set up LightGBM surrogate
            base_model = lgb.LGBMRegressor(verbose=-1, n_jobs=1, random_state=self.random_state, max_depth=10)

        if self.grid_search:
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
            grid_search.fit(self._sampler.coalitions_matrix, game_values)
            surrogate_model = grid_search.best_estimator_
        else:
            surrogate_model = base_model.fit(self._sampler.coalitions_matrix, game_values)

        ## print the number of trees in the best model
        print(f'Number of trees in best model: {surrogate_model.booster_.num_trees()}')

        proxy_fit_end_time = time.time()
        self.runtime_last_approximate_run["proxy_fit"] = (
                proxy_fit_end_time - game_evaluation_end_time
        )

        if budget < self.n*self.interaction_factor:
            # Directly extract Shapley values
            explainer = InterventionalTreeExplainer(
                    surrogate_model,
                    data=np.zeros((1, self.n)),
                    class_index=None,
                    debug=False,
                    index=self.approximation_index,
                    max_order=self.max_order,
            )
            interactions = explainer.explain_function_intervals(np.ones((1, self.n)))
            result = InteractionValues(
                interactions.interactions,
                index=self.approximation_index,
                n_players=self.n,
                # interaction_lookup=self.interaction_lookup,
                min_order=self.min_order,
                max_order=self.max_order,
                baseline_value=empty_set_value,
                estimated=not budget >= 2**self.n,
                estimation_budget=int(budget),
            )
            interaction_detection_end_time = time.time()
            self.runtime_last_approximate_run["extraction"] = (
                interaction_detection_end_time - proxy_fit_end_time
            )
            regression_end_time = time.time()
            self.runtime_last_approximate_run["regression"] = (
                regression_end_time - interaction_detection_end_time
            )
        else:
            # detect interactions and perform oddshap regression
            if self.interaction_detection == "Random":
                generator = ExplanationFrontierGenerator(self._grand_coalition_set)
                self.explanation_frontier = generator.generate_partial(self.n+n_interactions,sizes_to_exclude=[2,4,6,8,10,12,14,16,18,20])
                self.n_variables = len(self.explanation_frontier) - 1
                # Extend interaction_lookup with pre-defined interactions
                self.interaction_matrix_binary = np.zeros(
                    (len(self.explanation_frontier), self.n), dtype=bool
                )
                self.odd_interaction_lookup = {}
                self.odd_interaction_matrix_binary = np.zeros((len(self.explanation_frontier), self.n), dtype=bool)
                for S, pos in self.explanation_frontier.items():
                    self.odd_interaction_lookup[S] = pos
                    self.odd_interaction_matrix_binary[pos, S] = True
            if self.interaction_detection == "ProxySPEX":
                # use ProxySPEX on sampled coalitions
                initial_transform = lgboost_to_fourier(surrogate_model.booster_.dump_model())
                proxyspex_output = top_k_interactions(initial_transform, n_interactions, odd=self.odd_only)
                detected_interactions = [tuple(int(x) for x in k) for k in proxyspex_output.keys()]
                #new_interactions = proxy_spex_top_fourier(game, self.n, n_interactions, samples=(self._sampler.coalitions_matrix, game_values),odd=self.odd_only, grid_search=self.grid_search, random_state=self.random_state)
                prior = individuals + detected_interactions
                generator = ExplanationFrontierGenerator(self._grand_coalition_set)
                self.explanation_frontier = generator.generate_prior(prior)
                self.n_variables = len(self.explanation_frontier) -1 # exclude empty set
                #self.n_variables = len(self.explanation_frontier)
                # Extend interaction_lookup with pre-defined interactions
                self.interaction_matrix_binary = np.zeros(
                    (len(self.explanation_frontier), self.n), dtype=bool
                )
                self.odd_interaction_lookup = {}
                self.odd_interaction_matrix_binary = np.zeros((len(self.explanation_frontier),self.n),dtype=bool)
                for S, pos in self.explanation_frontier.items():
                    self.odd_interaction_lookup[S] = pos
                    self.odd_interaction_matrix_binary[pos, S] = True

            interaction_detection_end_time = time.time()
            self.runtime_last_approximate_run["extraction"] = (
                interaction_detection_end_time - proxy_fit_end_time
            )

            X_tilde = self._create_design_matrix(sampling_normalization, budget)

            # Center the game values around the Fourier intercept (per Eq 20)
            centered_game_values = game_values[2:] - 0.5 * full_set_value

            # Compress paired rows and construct odd targets if applicable
            # (halves regression cost with pairing trick)
            X_tilde, y_tilde = self._compress_paired_rows(
                X_tilde,
                centered_game_values,
                sampling_normalization
            )

            # interactions ordered by regression column position (excluding empty set at pos 0)
            interactions_by_pos = [None] * (self.n_variables + 1)
            for S, pos in self.odd_interaction_lookup.items():
                interactions_by_pos[pos] = S
            active_interactions = interactions_by_pos[1:] # exclude empty set

            if self.regression_basis == "Moebius":
                a = np.ones(self.n_variables, dtype=float)
                b = full_set_value
            elif self.regression_basis == "Fourier":
                # a_j = 1 if interaction j has odd size, 0 otherwise
                if not self.odd_only:
                    a_even = np.array([(len(S) % 2) == 0 for S in active_interactions], dtype=float)
                    b_even = 0
                a_odd = np.array([(len(S) % 2) == 1 for S in active_interactions], dtype=float)
                b_odd = -0.5 * full_set_value

            if not self.odd_only:
                den_even = float(a_even @ a_even)
                if den_even == 0:
                    den_even = 1
                self.projection_matrix_even = np.eye(self.n_variables) - np.outer(a_even, a_even) / den_even
                constant_even = (b_even / den_even) * a_even # minimum-norm point satisfying a^T beta = b

            den_odd = float(a_odd @ a_odd)

            self.projection_matrix_odd = np.eye(self.n_variables) - np.outer(a_odd, a_odd) / den_odd
            constant_odd = (b_odd / den_odd) * a_odd  # minimum-norm point satisfying a^T beta = b


            interaction_representation = np.zeros(self.n_variables + 1, dtype=float)
            interaction_representation[0] = empty_set_value
            if not self.odd_only:
                lstsq_solution = np.linalg.lstsq(
                    X_tilde @ self.projection_matrix_odd @ self.projection_matrix_even ,
                    y_tilde - X_tilde @ (constant_odd + constant_even),
                    rcond=None,
                )[0]
                interaction_representation[
                    1:] = constant_odd + constant_even + self.projection_matrix_odd @ self.projection_matrix_even @ lstsq_solution

            else:
                lstsq_solution = np.linalg.lstsq(
                    X_tilde @ self.projection_matrix_odd,
                    y_tilde - X_tilde @ (constant_odd),
                    rcond=None,
                )[0]
                interaction_representation[
                    1:] = constant_odd + self.projection_matrix_odd @ lstsq_solution

            #interaction_representation =  constant_odd + constant_even + self.projection_matrix_odd @ self.projection_matrix_even @ lstsq_solution

            # self.projection_matrix = np.identity(self.n_variables) - 1 / self.n_variables
            # if self.regression_basis == "Moebius":
            #     constant = full_set_value / self.n_variables
            # if self.regression_basis == "Fourier":
            #     constant = -(1 / 2) * full_set_value / self.n_variables
            #
            # lstsq_solution = np.linalg.lstsq(
            #     X_tilde @ self.projection_matrix,
            #     y_tilde - constant * np.sum(X_tilde, axis=1),
            #     rcond=None,
            # )[0]
            #
            #
            # interaction_representation = np.zeros(self.n_variables + 1, dtype=float)
            # interaction_representation[0] = empty_set_value
            # interaction_representation[1:] = constant + lstsq_solution

            # Transform to Shapley values
            sv, sv_lookup = self._transform_to_shapley(interaction_representation)

            regression_end_time = time.time()
            self.runtime_last_approximate_run["regression"] = (
                regression_end_time - interaction_detection_end_time
            )

            # Transform the output to the interaction values
            result = InteractionValues(
                values=sv,
                index="SV",
                interaction_lookup=sv_lookup,
                baseline_value=empty_set_value,
                min_order=self.min_order,
                max_order=self.max_order,
                n_players=self.n,
                estimated=not budget >= 2**self.n,
                estimation_budget=budget,
            )
        shapiq_post_processing_end_time = time.time()
        self.runtime_last_approximate_run["shapiq_post_processing"] = (
                shapiq_post_processing_end_time - regression_end_time
        )
        self.runtime_last_approximate_run["total"] = (
            shapiq_post_processing_end_time - approximate_start_time
        )
        return result

    def _transform_to_shapley(self, input_values):
        sv = np.zeros(self.n + 1)
        sv_lookup = {}
        if self.regression_basis == "Moebius":
            for interaction, interaction_pos in self.odd_interaction_lookup.items():
                if len(interaction) == 0:
                    sv[0] = input_values[interaction_pos]
                    sv_lookup[()] = 0
                for i in interaction:
                    sv[i + 1] += input_values[interaction_pos] / len(interaction)
                    sv_lookup[(i,)] = i + 1
        if self.regression_basis == "Fourier":
            for interaction, interaction_pos in self.odd_interaction_lookup.items():
                if len(interaction) == 0:
                    sv[0] = input_values[interaction_pos]
                    sv_lookup[()] = 0
                if (len(interaction) % 2) == 1:  # only odd interactions matter
                    for i in interaction:
                        sv[i + 1] += input_values[interaction_pos] / len(interaction)
                        sv_lookup[(i,)] = i + 1
            sv *= -2  # adjust for Fourier basis
        return sv, sv_lookup

    def _create_design_matrix(self, weights, budget):
        X_tilde = np.zeros((budget - 2, self.n_variables))
        coalitions = self._sampler.coalitions_matrix[
            2:, :
        ]  # drop empty and grand coalition
        interactions = self.odd_interaction_matrix_binary[1:, :]  # drop empty coalition
        #interactions = self.odd_interaction_matrix_binary

        if self.regression_basis == "Moebius":
            if self.n_variables > self.n:  # interactions available
                for pos, row in enumerate(interactions):
                    X_tilde[:, pos] = np.all(row <= coalitions, axis=1) * weights
            else:
                X_tilde = coalitions * weights[:, np.newaxis]

        elif self.regression_basis == "Fourier":
            # Precompute once (outside hot loop)
            coalitions_int = coalitions.astype(np.uint8)
            interactions_T_int = interactions.T.astype(np.uint8)
            # parity: 0/1 matrix of (dot % 2)
            parity = ((coalitions_int @ interactions_T_int) & 1).astype(np.int8)
            # map 0->1, 1->-1 and apply weights
            X_tilde = (1 - 2 * parity) * weights[:, np.newaxis]
            # X_tilde = (
            #     1 - 2 * ((coalitions.astype(int) @ interactions.T.astype(int)) % 2)
            # ) * weights[:, np.newaxis]
        else:
            raise ValueError(f"Basis {self.regression_basis} not recognized.")

        return X_tilde

    def _compress_paired_rows(
            self,
            X_tilde: np.ndarray,
            game_values_reg: np.ndarray,
            weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Efficiently compresses paired regression rows when pairing trick is active.

        For Fourier basis with odd_only=True and pairing_trick=True, paired coalitions form
        antipodal design rows. We can keep only one representative per pair, compute the exact
        odd target, and halve the regression size.
        """
        # Only compress if all conditions are met
        if not (self._sampler.pairing_trick and self.regression_basis == "Fourier" and self.odd_only):
            # Fallback: standard uncompressed targets
            return X_tilde, game_values_reg * weights

        # Retrieve the pairing metadata from the sampler
        paired_full_idx = self._sampler.paired_coalition_index
        is_rep = self._sampler.paired_coalition_representative

        if paired_full_idx is None or is_rep is None:
            # Fallback if pairing metadata was not computed
            return X_tilde, game_values_reg * weights

        # Adjust for the fact that X_tilde and game_values_reg skip the empty (0) and grand (1) coalitions
        # Slicing [2:] aligns the metadata perfectly with the regression arrays.
        paired_full_idx = paired_full_idx[2:]
        is_rep = is_rep[2:]

        # Representatives that have a valid partner (assumed all reps are paired)
        paired_rep_mask = is_rep & (paired_full_idx != -1)

        # If metadata exists but mask is empty, return empty compressed arrays
        if not np.any(paired_rep_mask):
            return np.empty((0, X_tilde.shape[1]), dtype=X_tilde.dtype), np.empty(
                (0,), dtype=game_values_reg.dtype
            )

        # Partner indices in local coordinates (regression arrays start at full index 2)
        Sc_local_idx = paired_full_idx[paired_rep_mask] - 2

        # Oddified targets: f_odd(S) = (f(S) - f(S^c)) / 2
        y_odd = 0.5 * (
            game_values_reg[paired_rep_mask] - game_values_reg[Sc_local_idx]
        )

        # Compressed design/targets
        X_compressed = X_tilde[paired_rep_mask]
        y_compressed = y_odd * weights[paired_rep_mask]

        return X_compressed, y_compressed


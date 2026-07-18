from __future__ import annotations

from copy import deepcopy

from torch.mtia import device
from xac.acquisition_optimizers.acquisition_optimizers import (
    BaseAcquisitionOptimizer, Exhaustive, Subset)

"""Abstract application class and various concrete implementations.

* ``BaseApplication`` enforces that every concrete application must
  lazily create three tensors:

  - **Z** : execution path input sequence
  - **A** : affine transformation matrix
  - **X0**: initial design / archive

The attributes are *derived* from the user‑supplied constructor arguments,
so they are declared with ``init=False`` and filled inside
``__post_init__``.
"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import botorch
import gpytorch
import numpy as np
import scipy
import shapiq
import torch
from linear_operator.utils.cholesky import psd_safe_cholesky
from shapiq.approximator import SVARM, KernelSHAP, PermutationSamplingSV

from xac.blackbox_functions import (BaseBlackboxFunction, BotorchTestFunction,
                                    TabRepoBenchmark)

from .polyshap import ExplanationFrontierGenerator, PolySHAP
from .regressionMSR import RegressionMSR


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Abstract parent
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseApplication(ABC):
    """Abstract base class for all applications."""

    eval_bb_only_on_Z: bool = field(init=False)
    lazy_setup: bool = field(init=False, default=False)

    # ---------------------------------------------------------------------
    # properties every subclass must implement
    # ---------------------------------------------------------------------
    @property
    @abstractmethod
    def Z(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def A(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def X0(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def candidate_set(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def candidate_idx_Z(self) -> torch.Tensor:
        # Indices of the elements in the candidate set in Z
        pass

    @abstractmethod
    def termination_criterion(self, property_posterior) -> bool:
        """Return True if the termination condition for this application is met."""

    @abstractmethod
    def run_lazy_setup(self, blackbox_function):
        """Setup class lazily. Must be called from outside."""
        pass

    # ------------------------------------------------------------------
    # post‑init & lazy setup
    # ------------------------------------------------------------------
    def __post_init__(self):
        # Maybe we need this in a later application
        if not self.lazy_setup:
            object.__setattr__(self, "Z", self._generate_Z())
            object.__setattr__(self, "A", self._generate_A())
            object.__setattr__(self, "X0", self._generate_X0())

        else:
            object.__setattr__(self, "lazy_setup_conducted", False)

    # ------------------------------------------------------------------
    # Compute posterior of the property for a given surrogate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def property_posterior(
        self, surrogate, noisy_variant: bool = False, recycle: bool = True
    ):
        """Return the posterior over the property using a fitted surrogate.

        Notes
        -----
        Should not be stored as part of the dataclass, since it may change over time.
        """
        if self.lazy_setup:
            assert self.lazy_setup_conducted, f"run_lazy_setup() must be called first."

        def _property_posterior(_surrogate, _noisy_variant):
            # Alternative efficient implementation
            if hasattr(self, "compute_AF_PPM") and hasattr(self, "compute_AF_PPV"):
                if _noisy_variant:
                    raise NotImplementedError(
                        "Noisy variant not implemented for AF_PPM."
                    )
                else:
                    # Efficient implementation
                    # A_KZX = (
                    #     self.A_KZX
                    #     if hasattr(self, "A_KZX") and recycle
                    #     else self.compute_A_KZX(_surrogate)
                    # )
                    A_KZX = self.compute_A_KZW_new(W= _surrogate._model.train_inputs[0],
                                                        surrogate= _surrogate)


                    # if not hasattr(self, "A_KZX"):
                    #     print("wait")

                    # A_KZX = self.compute_A_KZX(_surrogate)
                    PPM = self.compute_AF_PPM(_surrogate, precomputed_A_KZX=A_KZX)

                    PPV = (
                        self.scale_AEA(self.AEA_unscaled, _surrogate)
                        if hasattr(self, "AEA_unscaled") and recycle
                        else self.compute_AF_PPV(_surrogate, precomputed_A_KZX=A_KZX)
                    )  # AEA already computed, only needs to be rescaled
                    # PPV = self.compute_AF_PPV(_surrogate, precomputed_A_KZX=A_KZX)

                    compare_to_old = False
                    if compare_to_old:
                        PPM_GT = (
                            self.A
                            @ _surrogate.forward(
                                self.Z, observation_noise=_noisy_variant
                            ).mean
                        )
                        assert torch.allclose(PPM, PPM_GT, atol=1e-12)

                        mvn_lazy_covar = _surrogate.forward_lazy_covar(
                            self.Z, observation_noise=_noisy_variant
                        )
                        PPV_GT = self.A @ mvn_lazy_covar.matmul(self.A.T)
                        assert torch.allclose(PPV, PPV_GT, atol=1e-12)

                    try:
                        mvn = gpytorch.distributions.MultivariateNormal(PPM, PPV)

                    except Exception:
                        try:
                            PPV_stable = self.compute_AEA(
                                _surrogate,
                                scale_by_emp_std=True,
                                force_stable=True,
                            )
                            mvn = gpytorch.distributions.MultivariateNormal(
                                PPM, PPV_stable
                            )
                            log.warning(
                                "PPV was not PD with fast A*K path. Recomputed PPV with stable A*K fallback."
                            )
                        except Exception:
                            mvn = torch.distributions.MultivariateNormal(
                                PPM, torch.diag(torch.ones((PPM.shape[0])))
                            )
                            log.warning(
                                "Resorted to diagonal covariance due to non positive-definiteness issues in efficient AF posterior computation. This is not accurate!"
                            )

                        # #with gpytorch.settings.cholesky_max_tries(15):
                        # L = psd_safe_cholesky(PPV)
                        # mvn= torch.distributions.MultivariateNormal(loc=PPM, scale_tril=L)

                    return mvn
                    # return gpytorch.distributions.MultivariateNormal(PPM, PPV)

            else:
                mvn_mean = _surrogate.forward(
                    self.Z, observation_noise=_noisy_variant
                ).mean
                mvn_lazy_covar = _surrogate.forward_lazy_covar(
                    self.Z, observation_noise=_noisy_variant
                )

                if mvn_lazy_covar.ndim == 3:
                    mean = (self.A @ mvn_mean.T).T
                else:
                    mean = self.A @ mvn_mean

                cov = self.A @ mvn_lazy_covar.matmul(self.A.T)
                # DefaultCPUAllocator: can't allocate memory: you tried to allocate 68719476736 bytes. Error code 12 (Cannot allocate memory)

                return gpytorch.distributions.MultivariateNormal(mean, cov)
                # Might return a Gaussian mixture (if ndim=3 and leading dimension > 3)

        try:
            return _property_posterior(surrogate, noisy_variant)

        except:
            # Force jitter on diagonal, as this can avoid non positive-definiteness issues
            log.info(
                "Added jitter on diagonal (leading to PPD for y-Z) to avoid positive-definiteness issues."
            )
            return _property_posterior(surrogate, True)

    # ------------------------------------------------------------------
    # utility: dtype / device transfer
    # ------------------------------------------------------------------
    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        """Recursively move stored tensors to *device* / *dtype*."""
        if self.lazy_setup:
            assert self.lazy_setup_conducted, f"run_lazy_setup() must be called first."

        if device is not None or dtype is not None:
            self.Z = self.Z.to(device=device, dtype=dtype)
            self.A = self.A.to(device=device, dtype=dtype)
            self.X0 = self.X0.to(device=device, dtype=dtype)
        return self


# -----------------------------------------------------------------------------
# Application – Partial dependence plots
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PDPApplication(BaseApplication):
    # PDP for 1d marginal effects

    dim_of_interest: int = 0
    dim_of_interest_samples: int = (
        20  # How many samples to draw for dimension of interest
    )

    marginalized_dims_samples: int = (
        50  # How many samples to draw in total (per grid point) from all marginalized dimensions
    )

    init_design_dim_factor: int = 5

    eval_bb_only_on_Z: bool = field(init=False, default=True)
    lazy_setup: bool = field(init=False, default=True)

    def run_lazy_setup(
        self,
        blackbox_function: BotorchTestFunction,
        seed: int,
        amount_iterations: int = None,
        scalable_mode: bool = False,
        acquisition_optimizer: BaseAcquisitionOptimizer = None,
    ):
        object.__setattr__(self, "blackbox_function", blackbox_function)

        # Generate an equidistant set of grid  points for the dimension of interest
        dim_of_interest_bounds = self.blackbox_function.get_bounds_for_dim(
            dim=self.dim_of_interest
        )
        dim_of_interest_grid = torch.unsqueeze(
            torch.linspace(
                dim_of_interest_bounds[0],
                dim_of_interest_bounds[1],
                self.dim_of_interest_samples + 2,
            )[1:-1],
            dim=1,
        )

        # Draw marginalized_dims_samples * dims Sobol samples for the other dimensions
        amount_marginalized_dims = self.blackbox_function.dim - 1

        sobol = torch.quasirandom.SobolEngine(
            dimension=amount_marginalized_dims, scramble=True, seed=seed
        )  # To ensure different grids across seeds
        marginalized_dims_unscaled_grid = sobol.draw(
            self.marginalized_dims_samples  # * amount_marginalized_dims
        )

        marginalized_dims_idx = torch.arange(self.blackbox_function.dim)
        marginalized_dims_idx = marginalized_dims_idx[
            marginalized_dims_idx != self.dim_of_interest
        ]

        marginalized_dims_lower_bounds = torch.unsqueeze(
            self.blackbox_function.bounds[0, marginalized_dims_idx], dim=0
        )
        marginalized_dims_upper_bounds = torch.unsqueeze(
            self.blackbox_function.bounds[1, marginalized_dims_idx], dim=0
        )
        marginalized_dims_grid = (
            marginalized_dims_lower_bounds
            + (marginalized_dims_upper_bounds - marginalized_dims_lower_bounds)
            * marginalized_dims_unscaled_grid
        )

        # List with inputs of execution path per dimension in M
        execution_path_separated = []
        for dim_of_interest_grid_idx in range(dim_of_interest_grid.shape[0]):
            # For each of M dims span the execution path
            execution_path_dim_subset = torch.zeros(
                (marginalized_dims_grid.shape[0], self.blackbox_function.dim),
                dtype=float,
            )

            execution_path_dim_subset[:, self.dim_of_interest] = dim_of_interest_grid[
                dim_of_interest_grid_idx, :
            ]
            execution_path_dim_subset[:, marginalized_dims_idx] = marginalized_dims_grid

            execution_path_separated.append(execution_path_dim_subset)

            assert torch.all(
                execution_path_dim_subset[:, self.dim_of_interest]
                == execution_path_dim_subset[:, self.dim_of_interest].flatten()[0]
            )  # Assert all values are identical for dim_of_interest
            assert torch.any(
                execution_path_dim_subset[:, marginalized_dims_idx]
                != execution_path_dim_subset[:, marginalized_dims_idx].flatten()[0]
            )  # Assert other values vary

            for temp_dim_idx in torch.arange(self.blackbox_function.dim):
                temp_slice = execution_path_dim_subset[:, temp_dim_idx]
                lower, upper = self.blackbox_function.get_bounds_for_dim(
                    dim=temp_dim_idx
                )
                assert torch.all((temp_slice >= lower) & (temp_slice <= upper))

        Z = torch.stack(execution_path_separated).reshape(
            -1, self.blackbox_function.dim
        )

        # Generate affine transformation
        A = torch.zeros((self.dim_of_interest_samples, Z.shape[0]), dtype=float)
        for dim_of_interest_grid_idx in range(A.shape[0]):
            A[
                dim_of_interest_grid_idx,
                dim_of_interest_grid_idx
                * marginalized_dims_grid.shape[0] : (dim_of_interest_grid_idx + 1)
                * marginalized_dims_grid.shape[0],
            ] = (
                1 / marginalized_dims_grid.shape[0]
            )

        # Select size of initial design based on dimensionality
        init_design_size = self.init_design_dim_factor * self.blackbox_function.dim
        rand_perm = torch.randperm(Z.shape[0])
        X0 = Z[rand_perm[:init_design_size], :]
        candidate_set = Z[rand_perm[init_design_size:], :]
        candidate_idx_Z = rand_perm[init_design_size:]

        # Required overrides
        object.__setattr__(self, "_Z", Z)
        object.__setattr__(self, "_A", A)
        object.__setattr__(self, "_X0", X0)
        object.__setattr__(self, "_candidate_set", candidate_set)
        object.__setattr__(self, "_candidate_idx_Z", candidate_idx_Z)

        if self.blackbox_function.is_pseudo_expensive:
            f_Z_gt = self.blackbox_function.evaluate(X=self.Z)
            object.__setattr__(self, "f_Z_gt", f_Z_gt)

            prop_gt = self.A @ self.f_Z_gt[0]
            object.__setattr__(self, "prop_gt", prop_gt)

        object.__setattr__(self, "lazy_setup_conducted", True)

    # ---------------- required overrides -----------------------------
    @property
    def Z(self) -> str:
        return self._Z

    @property
    def A(self) -> str:
        return self._A

    @property
    def X0(self) -> str:
        return self._X0

    @property
    def candidate_set(self) -> str:
        return self._candidate_set

    @property
    def candidate_idx_Z(self) -> str:
        return self._candidate_idx_Z

    def termination_criterion(self, property_posterior) -> torch.Tensor:
        if self.lazy_setup:
            assert self.lazy_setup_conducted, f"run_lazy_setup() must be called first."

        return False  # Not implemented


# -----------------------------------------------------------------------------
# Application – Efficient Benchmarking
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TabRepoBenchmarkApplication(BaseApplication):
    """Concrete implementation for the TabRepo Benchmarking setting."""

    amount_challenger_configs: int = 1
    amount_init_design_configs: int = 5
    amount_random_init_design_samples: int = (
        0  # As opposed to fully evaluated configs in init design, simply add random samples from dataset
    )

    # The evaluation scope of the blackbox function is restricted to Z.
    eval_bb_only_on_Z: bool = field(init=False, default=True)
    lazy_setup: bool = field(init=False, default=True)

    # ---------------- required overrides -----------------------------
    def run_lazy_setup(
        self,
        blackbox_function: TabRepoBenchmark,
        seed: int,
        amount_iterations: int = None,
        scalable_mode: bool = False,
        acquisition_optimizer: BaseAcquisitionOptimizer = None,
    ):
        object.__setattr__(self, "blackbox_function", blackbox_function)

        # ---------------- asserts -----------------------------
        assert (
            self.amount_challenger_configs == 1
        ), f"Currently only supports a single challenger config."

        assert (
            self.amount_init_design_configs > 0
        ), f"Currently an initial design is required (at least 1, as this contains default)."

        # ------------------------------------------------------------------
        # 1. Randomly select configs
        # ------------------------------------------------------------------
        candidate_config_ids = self.blackbox_function.dataset[
            :, self.blackbox_function.config_id_idx
        ].unique()

        config_ids = candidate_config_ids[
            torch.randperm(len(candidate_config_ids))[
                : self.amount_init_design_configs + self.amount_challenger_configs
            ]
        ]
        config_data = [
            blackbox_function.dataset[
                blackbox_function.dataset[:, blackbox_function.config_id_idx]
                == temp_config_id
            ]
            for temp_config_id in config_ids
        ]

        config_dataset_ids = [
            config_data[i][:, blackbox_function.dataset_id_idx]
            for i in range(len(config_data))
        ]
        # Ensure that datasets are always identical
        assert all(
            [
                (
                    config_dataset_ids[0].sort()[0] == config_dataset_ids[i].sort()[0]
                ).all()
                for i in range(len(config_dataset_ids))
            ]
        )

        init_design_config_idx = config_ids[:-1]
        execution_path_config_idx = config_ids[-2:]
        candidate_set_config_idx = config_ids[-1]

        init_design_data = torch.concat(config_data[:-1])

        # ------------------------------------------------------------------
        # 2. If specified, add random samples from remaining dataset to initial design
        # ------------------------------------------------------------------

        if self.amount_random_init_design_samples > 0:
            config_data_complement = blackbox_function.dataset[
                ~torch.isin(
                    blackbox_function.dataset[:, blackbox_function.config_id_idx],
                    config_ids,
                )
            ]

            rows = config_data_complement.size(0)
            random_init_design_samples = config_data_complement[
                torch.randperm(rows)[: self.amount_random_init_design_samples]
            ]

            init_design_data = torch.concat(
                [init_design_data, random_init_design_samples]
            )

        init_design_x = init_design_data[:, blackbox_function.indep_attr_idx]
        init_design_y_perf = init_design_data[:, blackbox_function.perf_metric_idx]
        init_design_y_cost = init_design_data[:, blackbox_function.cost_metric_idx]

        execution_path_data = torch.concat(config_data[-2:])
        execution_path_x = execution_path_data[:, blackbox_function.indep_attr_idx]
        execution_path_y_perf = execution_path_data[
            :, blackbox_function.perf_metric_idx
        ]
        execution_path_y_cost = execution_path_data[
            :, blackbox_function.cost_metric_idx
        ]

        candidate_set_data = config_data[-1]
        candidate_set_x = candidate_set_data[:, blackbox_function.indep_attr_idx]
        candidate_set_y_perf = candidate_set_data[:, blackbox_function.perf_metric_idx]
        candidate_set_y_cost = candidate_set_data[:, blackbox_function.cost_metric_idx]

        object.__setattr__(self, "_X0", init_design_x)
        object.__setattr__(self, "Y0", (init_design_y_perf, init_design_y_cost))

        object.__setattr__(self, "_Z", execution_path_x)
        object.__setattr__(
            self, "f_Z_gt", (execution_path_y_perf, execution_path_y_cost)
        )

        object.__setattr__(self, "_candidate_set", candidate_set_x)
        object.__setattr__(
            self,
            "_candidate_idx_Z",
            torch.arange(
                config_data[-2].shape[0],
                config_data[-2].shape[0] + config_data[-1].shape[0],
            ),
        )  # torch.arange()

        object.__setattr__(
            self, "_A", self._generate_A(int(execution_path_data.shape[0] / 2))
        )

        if blackbox_function.is_pseudo_expensive:
            prop_gt = self.A @ self.f_Z_gt[0]
            object.__setattr__(self, "prop_gt", prop_gt)

        object.__setattr__(self, "lazy_setup_conducted", True)

    @property
    def Z(self) -> str:
        return self._Z

    @property
    def A(self) -> str:
        return self._A

    @property
    def X0(self) -> str:
        return self._X0

    @property
    def candidate_set(self) -> str:
        return self._candidate_set

    @property
    def candidate_idx_Z(self) -> str:
        return self._candidate_idx_Z

    def _generate_A(self, amount_instances) -> torch.Tensor:
        assert (
            self.amount_challenger_configs == 1
        ), f"Currently only supports comparing two configs."

        return torch.cat(
            [
                torch.full((1, amount_instances), 1.0 / amount_instances),
                torch.full((1, amount_instances), -1.0 / amount_instances),
            ],
            dim=-1,
        ).to(torch.float64)
        # Mean performance challenger minus mean performance incumbent => >0 corresponds to challenger is better

    def termination_criterion(self, property_posterior) -> bool:
        if self.lazy_setup:
            assert self.lazy_setup_conducted, f"run_lazy_setup() must be called first."

        return False  # Not implemented


# Define game
# Calls blackbox function internally on sampled coalitions
class ShapIqGame(shapiq.Game):
    def __init__(self, m, surrogate, archive_size, blackbox_fn, exact=False) -> None:
        super().__init__(
            n_players=m,
            player_names=[str(i) for i in range(m)],
        )  # normalization_value=self.characteristic_function[()],  # 0

        self.surrogate = surrogate
        self.archive_size = archive_size
        self.blackbox_fn = blackbox_fn
        self.exact = exact

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Defines the worth of a coalition as a lookup in the characteristic function.

        Args:
            coalitions: A 2D array where each row represents a coalition as a binary
                vector (1 for present, 0 for absent).

        Returns:
            A 1D array containing the value of each coalition based on the
                characteristic function.
        """
        # torch.tensor(coalitions, dtype= torch.int64)
        coalitions_int = torch.tensor(coalitions, dtype=torch.int64)
        coalitions_numeric = self.surrogate._model.input_transform.untransform(
            coalitions_int
        )

        output = self.blackbox_fn(coalitions_numeric)[0].squeeze()

        assert (
            self.surrogate._model.input_transform.transform(coalitions_numeric)
            == coalitions_int
        ).all(), "Siq coalition transformation does not match input."
        # assert torch.allclose(self.archive_X, coalitions_numeric), "Siq coalition numeric values do not match archive X."

        # if not self.exact:
        #     assert self.archive_size == coalitions_numeric.shape[0], "." #has to be disabled as PermutationSamplingSV evaluates coalitions one by one
        # #i think we dont really need archive xevaluates

        return np.array([output] if output.ndim == 0 else output)


@dataclass(frozen=True)
class ShapleyApplication(BaseApplication):
    """Concrete implementation for Shapley value estimation."""

    # The evaluation scope of the blackbox function is restricted to Z.
    eval_bb_only_on_Z: bool = field(init=False, default=True)
    lazy_setup: bool = field(init=False, default=True)

    init_design_factor: int = 2  # Initial design size is init_design_factor * m
    init_design_size: int = False # Computed in lazy setup based on dimensionality
    random_init_design: bool = False  # Random init design option for ablation study

    # ---------------- required overrides -----------------------------
    def run_lazy_setup(
        self,
        blackbox_function: BaseBlackboxFunction,
        seed: int,
        amount_iterations: int = None,
        scalable_mode: bool = False,
        acquisition_optimizer: BaseAcquisitionOptimizer = None,
    ):

        object.__setattr__(self, "blackbox_function", blackbox_function)
        baseline_config, candidate_config = self.sample_configs()
        m = self.get_blackbox_dim()

        # Overall preparations
        # Initialize ShapIQ approximator
        # changes here should be mirrored in get_siq_approximation()

        # Modified to LeverageSHAP
        frontier_generator = ExplanationFrontierGenerator(N=[i for i in range(m)])
        explanation_frontier = frontier_generator.generate_kadd(max_order=1)
        sampling_weights_1 = np.ones(m + 1)

        siq_approximator = PolySHAP(
            n=m,
            explanation_frontier=explanation_frontier,
            sampling_weights=sampling_weights_1,
            pairing_trick=True,  # replacement= False,
            random_state=seed,
        )
        # siq_approximator = KernelSHAP(n=m,
        #                               index= 'SV',
        #                               max_order= 1,
        #                               random_state= seed)

        object.__setattr__(self, "siq_approximator", siq_approximator)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "m", m)

        object.__setattr__(self, "baseline_config", baseline_config)
        object.__setattr__(self, "candidate_config", candidate_config)

        # Initial design
        # assert amount_iterations is not None, "amount_iterations must be specified for ShapleyApplication lazy setup."
        # amount_samples= amount_iterations + self.init_design_factor * m

        if self.init_design_size:
            init_design_size= self.init_design_size
            #Small init designs might lead to bad performance and that competitors cant be comppared

        else:
            init_design_size = max(
                self.init_design_factor * m, m + 1
            )  # At least m+1 samples for numerical stability
        object.__setattr__(self, "init_design_size", init_design_size)

        # siq_approximator._sampler.sample(3)
        # siq_approximator._sampler.coalitions_matrix

        object.__setattr__(self, "amount_players", self.m)
        object.__setattr__(self, "amount_coalitions", 2**self.m)

        object.__setattr__(self, "scalable_mode", scalable_mode)

        if scalable_mode:
            # assert not isinstance(
            #     acquisition_optimizer, Exhaustive
            # ), "Exhaustive acquisition optimization not compatible with scalable_mode."

            if isinstance(acquisition_optimizer, Subset):
                candidate_size = acquisition_optimizer.subset_size

                assert candidate_size < 2**self.m  - init_design_size, "Subset acquisition optimization cannot use a candidate set larger than the full coalition space."

            elif isinstance(acquisition_optimizer, Exhaustive):
                assert self.m <= 16, "Exhaustive acquisition optimization not compatible with more than 16 players due to computational constraints."

                candidate_size = 2**self.m - init_design_size

            else:
                raise NotImplementedError()

            assert (
                candidate_size >= amount_iterations
            ), "Candidate subset size must be larger than amount_iterations."


            # Ensure that initial design is not from border trick samples (due to performance deterioration)
            sampling_budget = init_design_size + candidate_size

            if self.random_init_design:
                log.warning(
                    "Using random initial design. This can lead to bad performance, but is used for ablation purposes."
                )
                #Generate a torch tensor of random bools with shape (init_design_size, m) for the initial design samples
                init_design_samples_bool = torch.randint(
                    low=0, high=2, size=(init_design_size, self.m), dtype=torch.bool
                )

                 # Sample sampling_budget coaltiions using the original sampler and use the first candidate_size novel samples

            else:
                # CoalitionSampler from shapiq/approximator/sampling.py behaves as follows:
                # (It always samples the full and empty coalition)
                # If many samples are requrested, it applies the border trick first (thereby changing adjusted_sampling_weights) and then samples the remaining coalitions randomly according to the adjusted_sampling_weights.
                # If few samples are requested, it only samples randomly according to the (initial) adjusted_sampling_weights

                # However, as we empirically observed that the border trick samples are not performing well as initial design for the GP , we want to ensure that the initial design samples are not from the border trick.
                from copy import deepcopy

                _temp_id_sampler = deepcopy(self.siq_approximator._sampler)
                # ALGO-03: Complement pairing for better initial design coverage.
                # Sample half the points, then add their complements (N\S).
                # This doubles effective coverage at same init_design_size (Covert & Lee 2021).
                base_count = max(init_design_size // 2 + init_design_size % 2, 1)
                _temp_id_sampler.sample(init_design_size)  # oversample for dedup safety
                all_id_samples = torch.tensor(_temp_id_sampler.coalitions_matrix)
                base_samples = all_id_samples[:base_count, :]
                complements = ~base_samples
                combined = torch.cat([base_samples, complements], dim=0)
                combined = torch.unique(combined, dim=0)
                if combined.shape[0] < init_design_size:
                    # Fill remaining slots from oversampled pool
                    already_in = (all_id_samples[:, None, :] == combined[None, :, :]).all(dim=2).any(dim=1)
                    fill = all_id_samples[~already_in][:init_design_size - combined.shape[0], :]
                    combined = torch.cat([combined, fill], dim=0)
                init_design_samples_bool = combined[:init_design_size, :]

            # Sample sampling_budget coaltiions using the original sampler and use the first candidate_size novel samples
            self.siq_approximator._sampler.sample(sampling_budget)
            candidate_samples_bool = torch.tensor(
                self.siq_approximator._sampler.coalitions_matrix
            )

            # Filter to the first candidate_size samples that are not in the initial design
            already_in_id = (
                (
                    candidate_samples_bool[:, None, :]
                    == init_design_samples_bool[None, :, :]
                )
                .all(dim=2)
                .any(dim=1)
            )  # shape (n1, n2)
            candidate_samples_bool = candidate_samples_bool[~already_in_id][
                :candidate_size, :
            ]

            init_design_samples_binary = torch.tensor(
                init_design_samples_bool, dtype=torch.float64
            )
            candidate_samples_binary = torch.tensor(
                candidate_samples_bool, dtype=torch.float64
            )

            def map_bin_to_cont(binary_samples: torch.Tensor) -> torch.Tensor:
                cont_samples = torch.zeros_like(binary_samples)
                for feature_idx in range(cont_samples.shape[-1]):
                    cont_samples[:, feature_idx] = torch.where(
                        binary_samples[:, feature_idx] == 1,
                        candidate_config[feature_idx],
                        baseline_config[feature_idx],
                    )
                return cont_samples

            init_design_samples_cont = map_bin_to_cont(init_design_samples_binary)
            candidate_samples_cont = map_bin_to_cont(candidate_samples_binary)


            object.__setattr__(self, "_X0_binary", init_design_samples_binary)
            object.__setattr__(self, "_X0", init_design_samples_cont)  # cont
            object.__setattr__(self, "_candidate_set_binary", candidate_samples_binary)
            object.__setattr__(self, "_candidate_set", candidate_samples_cont)  # cont

            object.__setattr__(self, "prop_gt", self.get_prop_gt())

        else:
            self.siq_approximator._sampler.sample(init_design_size)  # amount_samples
            object.__setattr__(
                self,
                "siq_init_design",
                self.siq_approximator._sampler.coalitions_matrix,
            )
            object.__setattr__(
                self,
                "siq_init_design_binary",
                torch.tensor(self.siq_init_design, dtype=torch.float64),
            )

            if m > 20:
                raise ValueError("Exhaustive computation for m>20 not tractable.")

            # Generate Z as all 2^m coalitions (grid of all combinations of 0/1 for m dimensions)
            Z_binary = torch.tensor(
                np.array(
                    [list(map(int, np.binary_repr(i, width=m))) for i in range(2**m)]
                ),
                dtype=torch.float64,
            )

            # Generate A as the Shapley value transformation matrix
            A = torch.zeros((m, Z_binary.shape[0]), dtype=torch.float64)

            Z_sum = Z_binary.sum(axis=1)

            def get_shapley_weight(
                amount_players_in_coalition: int,  # count(T)
                feature_in_coalition: bool,  # \i
            ) -> float:
                # avoid warnings here
                with np.errstate(divide="ignore", invalid="ignore"):
                    if feature_in_coalition:
                        # Case: Feature in coalition => w(T-1)
                        weight = 1 / (
                            m
                            * scipy.special.comb(m - 1, amount_players_in_coalition - 1)
                        )
                    else:
                        # Case: Feature not in coalition => -w(T)
                        weight = -1 / (
                            m * scipy.special.comb(m - 1, amount_players_in_coalition)
                        )
                    return torch.tensor(weight, dtype=torch.float64)

            weights_w = get_shapley_weight(
                amount_players_in_coalition=Z_sum, feature_in_coalition=True
            )
            weights_wo = get_shapley_weight(
                amount_players_in_coalition=Z_sum, feature_in_coalition=False
            )

            for feature_idx in range(m):
                # First set all values to -w(T)
                A[feature_idx, :] = weights_wo

                # Then override entries where feature is in coalition to w(T-1)
                A[feature_idx, Z_binary[:, feature_idx] == 1] = weights_w[
                    Z_binary[:, feature_idx] == 1
                ]

            assert not torch.any(torch.isnan(A)) and not torch.any(
                torch.isinf(A)
            ), "NaN or Inf values in Shapley A matrix."
            assert torch.allclose(
                torch.sum(A, dim=1), torch.zeros(m, dtype=torch.float64)
            ), "Row sums of Shapley A matrix are not zero."

            # Map binary Z to actual Z (choose baseline config whereever 0, candidate config wherever 1)
            # but only for each feature
            Z = torch.zeros_like(Z_binary)
            for feature_idx in range(Z_binary.shape[-1]):
                Z[:, feature_idx] = torch.where(
                    Z_binary[:, feature_idx] == 1,
                    candidate_config[feature_idx],
                    baseline_config[feature_idx],
                )

            # Use shapley kernel for initial design (even in our GP approach)
            # rand_perm = torch.randperm(Z_binary.shape[0])

            # Mapping from each entry in self.siq_samples_binary to equivalent index in Z_binary
            def map_siq_sample_to_Z_binary_idx(siq_sample: torch.Tensor) -> int:
                # Brute-force search
                for z_idx in range(Z_binary.shape[0]):
                    if torch.all(siq_sample == Z_binary[z_idx, :]):
                        return z_idx
                raise ValueError("Could not map siq sample to Z_binary index.")

            siq_init_design_indices_in_Z = []
            for siq_sample_idx in range(self.siq_init_design_binary.shape[0]):
                siq_sample = self.siq_init_design_binary[siq_sample_idx, :]
                siq_idx_in_Z = map_siq_sample_to_Z_binary_idx(siq_sample)
                siq_init_design_indices_in_Z.append(siq_idx_in_Z)

            # siq_candidate_indices_in_Z as complement of siq_init_design_indices_in_Z in range
            siq_candidate_indices_in_Z = torch.tensor(
                [i for i in range(Z.shape[0]) if i not in siq_init_design_indices_in_Z],
                dtype=torch.int64,
            )
            # siq_candidate_indices_in_Z= torch.ones(Z.shape[0], dtype=torch.bool)
            # siq_candidate_indices_in_Z[siq_init_design_indices_in_Z] = False

            X0 = Z[siq_init_design_indices_in_Z, :]
            candidate_set = Z[siq_candidate_indices_in_Z, :]
            candidate_idx_Z = siq_candidate_indices_in_Z

            X0_binary = Z_binary[siq_init_design_indices_in_Z, :]
            candidate_set_binary = Z_binary[siq_candidate_indices_in_Z, :]
            candidate_idx_Z_binary = siq_candidate_indices_in_Z


            assert not (
                (X0_binary[:, None, :] == candidate_set_binary[None, :, :]).all(dim=2)
            ).any(), "X0_binary and candidate_set_binary have overlapping rows."

            # candidate set is already "ordered" => we just iterate through it in AF maximimization for SIQ baseline

            # Required overrides
            object.__setattr__(self, "_A", A)

            object.__setattr__(self, "_Z", Z)
            object.__setattr__(self, "_X0", X0)
            object.__setattr__(self, "_candidate_set", candidate_set)
            object.__setattr__(self, "_candidate_idx_Z", candidate_idx_Z)

            object.__setattr__(self, "_Z_binary", Z_binary)
            object.__setattr__(self, "_X0_binary", X0_binary)
            object.__setattr__(self, "_candidate_set_binary", candidate_set_binary)
            object.__setattr__(self, "_candidate_idx_Z_binary", candidate_idx_Z_binary)

            if self.blackbox_function.is_pseudo_expensive:
                f_Z_gt = self.blackbox_function.evaluate(X=self._Z)
                object.__setattr__(self, "f_Z_gt", f_Z_gt)

                prop_gt = self.A @ self.f_Z_gt[0]
                object.__setattr__(self, "prop_gt", prop_gt)

            # object.__setattr__(self, "Y0", (None, None))

        object.__setattr__(self, "lazy_setup_conducted", True)

    def get_A_Z(self):
        #Used to benchmark ShaplEIG implementation agains naive computation.
        m= self.m

        if m > 16:
            raise ValueError("Exhaustive computation for m>16 not tractable.")

        # Generate Z as all 2^m coalitions (grid of all combinations of 0/1 for m dimensions)
        Z_binary = torch.tensor(
            np.array(
                [list(map(int, np.binary_repr(i, width=m))) for i in range(2**m)]
            ),
            dtype=torch.float64,
        )

        # Generate A as the Shapley value transformation matrix
        A = torch.zeros((m, Z_binary.shape[0]), dtype=torch.float64)

        Z_sum = Z_binary.sum(axis=1)

        def get_shapley_weight(
            amount_players_in_coalition: int,  # count(T)
            feature_in_coalition: bool,  # \i
        ) -> float:
            # avoid warnings here
            with np.errstate(divide="ignore", invalid="ignore"):
                if feature_in_coalition:
                    # Case: Feature in coalition => w(T-1)
                    weight = 1 / (
                        m
                        * scipy.special.comb(m - 1, amount_players_in_coalition - 1)
                    )
                else:
                    # Case: Feature not in coalition => -w(T)
                    weight = -1 / (
                        m * scipy.special.comb(m - 1, amount_players_in_coalition)
                    )
                return torch.tensor(weight, dtype=torch.float64)

        weights_w = get_shapley_weight(
            amount_players_in_coalition=Z_sum, feature_in_coalition=True
        )
        weights_wo = get_shapley_weight(
            amount_players_in_coalition=Z_sum, feature_in_coalition=False
        )

        for feature_idx in range(m):
            # First set all values to -w(T)
            A[feature_idx, :] = weights_wo

            # Then override entries where feature is in coalition to w(T-1)
            A[feature_idx, Z_binary[:, feature_idx] == 1] = weights_w[
                Z_binary[:, feature_idx] == 1
            ]

        assert not torch.any(torch.isnan(A)) and not torch.any(
            torch.isinf(A)
        ), "NaN or Inf values in Shapley A matrix."
        assert torch.allclose(
            torch.sum(A, dim=1), torch.zeros(m, dtype=torch.float64)
        ), "Row sums of Shapley A matrix are not zero."

        # Map binary Z to actual Z (choose baseline config whereever 0, candidate config wherever 1)
        # but only for each feature
        Z = torch.zeros_like(Z_binary)
        for feature_idx in range(Z_binary.shape[-1]):
            Z[:, feature_idx] = torch.where(
                Z_binary[:, feature_idx] == 1,
                self.candidate_config[feature_idx],
                self.baseline_config[feature_idx],
            )

        return A, Z

        

    def get_siq_values(
        self, amount_coalitions, blackbox_fn, surrogate, acquisition_fn_name
    ) -> torch.Tensor:
        # Returns the current ShapIQ SV approximations. Given sampled coalitions of the same size.
        # Note however that the sampled coalitions (of initial design; and between iterations) are independent.
        # Hence CRN is not applied and variance reduction effects are not achieved.

        # See https://shapiq.readthedocs.io/en/latest/_modules/shapiq/approximator/regression/base.html#Regression.kernel_shap_iq_routine
        from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

        if hasattr(self.blackbox_function, "shapiq_game") and isinstance(
            self.blackbox_function.shapiq_game, TreeSHAPIQXAI
        ):
            game = self.blackbox_function.shapiq_game

        else:
            game = ShapIqGame(
                m=self.m,
                surrogate=surrogate,
                archive_size=amount_coalitions,
                blackbox_fn=blackbox_fn,
                exact=False,
            )

        sampling_weights_1 = np.ones(self.m + 1)

        # Define KernelSHAP approximator
        if acquisition_fn_name == "KernelSHAPSampler":
            temp_siq_approximator = KernelSHAP(
                n=self.m,
                index="SV",
                max_order=1,
                pairing_trick=True,  # Modified
                sampling_weights=sampling_weights_1,  # Modified
                random_state=self.seed,
            )

        elif acquisition_fn_name == "LeverageSHAPSampler":
            frontier_generator = ExplanationFrontierGenerator(
                N=[i for i in range(self.m)]
            )
            explanation_frontier = frontier_generator.generate_kadd(max_order=1)

            temp_siq_approximator = PolySHAP(
                n=self.m,
                explanation_frontier=explanation_frontier,
                sampling_weights=sampling_weights_1,
                pairing_trick=True,  # replacement= False,
                random_state=self.seed,
            )

        elif acquisition_fn_name == "SVARMSampler":
            temp_siq_approximator = SVARM(
                n=self.m,
                index="SV",
                pairing_trick=True,  # Modified
                max_order=1,
                sampling_weights=sampling_weights_1,  # Modified
                random_state=self.seed,
            )

        elif acquisition_fn_name == "PermutationSampler":
            temp_siq_approximator = PermutationSamplingSV(
                n=self.m, random_state=self.seed
            )

        elif acquisition_fn_name == "RegressionMSRSampler":
            temp_siq_approximator = RegressionMSR(
                n=self.m,
                pairing_trick=True,
                replacement=False,
                sampling_weights=sampling_weights_1,  # Modified
                random_state=self.seed,
            )

            # even for permuation sampler the amount of evaluated coalitions is specified via budget argument in approximate() call

        else:
            raise ValueError(
                "Acquisition function type not supported for ShapIQ value approximation."
            )

        # Even though seed is fixed, sampling n coals vs. sampling n+1 coals does not lead to first n coals being identical

        if not acquisition_fn_name == "PermutationSampler":
            temp_ks = temp_siq_approximator.approximate(
                budget=amount_coalitions, game=game
            )
            temp_ks_values = temp_ks.values

        else:
            temp_ks = temp_siq_approximator.approximate(
                budget=amount_coalitions, game=game, batch_size=1
            )
            temp_ks_values = temp_ks.values

        empty_coalition_index = temp_siq_approximator._sampler.empty_coalition_index

        if empty_coalition_index is None:
            empty_coalition_index = 0

        assert temp_ks.baseline_value == temp_ks_values[empty_coalition_index]

        sv_approximations = np.concatenate(
            [
                temp_ks_values[:empty_coalition_index],
                temp_ks_values[empty_coalition_index + 1 :],
            ],
            axis=0,
        )

        return sv_approximations


    def get_exact_siq_values(
        self, amount_coalitions, blackbox_fn, surrogate
    ) -> torch.Tensor:
        from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

        if hasattr(self.blackbox_function, "shapiq_game") and isinstance(
            self.blackbox_function.shapiq_game, TreeSHAPIQXAI
        ):
            return self.get_prop_gt()  # PropGT already uses ShapIQ here

        else:
            game = ShapIqGame(
                m=self.m,
                surrogate=surrogate,
                archive_size=amount_coalitions,
                blackbox_fn=blackbox_fn,
                exact=True,
            )

            exact_computer = shapiq.ExactComputer(n_players=game.n_players, game=game)
            sv_exact = exact_computer(index="SV", order=1)
            # print(sv_exact)
            return sv_exact.values[
                1:
            ]  # extract dynamically #as long as assert works this should be fine


    def get_levgp_siq_value(
        self, amount_coalitions, blackbox_fn, partial_gp, acquisition_fn_name, temp_gp
    ) -> torch.Tensor:
        # Hybrid case (Fit GP surrogate on samples from LeverageSHAP)

        # 1. Sample coalitions according to LeverageSHAP (SHAPIQ)
        sampling_weights_1 = np.ones(self.m + 1)
        frontier_generator = ExplanationFrontierGenerator(N=[i for i in range(self.m)])
        explanation_frontier = frontier_generator.generate_kadd(max_order=1)

        siq_approximator = PolySHAP(
            n=self.m,
            explanation_frontier=explanation_frontier,
            sampling_weights=sampling_weights_1,
            pairing_trick=True,  # replacement= False,
            random_state=self.seed,
        )

        siq_approximator._sampler.sample(amount_coalitions)

        train_coals_bool = siq_approximator._sampler.coalitions_matrix

        if self.scalable_mode:
            train_coals_binary = torch.tensor(train_coals_bool, dtype=torch.int64)
            train_X = temp_gp._model.input_transform.untransform(train_coals_binary)
            train_Y = blackbox_fn.evaluate(train_X)[0]

        else:
            Z_coals_bool = self._Z_binary.bool()

            # Compare every row of a with every row of b
            matches = (train_coals_bool[:, None, :] == Z_coals_bool[None, :, :]).all(
                dim=-1
            )
            train_idx = matches.int().argmax(dim=1)

            train_X = self.Z[train_idx]
            train_Y = self.f_Z_gt[0][train_idx]

        # 2. Fit GP surrogate on these samples
        gp = partial_gp(train_X, train_Y)
        # gp._model.train_inputs[0]
        gp.fit()

        # 3. Get SVs according to GP surrogate
        return np.array(self.property_posterior(gp).mean)
    

#########################
    def _get_kernel_alpha_beta_new(self, 
                               surrogate,
                               dtype=torch.float64):
        p = self.amount_players

        #Retrieve lengthscales (player-wise)
        lengthscales = (
            surrogate._model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        )
        device = lengthscales.device
        lengthscales = lengthscales.to(dtype=dtype, device=device)

        #Retrieve outputscale of scale kernel (global)
        outputscale = surrogate._model.covar_module.outputscale.detach().to(
            dtype=dtype, device=device
        )

        #Distribute outputscale equally across players
        outputscale_factor = outputscale.pow(1.0 / p)

        alpha = torch.full((p,), 
                           outputscale_factor.item(), 
                           dtype=dtype, 
                           device=device)
        
        beta = (
            torch.exp(-torch.ones(p, dtype=dtype, device=device) / (lengthscales * p))
            * outputscale_factor
        )

        return alpha, beta

    @torch.no_grad()
    def compute_A_KZW_new(self,
                          W,
                          surrogate,
                          chunk_size: int = 1024) -> torch.Tensor:
        # Custom implementation of A_KZW for the EIG acquisition function.
        # Loops over W in chunks
        assert W.ndim == 2

        chunks = []

        for start in range(0, W.shape[0], chunk_size):
            stop = min(start + chunk_size, W.shape[0])

            chunks.append(
                self._compute_A_KZ_batch_new(
                    W[start:stop, :], 
                    surrogate,
                )
            )

        return torch.cat(chunks, dim=1)


    @torch.no_grad()
    def _compute_A_KZ_batch_new(self, 
                            _batch, 
                            surrogate,
                            stable: bool = True) -> torch.Tensor:
        # Custom implementation of A_KZX for a single batch.

        assert _batch.ndim == 2
        dtype = torch.float64
        p = self.amount_players
        batch_size = _batch.shape[0]

        #Compute alpha and beta
        alpha, beta = self._get_kernel_alpha_beta(surrogate, dtype=dtype)
        device = alpha.device

        _batch = _batch.to(device=device, dtype=dtype)

        #Map to gamma and delta
        gamma = torch.where(_batch == 0, alpha.unsqueeze(0), beta.unsqueeze(0))
        delta = torch.where(_batch == 0, beta.unsqueeze(0), alpha.unsqueeze(0))

        #Compute Shapley weights
        w_in, w_out = self._get_shapley_weights(p, dtype=dtype, device=device)

        if stable:
            result = self._compute_A_KZ_batch_stable_new(gamma, 
                                                    delta, 
                                                    w_in, 
                                                    w_out)
            
        else:
            raise NotImplementedError("Fast version of A_KZ computation (including synthetic division) not implemented in a numerically stable way yet.")
            
        return result.contiguous() #result.T.contiguous()
    
    def _compute_A_KZ_batch_stable_new(self, gamma, delta, w_in, w_out):
        #Stable implementation of A_KZ that avoids synthetic division for numerical stability (at the cost of computational efficiency).

        batch_size, p = gamma.shape
        dtype = gamma.dtype
        device = gamma.device

        #Step 1: Compute prefix and suffix coefficient tables across iterations and their log scales
        #Prefix (Suffix) tables track the coefficients of the generating polynomial across iterations (where in each iteration one factor is multiplied) when computed in forward (backward) direction
        def get_iter_coeff_tables(suffix: bool = False):
            #Vectorized over batch
            tiny = torch.finfo(dtype).tiny #Smallest positive normal number to avoid division by zero

            #Initialize list to store coefficient tables across iterations
            _iterations= (p + 1)
            _coeff_tables= [None] * _iterations
            _log_scales= torch.zeros((_iterations, batch_size), dtype=dtype, device=device)

            #Iterate over factors in forward (backward) direction for prefix (suffix) table
            #(_coeff_tables indexed by iteration, batch and coefficient index)
            _coeff_tables[0 if not suffix else p] = torch.ones((batch_size, 1), dtype=dtype, device=device)

            _range= range(p) if not suffix else range(p - 1, -1, -1)
            for r in _range: #This cannot be parallelized as each iteration depends on the previous one
                #Multiply previous table with next factor in the generating polynomial (for each observation in batch separately)
                prev_index= r if not suffix else r + 1
                next_index= r + 1 if not suffix else r

                prev_table = _coeff_tables[prev_index]
                curr_table = torch.zeros(
                    (batch_size, prev_table.shape[1] + 1), dtype=dtype, device=device
                )
                curr_table[:, :-1] += gamma[:, r : r + 1] * prev_table
                curr_table[:, 1:] += delta[:, r : r + 1] * prev_table #Shift by one for multiplication with variable

                #Normalize: Divide by max coefficient for numerical stability (and track log scales, for each observation in batch separately)
                scales = curr_table.abs().amax(dim=1).clamp_min(tiny)
                _coeff_tables[next_index] = curr_table / scales.unsqueeze(1)
                _log_scales[next_index, :] = _log_scales[prev_index, :] + torch.log(scales)

            return _coeff_tables, _log_scales
        
        prefix_coeffs, prefix_log_scales = get_iter_coeff_tables(suffix= False)
        suffix_coeffs, suffix_log_scales = get_iter_coeff_tables(suffix= True)

        #Step 2: Combine prefix and suffix tables to compute A_KZ entries
        A_KZX= torch.empty((p, batch_size), dtype=dtype, device=device)

        for j in range(p):
            #For each player j, extract prefix and suffix tables
            prefix_table = prefix_coeffs[j]
            suffix_table = suffix_coeffs[j + 1]
            suffix_table_len = suffix_table.shape[1]

            #Compute coefficients of product of prefix- and suffix polynomials via convolution (for each observation in batch separately)
            d = torch.zeros((batch_size, p), dtype=dtype, device=device)

            for i in range(prefix_table.shape[1]):
                d[:, i : i + suffix_table_len] += prefix_table[:, i : i + 1] * suffix_table

            #Compute final entry (Theorem B.1) - also considering j-th value of delta and gamma - and rescaling normalized tables
            total_scale = torch.exp(prefix_log_scales[j] + suffix_log_scales[j + 1]) #Combined scale factors from prefix and suffix tables that we divided by for numerical stability (exp of sum over log scales is equivalent to product over scales)
            A_KZX[j, :] = ((delta[:, j] * (d @ w_in[1:]) 
                           + gamma[:, j] * (d @ w_out[:-1])) 
                           * total_scale)

        return A_KZX


    @torch.no_grad()
    def compute_ASigmaW_new(
        self,
        W,
        surrogate,
        precomputed_A_KZX: torch.Tensor | None = None,
        force_stable: bool = False,
        is_no_refit_step: bool = False,
        new_train_x_binary: torch.Tensor | None = None,
        prev_max_index: int | None = None
    ):
        # Compute A*E*W for an arbitrary W (of various unit vectors)
        # takes W in binary form
        assert (W.ndim == 2) and (W.unique().shape[0] == 2)  # Must only contain 0,1

        # 1. Compute first term A*K(Z, W)
        if is_no_refit_step:
            #Same as in previous step but with one column removed
            A_KZW= torch.concat([
                self.A_KZW[:, :prev_max_index],
                self.A_KZW[:, prev_max_index+1:]
            ], dim=1)


        else:
            A_KZW = self.compute_A_KZW_new(W, surrogate)  # .double()

        object.__setattr__(self, "A_KZW", A_KZW) #Update - also for no refit step - as this is needed for repeated no refit steps

        # 2. Compute second term
        # 2.1: Compute A*K(Z,X)
        A_KZX = (
            precomputed_A_KZX
            if precomputed_A_KZX is not None
            else self.compute_A_KZW_new(W= surrogate._model.train_inputs[0], surrogate= surrogate) #self.compute_A_KZX(surrogate, force_stable=force_stable)
        )  # .double()

        # 2.2: Compute noisy K(X,X)
        K_XX_noisy = self.get_K_XX_noisy(surrogate) #ok

        # 2.3: Compute K(X,W)
        X = surrogate._model.train_inputs[0]
        K_XW = surrogate._model.covar_module(X, W)  # .to_dense() => Lazy variant
        #In case of no refit step, we could also update K_XW from previous step by removing one column

        # 2.4: Solve LES
        inv_K_XX_A_KZX = K_XX_noisy.solve(A_KZX.T)

        A_KZX_inv_K_XX_K_XW = inv_K_XX_A_KZX.T @ K_XW

        # 3. Combine components (no scaling by emp std. yet for numerical stability)
        # emp_std= surrogate._model.outcome_transform.stdvs.squeeze() #Scale by empirical variance due to output standardization
        ASigmaW = A_KZW - A_KZX_inv_K_XX_K_XW  # * (emp_std ** 2)

        # # 2.4: Solve LES and add everything up
        # inv_K_XX_K_XW= K_XX_noisy.solve(K_XW)

        # # 3. Combine components
        # emp_std= surrogate._model.outcome_transform.stdvs.squeeze() #Scale by empirical variance due to output standardization
        # ASigmaW= (A_KZW - (A_KZX @ inv_K_XX_K_XW)) * (emp_std ** 2)

        return ASigmaW


    @torch.no_grad()
    def compute_AEA_new(
        self,
        surrogate,
        scale_by_emp_std: bool = False,
        force_stable: bool = False,
        precomputed_A_KZX: torch.Tensor | None = None,
        is_no_refit_step: bool = False,
    ):
        if is_no_refit_step and hasattr(self, "AKA"):
            # In no-refit steps, we can reuse the previously computed AKA (which is independent of training data) to save compute.
            AKA= self.AKA
            #No need for object.__setattr__(self, "AKA", AKA) as it does not change

        else:
            AKA = self.compute_AKZZA_new(surrogate) #compute_AKZZA_new
            object.__setattr__(self, "AKA", AKA)

            # AKA_old= self.compute_AKZZA_old(surrogate)
            # assert torch.allclose(AKA, AKA_old)

            # AKA_chunked= self.compute_AKZZA_new_chunked(surrogate, chunk_size= 16)
            # assert torch.allclose(AKA, AKA_chunked)
            # Not faster in pratice due to memory overhead for larger p

        A_KZX = (
            precomputed_A_KZX
            if precomputed_A_KZX is not None
            else self.compute_A_KZW_new(W= surrogate._model.train_inputs[0], 
                                        surrogate= surrogate)
        )
        C = self.get_K_XX_noisy(surrogate)

        # C = L L^T
        L = psd_safe_cholesky(C.to_dense())
        # surrogate._model.prediction_strategy.lik_train_train_covar.root_decomposition().root.to_dense() (but this does not always work)

        # Solve L Y = A_KZX^T  -> Y = L^{-1} A_KZX^T
        Y = torch.linalg.solve_triangular(L, A_KZX.T, upper=False)

        # Then T = Y^T Y = A_KZX C^{-1} A_KZX^T
        T = Y.T @ Y

        AEA = AKA - T
        AEA = 0.5 * (AEA + AEA.T) #Symmetrize for numerical stability
        if scale_by_emp_std:
            emp_std = surrogate._model.outcome_transform.stdvs.squeeze()
            AEA = AEA * (emp_std**2)

        return AEA

    def compute_AKZZA_new(self, surrogate) -> torch.Tensor:
        alpha, beta = self._get_kernel_alpha_beta_new(surrogate, dtype=torch.float64)
        p = alpha.numel()
        dtype = alpha.dtype
        device = alpha.device
        tiny = torch.finfo(dtype).tiny

        w_in, w_out = self.shapley_w_in_out(p, dtype=dtype, device=device)

        # Shifted weight vectors
        O0 = w_out.clone()

        O1 = torch.zeros(p + 1, dtype=dtype, device=device)
        O1[:p] = w_out[1:]

        I1 = torch.zeros(p + 1, dtype=dtype, device=device)
        I1[:p] = w_in[1:]

        I2 = torch.zeros(p + 1, dtype=dtype, device=device)
        I2[:p - 1] = w_in[2:]

        LEFT_T = torch.stack([O0, O1, I1, I2]).T.contiguous()   # (p+1, 4)
        RIGHT_T = torch.stack([O0, O1, I1, I2]).T.contiguous() # (p+1, 4)

        ps = p + 1  # shorthand for coefficient table side length

        # Pre-store alpha/beta as plain floats.
        alpha_cpu = alpha.tolist()
        beta_cpu = beta.tolist()

        #Prefix computation helper buffers (reusable across i)
        P_left_curr = torch.zeros((ps, ps, 4), dtype=dtype, device=device)
        P_left_new = torch.empty_like(P_left_curr)

        #Initialize reusable buffer (across i) for contracted suffix table (For each player i, it stores the contracted suffix coefficient tables for all pairs (i,r) from the backward pass.)
        suffix_contr_i= torch.zeros((p, ps, ps, 4), dtype=dtype, device=device) # max nf = p-1
        suffix_log_i = torch.zeros(p, dtype=dtype, device=device)

        #Suffix computation helper buffers (reusable across i)
        S_right_prev = torch.zeros((ps, ps, 4), dtype=dtype, device=device)
        S_right_new = torch.empty_like(S_right_prev)

        AKZZA = torch.zeros((p, p), dtype=dtype, device=device)

        for i in range(p): #Computed for each player i independently. However, vectorization or chunking across i is not faster due to memory overhead.
            # ---- Backward pass: Iteratively build suffix coefficient tables for i across factors r in reverse order ----
            #(In detail, it takes all factors in reverse order except i.)
            factors_wo_i = [r for r in range(p) if r != i] #All players except i
            nf = len(factors_wo_i)

            # Reset helpers
            S_right_prev.zero_()
            S_right_new.zero_() 
            suffix_contr_i.zero_()
            suffix_log_i.zero_()

            #Initialize S_right to correspond to empty suffix (which is the base case for the backward pass)
            S_right_prev[0, :, :] = RIGHT_T
            log_S = 0.0

            #For a certain i, compute contracted suffices all the way until the full suffix
            #(Only until i is not sufficient. Consider e.g. case i>j.) 
            for k in range(nf - 1, -1, -1):
                r = factors_wo_i[k]
                a_j, b_j = alpha[r], beta[r]

                #Compute new contracted suffix based on previous one (from previous backward pass)
                S_right_new = a_j * S_right_prev
                S_right_new[1:, :, :]  +=  b_j * S_right_prev[:-1, :, :]
                S_right_new[:, :-1, :]  +=  b_j * S_right_prev[:, 1:, :]
                S_right_new[1:, :-1, :]  +=  a_j * S_right_prev[:-1, 1:, :]

                #Normalize for numerical stability
                s_new = S_right_new.abs().max().clamp_min(tiny)
                S_right_new = S_right_new / s_new
                log_S = log_S + s_new.log().item()

                #Persist and update previous for next iteration
                suffix_contr_i[k] = S_right_new
                suffix_log_i[k] = log_S

                S_right_prev= S_right_new

            # ---- Compute diagonal entry of AKZZA for i ----
            #Given that the last iteration of the backward pass computes contracted suffices for all factors except i (suffix_contr_i[0]), we can directly compute the diagonal entry.

            #Contracted prefix for diagonal entry is just the initial prefix
            P_left_curr.zero_()
            P_left_curr[:, 0, :] = LEFT_T
            log_P = 0.0

            contracted_diag = torch.einsum('abl,abr->lr', P_left_curr, suffix_contr_i[0])
            scale_diag = log_P + suffix_log_i[0].item()

            ai_v = alpha_cpu[i]
            bi_v = beta_cpu[i]
            diag_val = ai_v * (contracted_diag[2, 2] + contracted_diag[0, 0]) \
                     - bi_v * (contracted_diag[2, 0] + contracted_diag[0, 2])
            AKZZA[i, i] = diag_val.item() * math.exp(scale_diag) / (p * p)

            # ---- Forward pass: Iteratively compute off-diagonal entries for pairs (i,j) by updating prefix state and combining with corresponding suffix state from backward pass ----
            #It suffices to maintain current prefix state as opposed to persisting all prefix states across iterations, as they are not reused anymore.

            for m in range(nf):
                j = factors_wo_i[m]

                if j > i:
                    continue #Only compute forward pass for j <= i, as M is symmetric and we fill in both M[i,j] and M[j,i] in a latter step.
                    #Case i=j does not occur in this loop as i is not in factors_wo_i

                aj_v = alpha_cpu[j]
                bj_v = beta_cpu[j]

                # Suffix for this step is at index m+1 in suffix_S if m < nf-1,
                # or the identity suffix if m == nf-1
                if m < nf - 1:
                    s_idx = m + 1
                    contracted = torch.einsum('abl,abr->lr', P_left_curr, suffix_contr_i[s_idx])
                    scale = log_P + suffix_log_i[s_idx].item()
                else:
                    # Empty suffix: S_right(0, b1, r) = RIGHT[r, b1], rest 0
                    contracted = torch.einsum('bl,br->lr', P_left_curr[0, :, :], RIGHT_T)
                    scale = log_P
                #Here we can simply continue using P_left_curr (with P_left_curr[:, 0, :] = LEFT_T) from the diagonal entry computation of the first iteration. For the diagonal computation, we want from the left only the weights, as all except i are in right.
                # For the initial off-diagonal entry (i,1), it holds that suffix_contr_i[s_idx]== suffix_contr_i[1] contains all players except i and j=1. Thus we do not need to update the left term yet, as removing j=1 is accounted for in right /suffix_contr_i[1].
                #  From this step on, P_left_curr is updated such that for (i,2) it contains player 1. 

                # Build K and compute M[i,j]
                val = (
                    ai_v * aj_v * contracted[0, 0] + bi_v * aj_v * contracted[0, 1]
                    - ai_v * bj_v * contracted[0, 2] - bi_v * bj_v * contracted[0, 3]
                    + ai_v * bj_v * contracted[1, 0] + bi_v * bj_v * contracted[1, 1]
                    - ai_v * aj_v * contracted[1, 2] - bi_v * aj_v * contracted[1, 3]
                    - bi_v * aj_v * contracted[2, 0] - ai_v * aj_v * contracted[2, 1]
                    + bi_v * bj_v * contracted[2, 2] + ai_v * bj_v * contracted[2, 3]
                    - bi_v * bj_v * contracted[3, 0] - ai_v * bj_v * contracted[3, 1]
                    + bi_v * aj_v * contracted[3, 2] + ai_v * aj_v * contracted[3, 3]
                ).item() * math.exp(scale) / (p * p)

                AKZZA[i, j] = val
                AKZZA[j, i] = val

                # Update P_left (add factor j only after computing M[i,j])
                a_j, b_j = alpha[j], beta[j]

                P_left_new= a_j * P_left_curr
                P_left_new[:-1, :, :] += b_j * P_left_curr[1:, :, :] #Pr_i,j-1(a_2+1,b_1) (with last entry multiplied by out-of-range/ 0)
                P_left_new[:, 1:, :] += b_j * P_left_curr[:, :-1, :] #Pr_i,j-1(a_2,b_1-1) (with first entry multiplied by out-of-range/ 0)
                P_left_new[:-1, 1:, :] += a_j * P_left_curr[1:, :-1, :]

                s_new = P_left_new.abs().max().clamp_min(tiny)
                P_left_new = P_left_new / s_new
                log_P = log_P + s_new.log().item()
                P_left_curr = P_left_new

        return AKZZA
##########################






    def _get_kernel_alpha_beta(self, 
                               surrogate,
                               ignore_outputscale = False, 
                               dtype=torch.float64):
        p = self.amount_players
        lengthscales = (
            surrogate._model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        )
        device = lengthscales.device
        lengthscales = lengthscales.to(dtype=dtype, device=device)
        outputscale = surrogate._model.covar_module.outputscale.detach().to(
            dtype=dtype, device=device
        )

        if ignore_outputscale:
            log.warning("ignoring outputscale and setting to 1.")
            alpha = torch.full((p,), 1, dtype=dtype, device=device)
            beta = (
                torch.exp(-torch.ones(p, dtype=dtype, device=device) / (lengthscales * p))
            )
            return alpha, beta

        else:
            outputscale_factor = outputscale.pow(1.0 / p)
            alpha = torch.full((p,), outputscale_factor.item(), dtype=dtype, device=device)
            beta = (
                torch.exp(-torch.ones(p, dtype=dtype, device=device) / (lengthscales * p))
                * outputscale_factor
            )
            return alpha, beta

    def _get_shapley_weights(self, p, dtype, device):
        w_in = torch.zeros(p + 1, dtype=dtype, device=device)
        w_out = torch.zeros(p + 1, dtype=dtype, device=device)
        for k in range(1, p + 1):
            w_in[k] = 1.0 / (math.comb(p - 1, k - 1) * p)
        for k in range(0, p):
            w_out[k] = -1.0 / (math.comb(p - 1, k) * p)
        return w_in, w_out

    def _compute_A_KZ_batch_stable(self, gamma, delta, w_in, w_out):
        batch_size, p = gamma.shape
        dtype = gamma.dtype
        device = gamma.device
        tiny = torch.finfo(dtype).tiny

        prefix_coeffs = [None] * (p + 1)
        prefix_log_scale = torch.zeros((batch_size, p + 1), dtype=dtype, device=device)
        prefix_coeffs[0] = torch.ones((batch_size, 1), dtype=dtype, device=device)
        for r in range(p):
            prev = prefix_coeffs[r]
            curr = torch.zeros(
                (batch_size, prev.shape[1] + 1), dtype=dtype, device=device
            )
            curr[:, :-1] += gamma[:, r : r + 1] * prev
            curr[:, 1:] += delta[:, r : r + 1] * prev
            scale = curr.abs().amax(dim=1).clamp_min(tiny)
            prefix_coeffs[r + 1] = curr / scale.unsqueeze(1)
            prefix_log_scale[:, r + 1] = prefix_log_scale[:, r] + torch.log(scale)

        suffix_coeffs = [None] * (p + 1)
        suffix_log_scale = torch.zeros((batch_size, p + 1), dtype=dtype, device=device)
        suffix_coeffs[p] = torch.ones((batch_size, 1), dtype=dtype, device=device)
        for r in range(p - 1, -1, -1):
            prev = suffix_coeffs[r + 1]
            curr = torch.zeros(
                (batch_size, prev.shape[1] + 1), dtype=dtype, device=device
            )
            curr[:, :-1] += gamma[:, r : r + 1] * prev
            curr[:, 1:] += delta[:, r : r + 1] * prev
            scale = curr.abs().amax(dim=1).clamp_min(tiny)
            suffix_coeffs[r] = curr / scale.unsqueeze(1)
            suffix_log_scale[:, r] = suffix_log_scale[:, r + 1] + torch.log(scale)

        A_stable = torch.empty((batch_size, p), dtype=dtype, device=device)
        for j in range(p):
            left = prefix_coeffs[j]
            right = suffix_coeffs[j + 1]
            right_len = right.shape[1]
            weighted_in = torch.zeros(batch_size, dtype=dtype, device=device)
            weighted_out = torch.zeros(batch_size, dtype=dtype, device=device)
            for idx in range(left.shape[1]):
                weighted_in += left[:, idx] * (
                    right @ w_in[1 + idx : 1 + idx + right_len]
                )
                weighted_out += left[:, idx] * (right @ w_out[idx : idx + right_len])
            total_scale = torch.exp(prefix_log_scale[:, j] + suffix_log_scale[:, j + 1])
            A_stable[:, j] = (
                delta[:, j] * weighted_in + gamma[:, j] * weighted_out
            ) * total_scale

        return A_stable

    def _rowwise_force_stable_mask(self, gamma, delta):
        tiny = torch.finfo(gamma.dtype).tiny
        coeff_min = torch.minimum(gamma.abs(), delta.abs()).amin(dim=1)
        coeff_max = torch.maximum(gamma.abs(), delta.abs()).amax(dim=1)
        dynamic_range = coeff_max / coeff_min.clamp_min(tiny)
        return (coeff_min < 1e-15) | (dynamic_range > 1e12)

    def _combine_A_from_d_batch(self, gamma, delta, d, scale, w_in, w_out):
        A_in = torch.einsum("bj,bjk,k->bj", delta, d, w_in[1:])
        A_out = torch.einsum("bj,bjk,k->bj", gamma, d, w_out[:-1])
        return (A_in + A_out) * scale.unsqueeze(1)

    def _build_scaled_c_batch(self, gamma, delta):
        #Build coefficient table according to Lemma 1
        batch_size, p = gamma.shape
        dtype = gamma.dtype
        device = gamma.device
        tiny = torch.finfo(dtype).tiny
        c = torch.zeros((batch_size, p + 1), dtype=dtype, device=device)
        c[:, 0] = 1.0
        log_scale = torch.zeros(batch_size, dtype=dtype, device=device)
        for r in range(p):
            c_new = gamma[:, r : r + 1] * c #For each independent row, compute new coeff as previous time current gamma_r + previous coeff of lower order time current delta_r (next line); this is iteratively as opposed to recursively and we cant paralellize
            c_new[:, 1:] += delta[:, r : r + 1] * c[:, :-1]
            c[:, : r + 2] = c_new[:, : r + 2]
            scale = c[:, : r + 2].abs().amax(dim=1).clamp_min(tiny)
            c[:, : r + 2] = c[:, : r + 2] / scale.unsqueeze(1)
            log_scale = log_scale + torch.log(scale)
        return c, log_scale
    
    def _solve_q_forward_batch(self, c, gamma_j, delta_j):
        batch_size, p1 = c.shape
        p = p1 - 1
        q = torch.zeros((batch_size, p), dtype=c.dtype, device=c.device)
        q[:, 0] = c[:, 0] / gamma_j
        for k in range(1, p):
            q[:, k] = (c[:, k] - delta_j * q[:, k - 1]) / gamma_j
            #Inverse operation of generating polynomial dynamic program for j-th step
            #But needs to be looped over as q's depend on previous q's
        return q

    def _solve_q_backward_batch(self, c, gamma_j, delta_j):
        batch_size, p1 = c.shape
        p = p1 - 1
        q = torch.zeros((batch_size, p), dtype=c.dtype, device=c.device)
        q[:, p - 1] = c[:, p] / delta_j
        for k in range(p - 2, -1, -1):
            q[:, k] = (c[:, k + 1] - gamma_j * q[:, k + 1]) / delta_j
        return q

    def _recon_error_batch(self, q, c, gamma_j, delta_j):
        batch_size, p = q.shape
        recon = torch.zeros((batch_size, p + 1), dtype=q.dtype, device=q.device)
        recon[:, 0] = gamma_j * q[:, 0]
        recon[:, 1:p] = (
            gamma_j.unsqueeze(1) * q[:, 1:] + delta_j.unsqueeze(1) * q[:, :-1]
        )
        recon[:, p] = delta_j * q[:, p - 1]
        denom = c.abs().amax(dim=1).clamp_min(1e-30)
        return (recon - c).abs().amax(dim=1) / denom

    def _compute_A_KZ_batch_fast(self, gamma, delta, w_in, w_out):
        batch_size, p = gamma.shape
        c, log_scale = self._build_scaled_c_batch(gamma, delta)
        scale = torch.exp(log_scale)

        d_fast = torch.zeros((batch_size, p, p), dtype=gamma.dtype, device=gamma.device)
        prefer_forward = gamma.abs() >= delta.abs()
        fast_ok = torch.ones(batch_size, dtype=torch.bool, device=gamma.device)

        for j in range(p): #this should be parallelizable
            gamma_j = gamma[:, j]
            delta_j = delta[:, j]
            if prefer_forward[:, j].all():
                if (gamma_j.abs() <= 1e-30).any():
                    fast_ok &= gamma_j.abs() > 1e-30
                    continue
                d_fast[:, j, :] = self._solve_q_forward_batch(c, gamma_j, delta_j)
                continue

            if (~prefer_forward[:, j]).all():
                if (delta_j.abs() <= 1e-30).any():
                    fast_ok &= delta_j.abs() > 1e-30
                    continue
                d_fast[:, j, :] = self._solve_q_backward_batch(c, gamma_j, delta_j)
                continue

            q_j = torch.zeros((batch_size, p), dtype=gamma.dtype, device=gamma.device)
            forward_mask = prefer_forward[:, j]
            backward_mask = ~forward_mask
            if forward_mask.any():
                q_j[forward_mask, :] = self._solve_q_forward_batch(
                    c[forward_mask], gamma_j[forward_mask], delta_j[forward_mask]
                )
            if backward_mask.any():
                q_j[backward_mask, :] = self._solve_q_backward_batch(
                    c[backward_mask], gamma_j[backward_mask], delta_j[backward_mask]
                )
            d_fast[:, j, :] = q_j

        fast_ok &= torch.isfinite(d_fast).all(dim=2).all(dim=1)
        fast_ok &= d_fast.abs().amax(dim=2).amax(dim=1) <= 1e10
        fast_ok &= torch.isfinite(scale) & (scale.abs() > 0.0)

        max_recon_err = torch.zeros(batch_size, dtype=gamma.dtype, device=gamma.device)
        for j in range(p):
            max_recon_err = torch.maximum(
                max_recon_err,
                self._recon_error_batch(d_fast[:, j, :], c, gamma[:, j], delta[:, j]),
            )
        fast_ok &= max_recon_err <= 5e-10

        A_fast = self._combine_A_from_d_batch(gamma, delta, d_fast, scale, w_in, w_out)
        return A_fast, fast_ok, c, scale

    @torch.no_grad()
    def _compute_A_KZ_batch(self, _batch, surrogate, ignore_outputscale: bool = False, force_stable: bool = False):
        assert _batch.ndim == 2
        p = self.amount_players
        dtype = torch.float64
        alpha, beta = self._get_kernel_alpha_beta(surrogate, ignore_outputscale= ignore_outputscale, dtype=dtype)
        device = alpha.device
        _batch = _batch.to(device=device, dtype=dtype)
        gamma = torch.where(_batch == 0, alpha.unsqueeze(0), beta.unsqueeze(0))
        delta = torch.where(_batch == 0, beta.unsqueeze(0), alpha.unsqueeze(0))
        w_in, w_out = self._get_shapley_weights(p, dtype=dtype, device=device)


        if ignore_outputscale:
            log.warning("ignoring outputscale and testing fast computation.")
            #Intermezzo. Testing new code.

            A_fast, fast_ok, _, _ = self._compute_A_KZ_batch_fast(gamma, delta, w_in, w_out)
            outputscale = surrogate._model.covar_module.outputscale.detach().to(dtype=dtype, device=device)
            return (outputscale * A_fast).T.contiguous()


        batch_size = _batch.shape[0]
        result = torch.empty((batch_size, p), dtype=dtype, device=device)
        stable_mask = (
            torch.ones(batch_size, dtype=torch.bool, device=device)
            if force_stable
            else self._rowwise_force_stable_mask(gamma, delta)
        )
        active_mask = ~stable_mask

        if active_mask.any():
            active_indices = active_mask.nonzero(as_tuple=False).squeeze(1)
            active_gamma = gamma[active_indices]
            active_delta = delta[active_indices]
            A_active = None


            if A_active is None:
                A_active = self._compute_A_KZ_batch_stable(
                    active_gamma, active_delta, w_in, w_out
                )



            result[active_indices, :] = A_active

        stable_indices = stable_mask.nonzero(as_tuple=False).squeeze(1)
        if stable_indices.numel() > 0:
            result[stable_indices, :] = self._compute_A_KZ_batch_stable(
                gamma[stable_indices], delta[stable_indices], w_in, w_out
            )


        return result.T.contiguous()

    @torch.no_grad()
    def compute_A_KZz(self, z, surrogate, force_stable: bool = False):
        assert (z.ndim == 2) and (z.shape[0] == 1)
        return self._compute_A_KZ_batch(z, surrogate, force_stable=force_stable)[:, 0]

    @torch.no_grad()
    def compute_A_KZX(self, surrogate, ignore_outputscale: bool = False, force_stable: bool = False):
        X = surrogate._model.train_inputs[0]
        chunk_size = 1024
        chunks = []
        for start in range(0, X.shape[0], chunk_size):
            stop = min(start + chunk_size, X.shape[0])
            chunks.append(
                self._compute_A_KZ_batch(
                    X[start:stop, :], 
                    surrogate, 
                    ignore_outputscale=ignore_outputscale,
                    force_stable=force_stable
                )
            )
        return torch.cat(chunks, dim=1)
    
    def update_A_KZX(self, 
                     surrogate,
                     new_train_x_binary: torch.Tensor,
                     force_stable: bool = False):
        assert (surrogate._model.train_inputs[0][-1] == new_train_x_binary).all()
        
        #Compute A_KZz for new training point
        A_KZx= self.compute_A_KZz(new_train_x_binary, 
                                  surrogate, 
                                  force_stable=force_stable)
        
        #Append to previous A_KZX
        A_KZX_new= torch.cat([self.A_KZX, 
                              A_KZx.unsqueeze(1)], dim=1)

        return A_KZX_new

    @torch.no_grad()
    def compute_A_KZW(self, W, surrogate, force_stable: bool = False):
        assert W.ndim == 2
        chunk_size = 1024
        chunks = []
        for start in range(0, W.shape[0], chunk_size):
            stop = min(start + chunk_size, W.shape[0])
            chunks.append(
                self._compute_A_KZ_batch(
                    W[start:stop, :], surrogate, force_stable=force_stable
                )
            )
        return torch.cat(chunks, dim=1)


    def compute_A_1(self, M):
        # Computes multiplication of A (MxS) with vector 1 (Sx1)
        # However, by definition rows of A sum up to 0
        return torch.zeros(M)

    def get_K_XX_noisy(self, surrogate):
        X = surrogate._model.train_inputs[0]
        K_XX = surrogate._model.covar_module(
            X, X
        )  # __call__ as opposed to forward returns LazyTensor
        K_XX_noisy = K_XX.add_diagonal(
            surrogate._model.likelihood.noise
        )  # Remains lazy

        return K_XX_noisy

    @torch.no_grad()
    def compute_AF_PPM(self, surrogate, precomputed_A_KZX: torch.Tensor | None = None):
        if isinstance(
            surrogate._model.mean_module, gpytorch.means.ConstantMean
        ) and isinstance(
            surrogate._model.outcome_transform,
            botorch.models.transforms.outcome.Standardize,
        ):
            # 0. Declare relevant variables
            # M= self.A.shape[0]
            # S= self.A.shape[1]
            M = self.amount_players
            S = self.amount_coalitions

            # Mean prior (learnable)
            mean_prior_trans = surrogate._model.mean_module.constant
            mean_prior_untrans = surrogate._model.outcome_transform.untransform(
                mean_prior_trans
            )[0].squeeze()

            # Empirical mean and std of training data
            emp_mean_untrans = surrogate._model.outcome_transform.means.squeeze()
            emp_mean_trans = surrogate._model.outcome_transform(emp_mean_untrans)[
                0
            ].squeeze()  # 0

            emp_std_untrans = surrogate._model.outcome_transform.stdvs.squeeze()

            # 1. Compute SV prior (transformed)
            # A_muZ_prior= self.A @ emp_mean_untrans.repeat(S)
            A_muZ_prior = (
                self.compute_A_1(M) * emp_mean_untrans
            )  # Could be ignored, as it reduces to 0

            # 2. Compute transformed correction term (for standardization)
            # A_muZ_trans_scaled= (self.A @ mean_prior_trans.repeat(S)) * emp_std_untrans
            A_muZ_trans_scaled = (
                self.compute_A_1(M) * mean_prior_trans * emp_std_untrans
            )  # Could be ignored, as it reduces down to 0 for constant prior

            # 3. Compute scaled mean predictions
            # 3.1 Compute A*K(Z,X)
            A_KZX = (
                precomputed_A_KZX
                if precomputed_A_KZX is not None
                else self.compute_A_KZX(surrogate)
            )

            # As in gpytorch/models/exact_prediction_strategies.py exact_predictive_mean and _mean_cache
            # Solve linear system K(X,X)*X= Y for X
            K_XX_noisy = self.get_K_XX_noisy(surrogate)
            Y = surrogate._model.train_targets - mean_prior_trans  # Subtract mean
            inv_K_XX_Y = K_XX_noisy.solve(
                Y
            )  # .evaluate_kernel() as done in exact_prediction_strategies.py?
            # Equivalent to mean_cache

            # Multiply A*K(Z,X) with K(X,X)^(-1)*Y to get SV estimates
            A_PPM_scaled = (A_KZX @ inv_K_XX_Y) * emp_std_untrans

            # 4. Compute final result
            SV_PPM = A_muZ_prior + A_muZ_trans_scaled + A_PPM_scaled

        else:
            raise NotImplementedError()

        return SV_PPM

    @torch.no_grad()
    def compute_ASigmaW(
        self,
        W,
        surrogate,
        precomputed_A_KZX: torch.Tensor | None = None,
        force_stable: bool = False,
        is_no_refit_step: bool = False,
        new_train_x_binary: torch.Tensor | None = None,
        prev_max_index: int | None = None
    ):
        # Compute A*E*W for an arbitrary W (of various unit vectors)
        # takes W in binary form
        assert (W.ndim == 2) and (W.unique().shape[0] == 2)  # Must only contain 0,1

        # 1. Compute first term A*K(Z, W)
        if is_no_refit_step:
            #Same as in previous step but with one column removed
            A_KZW= torch.concat([
                self.A_KZW[:, :prev_max_index],
                self.A_KZW[:, prev_max_index+1:]
            ], dim=1)


        else:
            A_KZW = self.compute_A_KZW(W, surrogate, force_stable=force_stable)  # .double()

        object.__setattr__(self, "A_KZW", A_KZW) #Update - also for no refit step - as this is needed for repeated no refit steps

        # 2. Compute second term
        # 2.1: Compute A*K(Z,X)
        A_KZX = (
            precomputed_A_KZX
            if precomputed_A_KZX is not None
            else self.compute_A_KZX(surrogate, force_stable=force_stable)
        )  # .double()

        # 2.2: Compute noisy K(X,X)
        K_XX_noisy = self.get_K_XX_noisy(surrogate)

        # 2.3: Compute K(X,W)
        X = surrogate._model.train_inputs[0]
        K_XW = surrogate._model.covar_module(X, W)  # .to_dense() => Lazy variant
        #In case of no refit step, we could also update K_XW from previous step by removing one column

        # 2.4: Solve LES
        inv_K_XX_A_KZX = K_XX_noisy.solve(A_KZX.T)

        A_KZX_inv_K_XX_K_XW = inv_K_XX_A_KZX.T @ K_XW

        # 3. Combine components (no scaling by emp std. yet for numerical stability)
        # emp_std= surrogate._model.outcome_transform.stdvs.squeeze() #Scale by empirical variance due to output standardization
        ASigmaW = A_KZW - A_KZX_inv_K_XX_K_XW  # * (emp_std ** 2)

        # # 2.4: Solve LES and add everything up
        # inv_K_XX_K_XW= K_XX_noisy.solve(K_XW)

        # # 3. Combine components
        # emp_std= surrogate._model.outcome_transform.stdvs.squeeze() #Scale by empirical variance due to output standardization
        # ASigmaW= (A_KZW - (A_KZX @ inv_K_XX_K_XW)) * (emp_std ** 2)

        return ASigmaW

    ###############

    def build_base_table(self, alpha, beta, i, j):
        """
        Coeff table for prod_{r != i,j} f_r(z1,z2),
        where f_r = alpha_r(1+z1 z2) + beta_r(z1+z2).
        Returns C of shape (p+1, p+1) with degrees up to p.
        """
        p = alpha.numel()
        C = torch.zeros((p + 1, p + 1), dtype=alpha.dtype, device=alpha.device)
        C[0, 0] = 1.0
        log_scale = torch.tensor(0.0, dtype=alpha.dtype, device=alpha.device)

        for r in range(p):
            if r == i or r == j:  # Product over all except 2
                continue

            a = alpha[r]
            b = beta[r]
            # multiply by: g00 + g10 z1 + g01 z2 + g11 z1 z2 with
            g00, g10, g01, g11 = a, b, b, a

            Cnew = torch.zeros_like(C)

            # g00 * C
            Cnew += g00 * C

            # g10 z1 * C -> shift a-1
            Cnew[1:, :] += g10 * C[:-1, :]

            # g01 z2 * C -> shift b-1
            Cnew[:, 1:] += g01 * C[:, :-1]

            # g11 z1 z2 * C -> shift both
            Cnew[1:, 1:] += g11 * C[:-1, :-1]

            # NEW: normalize after each multiplication step
            s_r = Cnew.abs().max().clamp_min(torch.finfo(alpha.dtype).tiny)
            C = Cnew / s_r
            log_scale = log_scale + torch.log(s_r)
            # C = Cnew

        return C, log_scale  # New (also return log scale for later rescaling)
        # return C

    def apply_restriction_S(self, C, alpha_i, beta_i, u):
        """
        Multiply coeff table C by:
        u=1: f_i^{S in}  = z1 (beta_i + alpha_i z2) = beta_i z1 + alpha_i z1 z2
        u=0: f_i^{S not}= alpha_i + beta_i z2
        """
        p = C.shape[0] - 1
        out = torch.zeros_like(C)

        if u == 1:
            # beta_i z1 term
            out[1:, :] += beta_i * C[:-1, :]
            # alpha_i z1 z2 term
            out[1:, 1:] += alpha_i * C[:-1, :-1]
        else:
            # alpha_i * 1
            out += alpha_i * C
            # beta_i z2 term
            out[:, 1:] += beta_i * C[:, :-1]

        return out

    def apply_restriction_T(self, C, alpha_j, beta_j, v):
        """
        Multiply coeff table C by:
        v=1: f_j^{T in}  = z2 (beta_j + alpha_j z1) = beta_j z2 + alpha_j z1 z2
        v=0: f_j^{T not}= alpha_j + beta_j z1
        """
        p = C.shape[0] - 1
        out = torch.zeros_like(C)

        if v == 1:
            # beta_j z2 term
            out[:, 1:] += beta_j * C[:, :-1]
            # alpha_j z1 z2 term
            out[1:, 1:] += alpha_j * C[:-1, :-1]
        else:
            # alpha_j * 1
            out += alpha_j * C
            # beta_j z1 term
            out[1:, :] += beta_j * C[:-1, :]

        return out

    def shapley_w_in_out(self, p, dtype=torch.float64, device=None):
        """
        Returns w_in[a], w_out[a] for a=0..p with conventions:
        w_in(0)=0, w_out(p)=0
        w_in(a)=1/binom(p-1,a-1) for a>=1
        w_out(a)=1/binom(p-1,a)   for a<=p-1
        Note: The theorem uses these without the 1/p factor; the 1/p^2 is applied outside.
        """
        w_in = torch.zeros(p + 1, dtype=dtype, device=device)
        w_out = torch.zeros(p + 1, dtype=dtype, device=device)

        for a in range(1, p + 1):
            w_in[a] = 1.0 / math.comb(p - 1, a - 1)
        for a in range(0, p):
            w_out[a] = 1.0 / math.comb(p - 1, a)

        # conventions: already w_in[0]=0, w_out[p]=0
        return w_in, w_out

    def build_base_table_excluding_one(self, alpha, beta, i):
        p = alpha.numel()
        C = torch.zeros((p + 1, p + 1), dtype=alpha.dtype, device=alpha.device)
        C[0, 0] = 1.0

        log_scale = torch.tensor(
            0.0, dtype=alpha.dtype, device=alpha.device
        )  # New (stable)

        for r in range(p):
            if r == i:  # Product over all except one
                continue
            a = alpha[r]
            b = beta[r]
            Cnew = torch.zeros_like(C)
            Cnew += a * C
            Cnew[1:, :] += b * C[:-1, :]
            Cnew[:, 1:] += b * C[:, :-1]
            Cnew[1:, 1:] += a * C[:-1, :-1]

            # New: normalize after each multiplication step
            s_r = Cnew.abs().max().clamp_min(torch.finfo(alpha.dtype).tiny)
            C = Cnew / s_r
            log_scale = log_scale + torch.log(s_r)
            # C = Cnew

        return C, log_scale  # New (also return log scale for later rescaling)
        # return C

    def multiply_by_monomial_table(self, C, g00=0.0, g10=0.0, g01=0.0, g11=0.0):
        # multiply C by g00 + g10 z1 + g01 z2 + g11 z1 z2
        out = torch.zeros_like(C)
        out += g00 * C
        out[1:, :] += g10 * C[:-1, :]
        out[:, 1:] += g01 * C[:, :-1]
        out[1:, 1:] += g11 * C[:-1, :-1]
        return out

    def _aka_cache_key(self, surrogate):
        lengthscales = (
            surrogate._model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
        )
        outputscale = surrogate._model.covar_module.outputscale.detach().reshape(-1)
        return (
            tuple(lengthscales.to(dtype=torch.float64, device="cpu").tolist()),
            tuple(outputscale.to(dtype=torch.float64, device="cpu").tolist()),
        )

    @torch.no_grad()
    def compute_AKZZA_old(self, surrogate):
        """
        Compute M = A K(Z,Z) A^T using Theorem B.2 DP.
        alpha,beta: tensors shape (p,) (you can include outputscale distribution here)
        Returns M shape (p,p).
        """

        cache_key = self._aka_cache_key(surrogate)
        cached_key = getattr(self, "_aka_cache_key_value", None)
        if cached_key == cache_key:
            return getattr(self, "_aka_cache_matrix")

        alpha, beta = self._get_kernel_alpha_beta(surrogate, dtype=torch.float64)
        p = alpha.numel()
        dtype = alpha.dtype
        device = alpha.device

        w_in, w_out = self.shapley_w_in_out(p, dtype=dtype, device=device)

        M = torch.zeros((p, p), dtype=dtype, device=device)

        # Loop over all distinct pairs of i,j (with identical entries)
        for i in range(p):
            for j in range(i, p):
                if i == j:  # First p polynomials (identical entries)
                    # Cbase = self.build_base_table_excluding_one(alpha, beta, i)
                    Cbase, log_scale = self.build_base_table_excluding_one(
                        alpha, beta, i
                    )  # New (also get log scale for later rescaling)
                    G00 = self.multiply_by_monomial_table(Cbase, g00=alpha[i])  # alpha
                    G10 = self.multiply_by_monomial_table(Cbase, g10=beta[i])  # beta z1
                    G01 = self.multiply_by_monomial_table(Cbase, g01=beta[i])  # beta z2
                    G11 = self.multiply_by_monomial_table(
                        Cbase, g11=alpha[i]
                    )  # alpha z1 z2

                    # Double sum formula over coefficients G
                    term = (
                        (w_in[:, None] * w_in[None, :]) * G11
                        - (w_in[:, None] * w_out[None, :]) * G10
                        - (w_out[:, None] * w_in[None, :]) * G01
                        + (w_out[:, None] * w_out[None, :]) * G00
                    ).sum()
                    Mij = torch.exp(log_scale) * term / (p * p)  # New
                    # Mij = term / (p*p)
                    M[i, i] = Mij
                    continue

                else:  # Remaining (p over 2) polynomials (Case i!= j)

                    # base over all r != i,j
                    Cbase, log_scale = self.build_base_table(alpha, beta, i, j)  # New
                    # Cbase = self.build_base_table(alpha, beta, i, j)

                    # build the four G^{(u,v)} tables via restrictions
                    G = {}
                    for u in (0, 1):
                        Ci = self.apply_restriction_S(Cbase, alpha[i], beta[i], u)
                        for v in (0, 1):
                            Cij = self.apply_restriction_T(Ci, alpha[j], beta[j], v)
                            # Now Cij[a,b] = G^{(u,v)}_{i,j}(a,b)
                            G[(u, v)] = Cij

                    # Double sum formula over coefficients G
                    # M_ij = (1/p^2) sum_{a,b} [ w_in(a)w_in(b) G11 - w_in(a)w_out(b) G10
                    #                           - w_out(a)w_in(b) G01 + w_out(a)w_out(b) G00 ]
                    # Note: a,b run 0..p
                    term = (
                        (w_in[:, None] * w_in[None, :]) * G[(1, 1)]
                        - (w_in[:, None] * w_out[None, :]) * G[(1, 0)]
                        - (w_out[:, None] * w_in[None, :]) * G[(0, 1)]
                        + (w_out[:, None] * w_out[None, :]) * G[(0, 0)]
                    ).sum()

                    Mij = (
                        torch.exp(log_scale) * term / (p * p)
                    )  # New (rescale by log scale)
                    # Mij = term / (p * p)

                    M[i, j] = Mij
                    M[j, i] = Mij

        M = 0.5 * (M + M.transpose(-2, -1))

        if (not torch.isfinite(M).all()) or (M.abs().max() > 1e10):
            max_abs = M.abs().max().item()
            log.warning(
                "Numerical overflow in compute_AKZZA (max abs %.3e). "
                "Applying eigenvalue clipping to stabilize.",
                max_abs,
            )
            evals, evecs = torch.linalg.eigh(M)
            floor = M.diagonal().abs().max().clamp_min(torch.finfo(M.dtype).eps) * 1e-10
            evals = evals.clamp_min(floor)
            M = evecs @ torch.diag(evals) @ evecs.transpose(-2, -1)

        object.__setattr__(self, "_aka_cache_key_value", cache_key)
        object.__setattr__(self, "_aka_cache_matrix", M)
        return M

    @torch.no_grad()
    def compute_AEA(
        self,
        surrogate,
        scale_by_emp_std: bool = False,
        force_stable: bool = False,
        precomputed_A_KZX: torch.Tensor | None = None,
        is_no_refit_step: bool = False,
    ):



        if is_no_refit_step and hasattr(self, "AKA"):
            # In no-refit steps, we can reuse the previously computed AKA (which is independent of training data) to save compute.
            AKA= self.AKA
            #No need for object.__setattr__(self, "AKA", AKA) as it does not change

        else:
            AKA = self.compute_AKZZA_old(surrogate)
            object.__setattr__(self, "AKA", AKA)

        A_KZX = (
            precomputed_A_KZX
            if precomputed_A_KZX is not None
            else self.compute_A_KZX(surrogate, force_stable=force_stable)
        )
        C = self.get_K_XX_noisy(surrogate)

        # C = L L^T
        L = psd_safe_cholesky(C.to_dense())
        # surrogate._model.prediction_strategy.lik_train_train_covar.root_decomposition().root.to_dense() (but this does not always work)

        # Solve L Y = A_KZX^T  -> Y = L^{-1} A_KZX^T
        Y = torch.linalg.solve_triangular(L, A_KZX.T, upper=False)

        # Then T = Y^T Y = A_KZX C^{-1} A_KZX^T
        T = Y.T @ Y

        AEA = AKA - T
        AEA = 0.5 * (AEA + AEA.T)
        if scale_by_emp_std:
            emp_std = surrogate._model.outcome_transform.stdvs.squeeze()
            AEA = AEA * (emp_std**2)

        return AEA

    def scale_AEA(self, AEA, surrogate):
        # Scale AEA by empirical variance of training data (if not already done inside compute_AEA)
        emp_std = surrogate._model.outcome_transform.stdvs.squeeze()
        return AEA * (emp_std**2)

    def debug_aea_diagnostics(self, surrogate):
        dtype = torch.float64
        device = surrogate._model.train_inputs[0].device

        # 1) Build pieces
        AKA = self.compute_AKZZA(surrogate).to(dtype=dtype, device=device)
        AKA = 0.5 * (AKA + AKA.T)

        A_KZX = self.compute_A_KZX(surrogate).to(dtype=dtype, device=device)

        C = self.get_K_XX_noisy(surrogate).to_dense().to(dtype=dtype, device=device)
        C = 0.5 * (C + C.T)

        # 2) Stable correction term T = A_KZX C^{-1} A_KZX^T = Y^T Y
        Lc = psd_safe_cholesky(C)
        Y = torch.linalg.solve_triangular(Lc, A_KZX.T, upper=False)
        T = Y.T @ Y
        T = 0.5 * (T + T.T)

        # 3) AEA
        AEA = AKA - T
        AEA = 0.5 * (AEA + AEA.T)

        # 4) Eigenvalue diagnostics
        eig_AKA = torch.linalg.eigvalsh(AKA)
        eig_T = torch.linalg.eigvalsh(T)
        eig_AEA = torch.linalg.eigvalsh(AEA)

        lambda_min_AKA = eig_AKA.min()
        lambda_min_T = eig_T.min()
        lambda_min_AEA = eig_AEA.min()

        # 5) Generalized domination test:
        # S = AKA^{-1/2} T AKA^{-1/2}
        # If theory/numerics are good, lambda_max(S) should be <= 1 (up to tiny eps)
        LA = psd_safe_cholesky(AKA)
        Z = torch.linalg.solve_triangular(LA, T, upper=False)
        S = torch.linalg.solve_triangular(LA, Z.T, upper=False).T
        S = 0.5 * (S + S.T)
        eig_S = torch.linalg.eigvalsh(S)
        lambda_max_S = eig_S.max()

        return {
            "lambda_min_AKA": lambda_min_AKA.item(),
            "lambda_min_T": lambda_min_T.item(),
            "lambda_min_AEA": lambda_min_AEA.item(),
            "lambda_max_AKA_invhalf_T_invhalf": lambda_max_S.item(),
            "AKA_abs_max": AKA.abs().max().item(),
            "T_abs_max": T.abs().max().item(),
            "AEA_abs_max": AEA.abs().max().item(),
            "AKA_asym_max": (AKA - AKA.T).abs().max().item(),
            "T_asym_max": (T - T.T).abs().max().item(),
            "AEA_asym_max": (AEA - AEA.T).abs().max().item(),
        }

    @torch.no_grad()
    def compute_AF_PPV(self, surrogate, precomputed_A_KZX: torch.Tensor | None = None):
        return self.compute_AEA_new(
            surrogate,
            scale_by_emp_std=True,
            precomputed_A_KZX=precomputed_A_KZX,
        )

    def get_prop_gt(self):
        raise NotImplementedError(
            "Ground truth property values are not implemented for this application."
        )



    @property
    def Z(self) -> str:
        return self._Z

    @property
    def A(self) -> str:
        return self._A

    @property
    def X0(self) -> str:
        return self._X0

    @property
    def candidate_set(self) -> str:
        return self._candidate_set

    @property
    def candidate_idx_Z(self) -> str:
        return self._candidate_idx_Z

    def termination_criterion(self, property_posterior) -> torch.Tensor:
        if self.lazy_setup:
            assert self.lazy_setup_conducted, f"run_lazy_setup() must be called first."

        return False  # Not implemented

@dataclass(frozen=True)
class DummyShapleyApplication(ShapleyApplication):
    def sample_configs(self) -> torch.Tensor:
        baseline_config = torch.zeros(self.blackbox_function.dim, dtype=torch.float64)
        candidate_config = torch.ones(self.blackbox_function.dim, dtype=torch.float64)

        return baseline_config, candidate_config

    def get_blackbox_dim(self) -> int:
        return self.blackbox_function.dim

    def get_prop_gt(self, surrogate=None):
        #Return dummys
        return torch.zeros((self.amount_players, 1))

    def run_lazy_setup(
        self,
        blackbox_function: BaseBlackboxFunction,
        seed: int,
        amount_iterations: int = None,
        scalable_mode: bool = False,
        acquisition_optimizer: BaseAcquisitionOptimizer = None,
    ):
        #Run run_lazy_setup of parent
        super().run_lazy_setup(blackbox_function, seed, amount_iterations, scalable_mode, acquisition_optimizer) 

        temp_id= torch.concat([self.X0[0:1,:], self.candidate_set[:2,:], self.X0[3:,:], self.X0[1:2,:]])
        temp_cand= torch.concat([self.X0[2:3,:], self.candidate_set[2:,:]]) #self.X0[1:2,:]

        #object.__setattr__(self, "X0", temp_id)
        object.__setattr__(self, "_X0", temp_id)
        object.__setattr__(self, "_X0_binary", temp_id)

        #object.__setattr__(self, "candidate_set", temp_cand)
        object.__setattr__(self, "_candidate_set", temp_cand)
        object.__setattr__(self, "_candidate_set_binary", temp_cand)

@dataclass(frozen=True)
class BotorchShapleyApplication(ShapleyApplication):
    def sample_configs(self) -> torch.Tensor:
        # Sample baseline and candidate configurations
        lower_bound = self.blackbox_function._bounds[0, :]
        upper_bound = self.blackbox_function._bounds[1, :]

        baseline_config = lower_bound + torch.rand(lower_bound.shape) * (
            upper_bound - lower_bound
        )
        candidate_config = lower_bound + torch.rand(lower_bound.shape) * (
            upper_bound - lower_bound
        )

        return baseline_config, candidate_config

    def get_blackbox_dim(self) -> int:
        return self.blackbox_function.dim
    
    def get_prop_gt(self):
        #Return dummy 0s for scalability experiments
        log.warning(
            "Using dummy ground truth values."
        )
        return torch.zeros((self.amount_players, 1))  # by convention this is (p,1)

@dataclass(frozen=True)
class ShapiqShapleyApplication(ShapleyApplication):
    def sample_configs(self) -> torch.Tensor:
        baseline_config = torch.zeros(self.blackbox_function.dim, dtype=torch.float64)
        candidate_config = torch.ones(self.blackbox_function.dim, dtype=torch.float64)

        return baseline_config, candidate_config

    def get_blackbox_dim(self) -> int:
        return self.blackbox_function.dim

    def get_prop_gt(self, surrogate=None):
    #def get_prop_gt(self):
        from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

        if hasattr(self.blackbox_function, "shapiq_game") and isinstance(
            self.blackbox_function.shapiq_game, TreeSHAPIQXAI
        ):
            game = self.blackbox_function.shapiq_game
            sv_exact_dict = game.exact_values(index="SV", order=1).dict_values
            sv_exact = np.array(
                [
                    sv_exact_dict.get((i,), 0.0)
                    for i in range(0, self.blackbox_function.dim)
                ]
            )

            return torch.tensor(sv_exact, dtype=torch.float64).unsqueeze(
                1
            )  ##by convention this is of shape (p,1)

        elif surrogate is None:
            log.warning(
                "Using dummy ground truth values. Prop GT should be computed later."
            )
            return torch.zeros((self.amount_players, 1))  # by convention this is (p,1)

        else:
            if self.amount_players < 20:
                #Compute ground truth via ShapIQ
                prop_gt= self.get_exact_siq_values(self.X0.shape[0],
                                                self.blackbox_function,
                                                surrogate
                                                )
                
                return None
                
            else:
                raise NotImplementedError(
                    "Ground truth property values cannot be computed exhaustively for p >= 20."
                )

        # else:
        #     raise NotImplementedError(
        #         "Ground truth property values are not implemented for this application."
        #     )
        




@dataclass(frozen=True)
class YahpoShapleyApplication(ShapleyApplication):
    def sample_configs(self) -> torch.Tensor:

        # if self.blackbox_function.yahpo_name

        def get_xgboost_dart_config():
            temp_config = self.blackbox_function.yahpo_opt_space.sample_configuration(1)

            while temp_config["booster"] != "dart":
                temp_config = (
                    self.blackbox_function.yahpo_opt_space.sample_configuration(1)
                )

            return temp_config

        if self.blackbox_function.yahpo_name == "rbv2_xgboost":
            # Simple workaround to ensure that booster is dart
            baseline_config = get_xgboost_dart_config()

            fign = min(
                self.blackbox_function.task_id_index,
                self.blackbox_function.booster_index,
            )
            sign = max(
                self.blackbox_function.task_id_index,
                self.blackbox_function.booster_index,
            )

            baseline_config_numeric = np.concatenate(
                [
                    baseline_config.get_array()[:fign],
                    baseline_config.get_array()[fign + 1 : sign],
                    baseline_config.get_array()[sign + 1 :],
                ]
            )

        else:
            baseline_config = (
                self.blackbox_function.yahpo_opt_space.sample_configuration(1)
            )

            baseline_config_numeric = np.concatenate(
                [
                    baseline_config.get_array()[: self.blackbox_function.task_id_index],
                    baseline_config.get_array()[
                        self.blackbox_function.task_id_index + 1 :
                    ],
                ]
            )

        # baseline_config_numeric= baseline_config.get_array()[1:] #Ignore OpenML ID at idx 0 (as this is always identical)

        temp_candidate_config_numeric = None  # torch.zeros(amount_features)

        while not (
            baseline_config_numeric
            != (
                temp_candidate_config_numeric
                if temp_candidate_config_numeric is not None
                else baseline_config_numeric
            )
        ).all():
            if self.blackbox_function.yahpo_name == "rbv2_xgboost":
                # Simple workaround to ensure that booster is dart
                temp_candidate_config = get_xgboost_dart_config()

                temp_candidate_config_numeric = np.concatenate(
                    [
                        temp_candidate_config.get_array()[:fign],
                        temp_candidate_config.get_array()[fign + 1 : sign],
                        temp_candidate_config.get_array()[sign + 1 :],
                    ]
                )

            else:
                temp_candidate_config = (
                    self.blackbox_function.yahpo_opt_space.sample_configuration(1)
                )

                # temp_candidate_config = (
                # self.blackbox_function.yahpo_opt_space.sample_configuration(1)
                # )

                temp_candidate_config_numeric = np.concatenate(
                    [
                        temp_candidate_config.get_array()[
                            : self.blackbox_function.task_id_index
                        ],
                        temp_candidate_config.get_array()[
                            self.blackbox_function.task_id_index + 1 :
                        ],
                    ]
                )
                # temp_candidate_config.get_array()[1:] #Ignore OpenML ID at idx 0

        # wie schauen welche anderen features nuisance sind (zb repl)

        candidate_config = temp_candidate_config
        candidate_config_numeric = temp_candidate_config_numeric

        assert (
            baseline_config_numeric != candidate_config_numeric
        ).all(), "Baseline and candidate config are identical."
        assert (
            int(baseline_config[self.blackbox_function.task_id_column_name])
            == int(candidate_config[self.blackbox_function.task_id_column_name])
            == self.blackbox_function.instance
        ), "Baseline and candidate config have wrong dataset IDs."


        return torch.tensor(baseline_config_numeric), torch.tensor(
            candidate_config_numeric
        )
        # Map to numerical values

        # only return features where values are different

    def get_blackbox_dim(self) -> int:
        temp_config = self.blackbox_function.yahpo_opt_space.sample_configuration(1)

        if self.blackbox_function.yahpo_name == "rbv2_xgboost":
            return (
                temp_config.get_array().shape[0] - 2
            )  # Ignore OpenML ID at idx 0 and booster

        else:
            return temp_config.get_array().shape[0] - 1  ##Ignore OpenML ID at idx 0
        # We dont want openml id as input, however, in evaluate this has to be added

    def get_prop_gt(self, surrogate=None):
        if surrogate is None:
            log.warning(
                "Using dummy ground truth values. Prop GT should be computed later."
            )
            return torch.zeros((self.amount_players, 1))  # by convention this is (p,1)

        else:
            if self.amount_players < 20:
                #Compute ground truth via ShapIQ
                prop_gt= self.get_exact_siq_values(self.X0.shape[0],
                                                self.blackbox_function,
                                                surrogate
                                                )
                
                return None
                
            else:
                raise NotImplementedError(
                    "Ground truth property values cannot be computed exhaustively for p >= 20."
                )
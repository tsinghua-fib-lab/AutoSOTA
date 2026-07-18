from __future__ import annotations

"""Exact single‑output Gaussian‑process surrogate.

The class supports various kernel families, optional observation noise, and two
hyper‑parameter inference modes (MLM or NUTS). Not all variants are fully 
implemented for now.
"""

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Literal

import torch
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models.fully_bayesian import (FullyBayesianSingleTaskGP,
                                           SaasFullyBayesianSingleTaskGP)
from botorch.models.gp_regression_mixed import MixedSingleTaskGP, SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import (ChainedInputTransform,
                                             FilterFeatures, Log10, Warp)
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior, get_matern_kernel_with_gamma_prior)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from linear_operator import to_linear_operator
from linear_operator.operators import (CatLinearOperator,
                                       KroneckerProductLinearOperator)
from torch import Tensor

from typing import Optional

__all__ = [
    # "KernelChoice",
    "FitMethod",
    "Optimizer",
    "FitConfig",
    "MLMConfig",
    "NUTSConfig",
    "GPSurrogateConfig",
    "GPSurrogate",
    "NoiseConfig",
    "GaussianNoiseConfig",
    "ConstantNoiseConfig",
    "KernelConfig",
    "RBFKernelConfig",
    "Matern52KernelConfig",
    "HammingKernelConfig",
]

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public enums / aliases
# -----------------------------------------------------------------------------


# class KernelChoice(str, Enum):
#     RBF = "rbf"
#     MATERN52 = "matern52"  # ν = 5⁄2
#     DEFAULT = "default"

FitMethod = Literal["mlm", "nuts"]
Optimizer = Literal["default"]
MissingStrategy = Literal["mean"]

# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseConfig:
    pass


NoisePriorType = Literal["lognormal", "gamma"]


@dataclass(frozen=True)
class GaussianNoiseConfig(NoiseConfig):
    min_inferred_noise_level: float = 1e-4
    prior_type: NoisePriorType = (
        "lognormal"  # Default according to https://github.com/meta-pytorch/botorch/discussions/2451
    )


@dataclass(frozen=True)
class ConstantNoiseConfig(NoiseConfig):
    noise_level: float = 1e-4


@dataclass(frozen=True)
class KernelConfig:
    pass


@dataclass(frozen=True)
class RBFKernelConfig(KernelConfig):
    min_lengthscale: float = 2.5e-2


@dataclass(frozen=True)
class Matern52KernelConfig(KernelConfig):
    min_lengthscale: float = 2.5e-2


@dataclass(frozen=True)
class HammingKernelConfig(KernelConfig):
    min_lengthscale: float = 1e-6


@dataclass(frozen=True)
class FitConfig:
    pass


@dataclass(frozen=True)
class MLMConfig(FitConfig):
    optimizer: Optimizer = "default"
    amount_restarts: int = 5
    run_all_attempts: int = False
    warmstart: bool = False
    refit_interval: int = 1 #Indicates how often HPs are refitted
    refit_schedule: Optional[Literal["init_64_factor_4"]] = None


@dataclass(frozen=True)
class NUTSConfig(FitConfig):
    max_tree_depth: int = 6
    warmup_steps: int = 512
    num_samples: int = 256
    thinning: int = 16
    #refit_interval not implemented yet


@dataclass(frozen=True)
class GPSurrogateConfig:
    # kernel: KernelChoice = KernelChoice.DEFAULT
    kernel_config: KernelConfig = field(default_factory=Matern52KernelConfig)
    # noise: NoiseChoice = "none"
    noise_config: NoiseConfig = field(default_factory=GaussianNoiseConfig)
    missing_strategy: MissingStrategy = "mean"
    warp: bool = False
    fit_config: FitConfig = field(default_factory=MLMConfig)


# -----------------------------------------------------------------------------
# GPSurrogate
# -----------------------------------------------------------------------------


class CategoricalKronGridKernel(Kernel):
    r"""Copied and modified from /botorch/models/kernels/categorical.py.
    Overwrites behavior to return KroneckerProductLazyTensor for appropriate inputs

    A Kernel for categorical features.

    Computes `exp(-dist(x1, x2) / lengthscale)`, where
    `dist(x1, x2)` is zero if `x1 == x2` and one if `x1 != x2`.
    If the last dimension is not a batch dimension, then the
    mean is considered.

    Note: This kernel is NOT differentiable w.r.t. the inputs.
    """

    has_lengthscale = True

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:

        # Infer amount of players and coalitions
        amount_players = x1.shape[-1]
        amount_coalitions = 2**amount_players
        assert (x1.shape[1] == amount_players) and (
            x2.shape[1] == amount_players
        ), f"CategoricalKronGridKernel only supports inputs with shape (n, {amount_players})"

        ###############################
        def get_kron_test_test_covar():
            # Concatenate amount_players times the identity matrix in 2d
            kron_list = tuple(
                torch.exp(
                    -torch.tensor([[0, 1], [1, 0]], dtype=x1.dtype)
                    / (self.lengthscale[0, lengthscale_idx] * amount_players)
                )  # Divide by lengthscale and amount of dimensions to match mean division in standard case
                for lengthscale_idx in range(self.lengthscale.shape[1])
            )

            res = KroneckerProductLinearOperator(*kron_list)

            if diag:
                res = torch.diagonal(res, dim1=-1, dim2=-2)
            return res

        ###############################
        def get_standard_covar(_x1, _x2):
            # Added: Efficient route (avoiding OOM) for diagonal test-test
            if diag and (_x1.shape[0] == _x2.shape[0]) and (_x1 == _x2).all():
                return torch.ones(
                    _x1.shape[0], dtype=_x1.dtype, device=_x1.device
                )  # Covariance is 1 on the diagonal (as distance is 0) and we only return the diagonal

            # Standard variant without Kronecker optimizations
            # Implementatio of /botorch/models/kernels/categorical.py
            delta = _x1.unsqueeze(-2) != _x2.unsqueeze(-3)
            dists = delta / self.lengthscale.unsqueeze(-2)
            if last_dim_is_batch:
                dists = dists.transpose(-3, -1)
            else:
                dists = dists.mean(-1)
            res = torch.exp(-dists)
            if diag:
                res = torch.diagonal(res, dim1=-1, dim2=-2)
            return res

        ###############################
        ###############################
        if (x1.shape[0] >= amount_coalitions) and (
            x2.shape[0] >= amount_coalitions
        ):  # Efficient Kronecker product in case that both x1 and x2 contain full grid (perhaps with current training data)
            assert not last_dim_is_batch

            if (
                (x1.shape[0] == amount_coalitions)
                and (x2.shape[0] == amount_coalitions)
                and (x1 == x2).all()
            ):
                # Case: Test-test covariance (Both inputs contain full grid)

                # import time
                # start = time.perf_counter()

                full_covar = get_kron_test_test_covar()

                # end = time.perf_counter()
                # log.info("Time " + str(end - start))

                # if not diag:
                #     assert torch.allclose(full_covar[:256, :256].to_dense(),
                #                         get_standard_covar(x1[:256, :], x2[:256, :]),
                #                         atol= 1e-15), f"CategoricalKronGridKernel: Full covariance does not match standard implementation."

                return full_covar

            else:
                # Case Test-test covariance and test-train covariance (x1 is full grid, x2 contains grid + train)
                assert x1.shape[0] == amount_coalitions

                amount_train_samples = x2.shape[0] - amount_coalitions

                x2_train = x2[:amount_train_samples,]
                x2_grid = x2[amount_train_samples:,]

                assert (
                    x2_grid == x1
                ).all(), f"The last samples of x2 must correspond to the full grid when x1 contains the full grid."

                # compute test-train
                test_train_covar = get_standard_covar(x1, x2_train)

                # compute test-test
                test_test_covar = get_kron_test_test_covar()

                full_covar = CatLinearOperator(
                    *[to_linear_operator(test_train_covar), test_test_covar], dim=1
                )


                return full_covar

        else:
            # Standard mode in case that only x1 or x2 are on full grid => Use standard implementation
            return get_standard_covar(x1, x2)


class GPSurrogate:
    """Minimal wrapper for a single‑output Gaussian‑process surrogate for regression."""

    def __init__(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        *,
        config: GPSurrogateConfig,
        cat_dims: list[int] | None = None,
        log_trafo_dims: list[int] | None = None,
        bounds: (
            torch.Tensor | None
        ) = None,  # A 2 x d-dim tensor of lower and upper bounds for the inputs.
        shapley_configs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        # ---------------- helper -----------------------------
        def _build_base_kernel(
            kernel_config: KernelConfig,
            *,
            batch_shape: torch.Size | None = None,
            ard_num_dims: int,
            active_dims: Sequence[int] | None = None,
        ):
            if (isinstance(kernel_config, RBFKernelConfig)) or (
                isinstance(kernel_config, Matern52KernelConfig)
                or isinstance(kernel_config, HammingKernelConfig)
            ):
                # Copied and modified from get_covar_module_with_dim_scaled_prior (/botorch/models/utils/gpytorch_modules.py)
                # Overcomes bugs discussed in https://github.com/meta-pytorch/botorch/issues/2542

                from math import log, sqrt

                SQRT2 = sqrt(2)
                SQRT3 = sqrt(3)

                base_class = (
                    RBFKernel
                    if (isinstance(kernel_config, RBFKernelConfig))
                    else (
                        MaternKernel
                        if (isinstance(kernel_config, Matern52KernelConfig))
                        else CategoricalKronGridKernel  # CategoricalKernel
                    )
                )  # Added line
                # base_class = RBFKernel if (isinstance(kernel_config, RBFKernelConfig)) else MaternKernel #Modified line

                lengthscale_prior = LogNormalPrior(
                    loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3
                )
                lengthscale_prior_mode = lengthscale_prior.mode

                base_kernel = base_class(
                    ard_num_dims=ard_num_dims,
                    batch_shape=batch_shape,
                    lengthscale_prior=lengthscale_prior,
                    lengthscale_constraint=GreaterThan(
                        kernel_config.min_lengthscale,  # 2.5e-2,
                        initial_value=max(
                            lengthscale_prior_mode, kernel_config.min_lengthscale
                        ),
                    ),
                    active_dims=active_dims,
                )


                return ScaleKernel(base_kernel)

            else:
                raise NotImplementedError()

        # ---------------- main definition -----------------------------
        self.config = config
        self.cat_dims = cat_dims
        self.log_trafo_dims = log_trafo_dims
        self.bounds = bounds
        self._init_kwargs = {
            "config": config,
            "cat_dims": None if cat_dims is None else list(cat_dims),
            "log_trafo_dims": (
                None if log_trafo_dims is None else list(log_trafo_dims)
            ),
            "bounds": None if bounds is None else bounds.clone(),
            "shapley_configs": (
                None
                if shapley_configs is None
                else tuple(cfg.clone() for cfg in shapley_configs)
            ),
        }

        # -----------------------------------------------------------------
        # 1) learn imputation values based on training data (to avoid data leakage)
        train_X = self._impute(train_X, set_imputer_values=True)

        # impute should not be needed for hamming

        # -----------------------------------------------------------------
        # 2) set up input transformations
        if not isinstance(config.kernel_config, HammingKernelConfig):
            tfs = {}

            if len(log_trafo_dims) > 0:
                assert (
                    train_X[:, log_trafo_dims] > 0
                ).all(), f"Selected features for log-transformation must exclusively contain values greater 0."

                with warnings.catch_warnings(record=True) as w:
                    # Surpress warning in botorch/models/transforms/input.py:1027
                    warnings.simplefilter("always", UserWarning)
                    warnings.filterwarnings(
                        "ignore", message="To copy construct from a tensor*"
                    )

                    tfs["Log10"] = Log10(
                        indices=log_trafo_dims,
                        transform_on_train=True,
                        transform_on_eval=True,
                    )

                    # Bounds must be updated accordingly (_transform only operates on indices)
                    self.bounds = tfs["Log10"]._transform(self.bounds)

            if not self.config.warp:
                tfs["Normalize"] = Normalize(
                    d=train_X.shape[-1],
                    indices=[i for i in range(train_X.shape[-1]) if not i in cat_dims],
                    bounds=bounds,
                    learn_bounds=False,
                    transform_on_train=True,
                    transform_on_eval=True,
                )

            if self.config.warp:
                tfs["Warp"] = Warp(
                    d=train_X.shape[-1],
                    indices=[
                        i
                        for i in range(train_X.shape[-1])
                        if not (len(cat_dims) > 0 and i in cat_dims)
                    ],
                    transform_on_train=True,
                    transform_on_eval=True,
                    concentration0_prior=LogNormalPrior(
                        0.0, 0.75**0.5
                    ),  # Uses priors with median at 1. For a=b=1, the Kumaraswamy CDF is the identity function.
                    concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
                    bounds=bounds,  # Explicitly set bounds as opposed to inferring them from training data.
                )

                # Input warping contains an internal normalization module.

                # As Kumaraswamy expects inputs in the unit cube, we can't just use normalization with bounds learned on training set
                # (as this may lead to values outside the unit cube at test time), but need to pass them explicitly.

            input_transform = ChainedInputTransform(**tfs)

        else:
            assert (
                log_trafo_dims is None or len(log_trafo_dims) == 0
            ), "Log-transformation not supported with Hamming kernel."
            assert (
                cat_dims is None or len(cat_dims) == 0
            ), "Hamming kernel assumes all dimensions are categorical."
            assert (
                self.config.warp == False
            ), "Input warping not supported with Hamming kernel."

            from botorch.models.transforms.input import InputTransform

            class BinaryValueRemapping(InputTransform):
                """Remaps continuous values to binary 0/1 based on dictionary."""

                def __init__(
                    self,
                    baseline_config: torch.Tensor,
                    candidate_config: torch.Tensor,
                    transform_on_train: bool = True,
                    transform_on_eval: bool = True,
                    transform_on_fantasize: bool = True,
                ):
                    super().__init__()

                    self.baseline_config = baseline_config
                    self.candidate_config = candidate_config

                    self.transform_on_train = transform_on_train
                    self.transform_on_eval = transform_on_eval
                    self.transform_on_fantasize = transform_on_fantasize

                    self.is_one_to_many = False

                def transform(self, X: torch.Tensor) -> torch.Tensor:
                    # import time
                    # start = time.perf_counter()

                    X_transformed = X.detach().clone()

                    # Squeeze and unsqueeze for marginal variance case (EIG-EP)
                    squeezed = False
                    if X.ndim == 3:
                        squeezed = True
                        X_transformed = X_transformed.squeeze(1)
                        X = X.squeeze(1)

                    for dim in range(X.shape[-1]):
                        baseline_val = self.baseline_config[dim].detach().clone()
                        # torch.tensor(
                        #     self.baseline_config[dim]
                        # )  # .item()
                        candidate_val = self.candidate_config[dim].detach().clone()
                        # torch.tensor(
                        #     self.candidate_config[dim]
                        # )  # .item()

                        X_transformed[:, dim] = torch.where(
                            torch.isclose(X[:, dim], candidate_val),  # , atol= 1e-3
                            torch.tensor(1.0, device=X.device),
                            torch.tensor(0.0, device=X.device),
                        )

                        # print(torch.where(
                        #     torch.isclose(X[:, dim], candidate_val), #, atol= 1e-3
                        #     torch.tensor(1.0, device=X.device),
                        #     torch.tensor(0.0, device=X.device)
                        # ))

                        # print(X_transformed.shape)

                    # end = time.perf_counter()
                    # log.info("Time " + str(end - start))

                    if squeezed:
                        X_transformed = X_transformed.unsqueeze(1)

                    return X_transformed

                def untransform(self, X):
                    X_untransformed = torch.zeros_like(X, dtype=torch.float64)

                    for dim in range(X.shape[-1]):
                        baseline_val = self.baseline_config[dim].detach().clone()
                        # baseline_val = torch.tensor(
                        #     self.baseline_config[dim]
                        # )  # .item()

                        candidate_val = self.candidate_config[dim].detach().clone()
                        # candidate_val = torch.tensor(
                        #     self.candidate_config[dim]
                        # )  # .item()

                        # X_untransformed[:, dim] = torch.where(
                        #     X[:, dim] == 1.0,
                        #     torch.tensor(candidate_val.detach().clone(), device=X.device),
                        #     torch.tensor(baseline_val.detach().clone(), device=X.device),
                        # )

                        X_untransformed[:, dim] = torch.where(
                            X[:, dim] == 1.0, candidate_val, baseline_val
                        )

                    return X_untransformed

            # tfs = {}

            baseline_config = shapley_configs[0]
            candidate_config = shapley_configs[1]

            # assert that baseline and candidate differ in all features
            assert (
                baseline_config != candidate_config
            ).all(), "Baseline and candidate configuration must differ in all features for Hamming kernel with binary remapping."

            input_transform = BinaryValueRemapping(
                baseline_config=baseline_config, candidate_config=candidate_config
            )


        # -----------------------------------------------------------------
        # 3) initialize GP model
        if isinstance(config.noise_config, ConstantNoiseConfig):
            train_Yvar = torch.full_like(train_Y, config.noise_config.noise_level)
        else:
            train_Yvar = None
        # Noise-free uses FixedNoiseGaussianLikelihood with small constant (1e-4) and else it uses a standard GaussianLikelihood with inferred noise level (get_gaussian_likelihood_with_lognormal_prior) (holds for both SingleTaskGP and MixedSingleTaskGP)

        if isinstance(config.fit_config, MLMConfig):
            if len(cat_dims) > 0:
                cont_kernel_factory = partial(_build_base_kernel, config.kernel_config)

                model = MixedSingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    cat_dims=cat_dims,
                    train_Yvar=train_Yvar,
                    cont_kernel_factory=cont_kernel_factory,
                    input_transform=input_transform,
                    outcome_transform=Standardize(m=1),
                )

            else:
                covar_module = _build_base_kernel(
                    config.kernel_config, ard_num_dims=train_X.shape[-1]  # ard_num_dims
                )

                def get_likelihood():
                    # Copied and modified from get_gaussian_likelihood_with_gamma_prior and get_gaussian_likelihood_with_lognormal_prior (/botorch/models/utils/gpytorch_modules.py)
                    # Overcomes bugs discussed in https://github.com/meta-pytorch/botorch/issues/2542
                    batch_shape = (
                        torch.Size()
                    )  # if batch_shape is None else batch_shape #(Commended out)

                    if config.noise_config.prior_type == "lognormal":
                        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
                        noise_prior_mode = noise_prior.mode

                    else:
                        noise_prior = GammaPrior(1.1, 0.05)
                        noise_prior_mode = (
                            noise_prior.concentration - 1
                        ) / noise_prior.rate

                    # min_inferred_noise_level= min(config.noise_config.min_inferred_noise_level, noise_prior_mode)

                    # if noise_prior_mode < config.noise_config.min_inferred_noise_level:
                    #     log.info(
                    #         f"""The mode of the noise prior ({noise_prior_mode:.2e}) is smaller than the specified minimum inferred noise level ({config.noise_config.min_inferred_noise_level:.2e}). The minimum noise level is hence overwritten."""
                    #     )

                    return GaussianLikelihood(
                        noise_prior=noise_prior,
                        batch_shape=batch_shape,
                        noise_constraint=GreaterThan(
                            config.noise_config.min_inferred_noise_level,
                            initial_value=max(
                                noise_prior_mode,
                                config.noise_config.min_inferred_noise_level,
                            ),
                        ),  # transform=None,
                    )

                model = SingleTaskGP(
                    train_X=train_X,
                    train_Y=train_Y,
                    train_Yvar=train_Yvar,
                    likelihood=get_likelihood() if train_Yvar is None else None,
                    covar_module=covar_module,
                    input_transform=input_transform,
                    outcome_transform=Standardize(m=1),
                )

                # No prior for outputscale requiured

        else:
            if len(cat_dims) > 0:
                raise NotImplementedError(
                    "Fully Bayesian GP not implemented yet (FullyBayesianSingleTaskGP does not support categorical parameters)."
                )

            else:
                covar_module = None 

                model = SaasFullyBayesianSingleTaskGP(
                    train_X=train_X,  # Same as in SingleTaskGP
                    train_Y=train_Y,  # Same as in SingleTaskGP
                    train_Yvar=train_Yvar,  # Same as in SingleTaskGP ("Inferred if None.")
                    input_transform=input_transform,  # Same as in SingleTaskGP
                    outcome_transform=Standardize(
                        m=1
                    ),  # Same as in SingleTaskGP (An outcome transform that is applied to the training data during instantiation and to the posterior during inference (that is, the Posterior obtained by calling .posterior on the model will be on the original scale). Note that .train() will be called on the outcome transform during instantiation of the model.)
                )
                # likelihood: Cannot be specified here (uses default GaussianLikelihood)
                # covar_module: Cannot be specified here (uses "Matern-5/2 kernel and dimension-scaled priors")

        self._model = model

        # Case: Shapley values with product kernel and test data on Kronecker grid
        # gpytorch.kernels.GridKernel
        # gpytorch.kernels.ProductStructureKernel(

    def fit(self):
        # Automatically calls train() and eval()
        if isinstance(self._model, SaasFullyBayesianSingleTaskGP):
            fit_fully_bayesian_model_nuts(
                self._model,
                max_tree_depth=self.config.fit_config.max_tree_depth,
                warmup_steps=self.config.fit_config.warmup_steps,
                num_samples=self.config.fit_config.num_samples,
                thinning=self.config.fit_config.thinning,
                disable_progbar=False,
            )

        else:
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)

            if self.config.fit_config.optimizer == "default":
                optimizer = None
                optimizer_kwargs = None

                try:
                    mll = fit_gpytorch_mll(
                        mll,
                        optimizer=optimizer,
                        optimizer_kwargs=optimizer_kwargs,
                        max_attempts=self.config.fit_config.amount_restarts,  # See https://github.com/pytorch/botorch/issues/1724
                        pick_best_of_all_attempts=self.config.fit_config.run_all_attempts,  # Set True for multi-start optimization, and False to run the attempts until one succeeds (to avoid exceptions).
                    )  # after executing this, the added data point is missing again

                except:
                    # Catch error if MLL fails. (This should only occur rarely due to restarts.)
                    log.info(
                        f"""MLL failed (with {str(self.config.fit_config.amount_restarts)} attempts). Consider enhancing the amount of attempts."""
                    )

            else:
                raise NotImplementedError

    def update_data(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        """Update the training data of the GP surrogate, maintaining current hyperparameters as warmstart for further hyperparameter fitting.

        Args:
            train_X (torch.Tensor): New input data points.
            train_Y (torch.Tensor): Corresponding output values for the new input data points.
        """
        #Condition_on_observations is not suitable here, as it would e.g. not update standardization module
        is_eval_mode = not self._model.training
        old_model = self._model

        rebuilt_surrogate = type(self)(
            train_X=train_X,
            train_Y=train_Y,
            **self._init_kwargs,
        )
        self._copy_warmstart_parameters(
            source_model=old_model,
            target_model=rebuilt_surrogate._model,
        )

        if is_eval_mode:
            rebuilt_surrogate._model.eval()

        self.__dict__.update(rebuilt_surrogate.__dict__) #Updates the current object itself

    def forward(self, X, observation_noise=False):
        X = self._impute(X, set_imputer_values=False)


        with torch.no_grad():
            ppd = self._model.posterior(X, observation_noise=observation_noise).mvn
            # Automatically applies input transformations (see /botorch/models/gpytorch.py)

            # Check if observation_noise= True makes covariance diagonal?

        return ppd

    def forward_lazy_covar(self, X, observation_noise=False):
        # Forward pass with lazy covariance matrix (to save memory)
        X = self._impute(X, set_imputer_values=False)

        with torch.no_grad():
            lazy_covar = self._model.posterior(
                X, observation_noise=observation_noise
            ).mvn.lazy_covariance_matrix

        return lazy_covar

    def forward_marg_vars(self, X, observation_noise=False):
        # Forward pass with conditional indpendence assumption of test points

        X = self._impute(X, set_imputer_values=False)

        with torch.no_grad():
            ppd_batched = self._model.posterior(
                X.unsqueeze(1), observation_noise=observation_noise
            )

            mean = ppd_batched.mean.squeeze()
            marg_vars = ppd_batched.variance.squeeze().diag()

            ppd_marg_vars = MultivariateNormal(mean=mean, covariance_matrix=marg_vars)

        return ppd_marg_vars

    def __call__(self, X):
        return self.forward(X)

    @staticmethod
    def _copy_warmstart_parameters(source_model, target_model) -> None:
        """Copy learned hyperparameters while leaving data-dependent state intact."""
        source_parameters = dict(source_model.named_parameters())

        with torch.no_grad():
            for name, target_parameter in target_model.named_parameters():
                source_parameter = source_parameters.get(name)
                if source_parameter is None:
                    continue
                if source_parameter.shape != target_parameter.shape:
                    continue
                target_parameter.copy_(source_parameter.detach())

    # ---------- internal ----------
    def _impute(
        self, X: torch.Tensor, set_imputer_values: bool = False
    ) -> torch.Tensor:

        if self.config.missing_strategy == "mean":
            if set_imputer_values:
                # Set imputer values only once based on training data
                self.imputer_means = torch.nanmean(X, dim=0)

                # If values are nan, set to 0.5 (middle of normalized range [0, 1])
                nan_mask = torch.isnan(self.imputer_means)
                self.imputer_means[nan_mask] = 0.5

            X_filled = X.detach().clone()
            nan_mask = torch.isnan(X_filled)
            X_filled[nan_mask] = self.imputer_means.expand_as(X_filled)[nan_mask]
            return X_filled

        else:
            raise NotImplementedError()

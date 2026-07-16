"""GPyTorch-backed regression for tangent basis coefficients."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import gpytorch
import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.optim.fit import fit_gpytorch_mll_torch

from active_wasserstein.geometry import TangentBasis
from active_wasserstein.inference.kernels import KernelSpec
from active_wasserstein.inference.observation import (
    NoiseInitializer,
    TangentObservation,
)
from active_wasserstein.inference.predictive import PredictiveProcess
from active_wasserstein.utils.scaling import InputScaler, OutputScaler

MeanFunction = Callable[[float], np.ndarray]
logger = logging.getLogger(__name__)


class _ScalarTimeGP(gpytorch.models.ExactGP):
    """Single-output exact GP on time."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        kernel: gpytorch.kernels.Kernel,
        mean_module: gpytorch.means.Mean | None = None,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, X: torch.Tensor) -> torch.Tensor:
        return X


@dataclass
class GPyTorchHilbertPredictive(PredictiveProcess):
    """Posterior state backed by trained GPyTorch models."""

    basis: TangentBasis
    kernel_spec: KernelSpec
    scales: np.ndarray
    prior_variance: float
    mean_function: Optional[MeanFunction]
    models: List[_ScalarTimeGP]
    likelihoods: List[gpytorch.likelihoods.Likelihood]
    device: torch.device
    dtype: torch.dtype
    jitter: float
    warp: Optional[Any] = None
    input_scaler: Optional[InputScaler] = None
    output_scaler: Optional[OutputScaler] = None

    def __post_init__(self) -> None:
        self.scales = np.asarray(self.scales, dtype=float)
        if self.scales.ndim != 1:
            raise ValueError("scales must be one-dimensional")
        if self.scales.shape[0] != self.basis.rank:
            raise ValueError("scales must match basis rank")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if (
            self.output_scaler is not None
            and self.output_scaler.scales.shape[0] != self.basis.rank
        ):
            raise ValueError("output_scaler must match basis rank")

    @property
    def rank(self) -> int:
        return self.basis.rank

    def _mean_at(self, t: float) -> np.ndarray:
        if self.mean_function is None:
            return np.zeros(self.rank)
        value = np.asarray(self.mean_function(t), dtype=float)
        if value.shape != (self.rank,):
            raise ValueError("mean_function must return shape (rank,)")
        return value

    def _as_tensor(self, t: np.ndarray) -> torch.Tensor:
        arr = np.asarray(t, dtype=float)
        # Apply time warping first (if present)
        if self.warp is not None:
            arr = self.warp.forward(arr)
        # Then apply input scaling to [0, 1] (if enabled)
        if self.input_scaler is not None:
            arr = self.input_scaler.forward(arr)
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device).reshape(-1, 1)

    def _as_intrinsic_tensor(self, tau: np.ndarray) -> torch.Tensor:
        """Return a tensor from intrinsic (already warped/scaled) inputs."""
        arr = np.asarray(tau, dtype=float)
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device).reshape(-1, 1)

    def _predict(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            mean = np.array([self._mean_at(float(tt)) for tt in times]).T
            var = (self.scales * self.prior_variance)[:, None] * np.ones(
                (self.rank, len(times))
            )
            return mean, var

        # This includes transforming the time variable if needed via the warp and input scaler (optional)
        t_tensor = self._as_tensor(times)

        means = []
        variances = []

        for model, likelihood in zip(self.models, self.likelihoods):
            model.eval()
            likelihood.eval()
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(self.jitter),
            ):
                # model(t_tensor) returns a gpytorch MultivariateNormal for the latent GP function
                # then likelihood(model(t_tensor)) returns the corresponding output distribution by including the observation noise
                dist = likelihood(model(t_tensor))
            means.append(dist.mean.detach().cpu().numpy())
            variances.append(dist.variance.detach().cpu().numpy())

        # We stack to have something of shape (n_basis, n_times)
        stacked_mean = np.stack(means, axis=0)
        stacked_var = np.stack(variances, axis=0)

        # If the prior mean function is nonzero, we need to add it back
        base = np.array([self._mean_at(float(tt)) for tt in times], dtype=float).T

        if self.output_scaler is not None:
            stacked_mean = self.output_scaler.unscale(stacked_mean)
            stacked_var = self.output_scaler.unscale_variance(stacked_var)

        return base + stacked_mean, stacked_var

    def mean(self, t: float) -> np.ndarray:
        mean, _ = self._predict(np.array([t], dtype=float))
        return mean.reshape(-1)

    def marginal_variance(self, t: float) -> np.ndarray:
        _, var = self._predict(np.array([t], dtype=float))
        return var.reshape(self.rank)

    def trace_uncertainty(self, t: float) -> float:
        return float(np.sum(self.marginal_variance(t)))

    def latent_marginal_variance(
        self,
        t: float | np.ndarray,
        *,
        intrinsic: bool = False,
    ) -> np.ndarray:
        """Return latent posterior variances (without observation noise).

        Accepts either a scalar ``t`` or an array of times. Scalars return a
        vector of shape ``(rank,)``. Arrays return shape ``(rank, n_times)``.
        """
        t_np = np.asarray(t, dtype=float)
        scalar_input = t_np.ndim == 0
        t_arr = t_np.reshape(-1)
        n_times = t_arr.size

        if not self.models:
            prior = (self.scales * self.prior_variance)[:, None]
            prior = prior * np.ones((self.rank, n_times), dtype=float)
            return prior.reshape(self.rank) if scalar_input else prior

        t_tensor = (
            self._as_intrinsic_tensor(t_arr) if intrinsic else self._as_tensor(t_arr)
        )

        variances: list[np.ndarray] = []
        for model in self.models:
            model.eval()
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(self.jitter),
            ):
                dist = model(t_tensor)
            variances.append(dist.variance.detach().cpu().numpy())

        stacked = np.stack(variances, axis=0)
        if self.output_scaler is not None:
            stacked = self.output_scaler.unscale_variance(stacked)
        return stacked.reshape(self.rank) if scalar_input else stacked

    def cross_covariance(
        self,
        times: np.ndarray,
        t_star: float | np.ndarray,
        *,
        intrinsic: bool = False,
    ) -> np.ndarray:
        """Return latent cross-covariances per basis.

        ``times`` has shape ``(n_times,)``. If ``t_star`` is scalar, the
        result has shape ``(rank, n_times)``. If ``t_star`` is an array with
        shape ``(n_star,)``, the result has shape ``(rank, n_times, n_star)``.
        """
        if not self.models:
            raise RuntimeError("cross_covariance requires a fitted posterior")

        times_arr = np.asarray(times, dtype=float).reshape(-1)
        t_star_np = np.asarray(t_star, dtype=float)
        star_scalar = t_star_np.ndim == 0
        t_star_arr = t_star_np.reshape(-1)

        n_times = times_arr.size
        n_star = t_star_arr.size
        joint = np.concatenate([times_arr, t_star_arr])
        joint_tensor = (
            self._as_intrinsic_tensor(joint) if intrinsic else self._as_tensor(joint)
        )

        covariances: list[np.ndarray] = []
        for model in self.models:
            model.eval()
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                gpytorch.settings.cholesky_jitter(self.jitter),
            ):
                dist = model(joint_tensor)
            cov = dist.covariance_matrix.detach().cpu().numpy()
            covariances.append(cov[:n_times, n_times : n_times + n_star])

        stacked = np.stack(covariances, axis=0)
        if self.output_scaler is not None:
            scales_sq = (self.output_scaler.scales**2).reshape(-1, 1, 1)
            stacked = stacked * scales_sq
        if star_scalar:
            return stacked[..., 0]
        return stacked


@dataclass
class GPyTorchHilbertRegressor:
    """Independent coefficient regression using GPyTorch backends."""

    basis: TangentBasis
    scales: np.ndarray
    kernel_spec: KernelSpec
    prior_variance: float = 1.0
    mean_function: Optional[MeanFunction] = None
    mean_module: gpytorch.means.Mean | None = None
    training_iter: int = 200
    lr: float = 0.05
    jitter: float = 1e-6
    use_cuda: bool = False
    dtype: torch.dtype = torch.float64
    input_scaling: bool = True
    noise_scale_init: float = 1.0
    noise_prior: gpytorch.priors.Prior | None = None
    noise_prior_from_data_scale: float | None = None
    noise_prior_from_data_mode: str = "variance"
    noise_prior_concentration: float = 2.0
    noise_prior_loo_cv: bool = False
    noise_prior_loo_jitter: float = 1e-4
    noise_constraint_lower: float | None = None
    noise_initializer: NoiseInitializer | None = None
    init_lengthscale_median: bool = False
    init_lengthscale_median_multiplier: float = 1.0
    init_outputscale_from_data: bool = False
    output_scaling: bool = False
    output_scaling_min_std: float = 1e-6

    def __post_init__(self) -> None:
        self.scales = np.asarray(self.scales, dtype=float)
        if self.scales.ndim != 1:
            raise ValueError("scales must be one-dimensional")
        if self.scales.shape[0] != self.basis.rank:
            raise ValueError("scales must match basis rank")
        if self.kernel_spec is None:
            raise ValueError("kernel_spec must be provided")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive")
        if self.output_scaling_min_std <= 0:
            raise ValueError("output_scaling_min_std must be positive")
        if self.noise_scale_init <= 0:
            raise ValueError("noise_scale_init must be positive")
        if self.noise_prior_concentration <= 0:
            raise ValueError("noise_prior_concentration must be positive")
        if self.noise_prior_loo_jitter <= 0:
            raise ValueError("noise_prior_loo_jitter must be positive")
        if self.noise_prior_from_data_scale is not None:
            if self.noise_prior_from_data_scale <= 0:
                raise ValueError("noise_prior_from_data_scale must be positive")
            if self.noise_prior is not None:
                raise ValueError(
                    "noise_prior_from_data_scale and noise_prior are mutually exclusive"
                )
            if self.noise_prior_from_data_mode not in {"variance", "std"}:
                raise ValueError(
                    "noise_prior_from_data_mode must be 'variance' or 'std'"
                )
        if self.noise_constraint_lower is not None and self.noise_constraint_lower <= 0:
            raise ValueError("noise_constraint_lower must be positive")
        if self.noise_initializer is not None and not callable(self.noise_initializer):
            raise ValueError("noise_initializer must be callable")
        if self.mean_module is not None and not isinstance(
            self.mean_module, gpytorch.means.Mean
        ):
            raise ValueError("mean_module must be a gpytorch.means.Mean instance")

    @property
    def rank(self) -> int:
        return self.basis.rank

    def _mean_at(self, t: float) -> np.ndarray:
        if self.mean_function is None:
            return np.zeros(self.rank)
        value = np.asarray(self.mean_function(t), dtype=float)
        if value.shape != (self.rank,):
            raise ValueError("mean_function must return shape (rank,)")
        return value

    def _clone_mean_module(self) -> gpytorch.means.Mean:
        if self.mean_module is None:
            return gpytorch.means.ZeroMean()
        return copy.deepcopy(self.mean_module)

    def _select_device(self) -> torch.device:
        if self.use_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def _as_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().cpu().reshape(-1)[0].item())
        return float(value)

    def _noise_prior_mean_from_residuals(self, residuals: np.ndarray) -> float:
        arr = np.asarray(residuals, dtype=float).reshape(-1)
        if arr.size == 0:
            base = 0.0
        elif arr.size == 1:
            base = float(arr[0] ** 2)
        else:
            base = float(np.var(arr, ddof=1))
        if self.noise_prior_from_data_mode == "std":
            base = float(np.sqrt(max(base, 0.0)))
        mean = float(self.noise_prior_from_data_scale) * max(base, 1.0e-6)
        return max(mean, 1.0e-6)

    def _noise_prior_mean_from_loo_cv(
        self,
        residuals: np.ndarray,
        train_times: np.ndarray,
        prior_variance: float,
    ) -> float:
        arr = np.asarray(residuals, dtype=float).reshape(-1)
        n = arr.size
        if n <= 2:
            return self._noise_prior_mean_from_residuals(arr)
        times_arr = np.asarray(train_times, dtype=float).reshape(-1)
        kernel = self.kernel_spec.build(
            outputscale=float(prior_variance),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        with torch.no_grad():
            tx = torch.as_tensor(times_arr, dtype=torch.float64).reshape(-1, 1)
            K = kernel(tx).evaluate().detach().numpy()
        K_jitter = K + float(self.noise_prior_loo_jitter) * np.eye(n)
        try:
            K_inv = np.linalg.inv(K_jitter)
        except np.linalg.LinAlgError:
            return self._noise_prior_mean_from_residuals(arr)
        diag_inv = np.diag(K_inv)
        if np.any(np.abs(diag_inv) < 1e-15):
            return self._noise_prior_mean_from_residuals(arr)
        alpha = K_inv @ arr
        loo_residuals = alpha / diag_inv
        base = float(np.var(loo_residuals, ddof=1))
        base = max(base, 1.0e-6)
        if self.noise_prior_from_data_mode == "std":
            base = float(np.sqrt(base))
        mean = float(self.noise_prior_from_data_scale) * base
        return max(mean, 1.0e-6)

    def _build_gamma_prior(
        self,
        mean: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> gpytorch.priors.Prior:
        concentration = torch.tensor(float(self.noise_prior_concentration), device=device, dtype=dtype)
        rate = concentration / torch.tensor(float(mean), device=device, dtype=dtype)
        return gpytorch.priors.GammaPrior(concentration=concentration, rate=rate)

    def _compute_mll_value(
        self,
        mll: gpytorch.mlls.ExactMarginalLogLikelihood,
        model: _ScalarTimeGP,
    ) -> float:
        try:
            with torch.no_grad():
                train_x = model.train_inputs[0]
                train_y = model.train_targets
                output = model(train_x)
                value = mll(output, train_y)
            return self._as_float(value)
        except Exception as exc:
            logger.debug("Could not compute marginal log likelihood: %s", exc)
            return float("nan")

    def _log_noise_stats(
        self,
        likelihood: gpytorch.likelihoods.Likelihood,
        model_idx: int | None = None,
    ) -> None:
        noise_tensor = None
        if hasattr(likelihood, "noise"):
            noise_tensor = likelihood.noise
        elif hasattr(likelihood, "noise_covar"):
            noise_covar = likelihood.noise_covar
            if hasattr(noise_covar, "base_noise"):
                noise_tensor = noise_covar.base_noise
            elif hasattr(noise_covar, "noise"):
                noise_tensor = noise_covar.noise
        if noise_tensor is None:
            return
        noise = noise_tensor.detach().cpu().numpy().reshape(-1)
        if noise.size == 0:
            return
        idx = f"[{model_idx}]" if model_idx is not None else ""
        logger.debug(
            "GP%s noise stats: min=%.6f mean=%.6f max=%.6f",
            idx,
            float(np.min(noise)),
            float(np.mean(noise)),
            float(np.max(noise)),
        )

    def _snapshot_hyperparams(
        self,
        model: _ScalarTimeGP,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> dict[str, Any]:
        kernel_snapshot = self.kernel_spec.snapshot(model.covar_module)
        noise_scale = None
        if isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood):
            noise_scale = self._as_float(likelihood.noise)
        return {
            "kernel": kernel_snapshot,
            "noise_scale": noise_scale,
        }

    def _log_hyperparams(
        self,
        stage: str,
        snapshot: dict[str, Any],
        model_idx: int | None = None,
    ) -> None:
        idx = f"[{model_idx}]" if model_idx is not None else ""
        kernel_params = snapshot.get("kernel", {})
        if kernel_params:
            items = [f"{key}={kernel_params[key]:.6f}" for key in sorted(kernel_params)]
            kernel_str = " ".join(items)
        else:
            kernel_str = "none"
        if snapshot["noise_scale"] is None:
            logger.debug("%s GP%s kernel params: %s", stage, idx, kernel_str)
            return
        logger.debug(
            "%s GP%s kernel params: %s noise_scale=%.6f",
            stage,
            idx,
            kernel_str,
            snapshot["noise_scale"],
        )

    def _train_model(
        self,
        model: _ScalarTimeGP,
        likelihood: gpytorch.likelihoods.Likelihood,
        model_idx: int | None = None,
    ) -> None:
        if self.training_iter <= 0:
            model.eval()
            likelihood.eval()
            return
        idx_label = f"[{model_idx}]" if model_idx is not None else ""
        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        if logger.isEnabledFor(logging.DEBUG):
            num_obs = int(model.train_inputs[0].shape[0])
            logger.debug(
                "Training GP%s with %d obs (iter=%d lr=%.4g jitter=%.2e)",
                idx_label,
                num_obs,
                int(self.training_iter),
                float(self.lr),
                float(self.jitter),
            )
        if logger.isEnabledFor(logging.DEBUG):
            self._log_noise_stats(likelihood, model_idx=model_idx)
            initial = self._snapshot_hyperparams(model, likelihood)
            self._log_hyperparams("Initial", initial, model_idx=model_idx)
            initial_mll = self._compute_mll_value(mll, model)
            if np.isfinite(initial_mll):
                logger.debug("Initial GP%s mll=%.6f", idx_label, initial_mll)
        scipy_kwargs = {"options": {"maxiter": int(self.training_iter)}}
        try:
            with gpytorch.settings.cholesky_jitter(self.jitter):
                fit_gpytorch_mll(
                    mll,
                    optimizer_kwargs=scipy_kwargs,
                    max_attempts=1,
                )
        except ModelFittingError as exc:
            logger.warning(
                "GP%s fit failed with scipy optimizer (%s). Retrying with torch Adam.",
                idx_label,
                exc,
            )
            torch_kwargs = {
                "step_limit": int(self.training_iter),
                "optimizer": partial(torch.optim.Adam, lr=float(self.lr)),
            }
            with gpytorch.settings.cholesky_jitter(self.jitter):
                fit_gpytorch_mll(
                    mll,
                    optimizer=fit_gpytorch_mll_torch,
                    optimizer_kwargs=torch_kwargs,
                    max_attempts=1,
                )
        model.eval()
        likelihood.eval()
        if logger.isEnabledFor(logging.DEBUG):
            final = self._snapshot_hyperparams(model, likelihood)
            self._log_hyperparams("Final", final, model_idx=model_idx)
            final_mll = self._compute_mll_value(mll, model)
            if np.isfinite(final_mll):
                logger.debug("Final GP%s mll=%.6f", idx_label, final_mll)

    def condition(
        self,
        observations: Sequence[TangentObservation],
        warp: Optional[Any] = None,
    ) -> GPyTorchHilbertPredictive:
        if not observations:
            device = self._select_device()
            return GPyTorchHilbertPredictive(
                basis=self.basis,
                kernel_spec=self.kernel_spec,
                scales=self.scales,
                prior_variance=self.prior_variance,
                mean_function=self.mean_function,
                models=[],
                likelihoods=[],
                device=device,
                dtype=self.dtype,
                jitter=self.jitter,
            )
        times = np.array([obs.time for obs in observations], dtype=float)
        coeffs = np.stack([obs.coefficients for obs in observations], axis=1)
        if coeffs.shape[0] != self.rank:
            raise ValueError("observation rank does not match GP basis rank")
        base_means = np.array([self._mean_at(t) for t in times], dtype=float).T
        residuals = coeffs - base_means
        output_scaler = None
        if self.output_scaling:
            output_scaler = OutputScaler.from_data(
                residuals,
                min_scale=self.output_scaling_min_std,
            )
            residuals = output_scaler.scale(residuals)
        device = self._select_device()

        # Handle warping: apply warp to get warped times
        if warp is not None:
            train_times = warp.forward(times)
        else:
            train_times = times

        # Apply input scaling to [0, 1] (after warping if present)
        input_scaler = None
        if self.input_scaling:
            input_scaler = InputScaler.from_data(train_times)
            train_times = input_scaler.forward(train_times)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Conditioning GP regressors: n_obs=%d rank=%d input_scaling=%s output_scaling=%s",
                len(times),
                self.rank,
                bool(self.input_scaling),
                bool(self.output_scaling),
            )
            logger.debug(
                "Training time range: [%.6f, %.6f] (warped=%s)",
                float(np.min(times)),
                float(np.max(times)),
                warp is not None,
            )
            if warp is not None and hasattr(warp, "total_length"):
                logger.debug("Warp total length: %.6f", float(warp.total_length))
            if input_scaler is not None:
                logger.debug(
                    "Input scaler: t_min=%.6f t_max=%.6f",
                    float(input_scaler.t_min),
                    float(input_scaler.t_max),
                )
            residual_stats = (
                float(np.min(residuals)),
                float(np.mean(residuals)),
                float(np.max(residuals)),
            )
            logger.debug(
                "Residual stats (all coeffs): min=%.6f mean=%.6f max=%.6f",
                *residual_stats,
            )

        train_x = torch.as_tensor(
            train_times, dtype=self.dtype, device=device
        ).unsqueeze(-1)
        parameter_overrides = None
        if self.init_lengthscale_median:
            parameter_overrides = self.kernel_spec.parameter_overrides_from_inputs(
                np.asarray(train_times, dtype=float)
            )
            if parameter_overrides is not None and self.init_lengthscale_median_multiplier != 1.0:
                for key in list(parameter_overrides.keys()):
                    if "lengthscale" in key:
                        val = parameter_overrides[key]
                        if isinstance(val, (int, float)):
                            parameter_overrides[key] = float(val) * float(self.init_lengthscale_median_multiplier)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Kernel parameter overrides: %s", parameter_overrides)

        scales = np.asarray(self.scales, dtype=float)
        if output_scaler is not None:
            scales = scales / (output_scaler.scales**2)
        if self.init_outputscale_from_data and coeffs.shape[1] > 1:
            obs_var = np.var(residuals, axis=1, ddof=1)
            scales = np.maximum(obs_var, 1e-3)
        models: List[_ScalarTimeGP] = []
        likelihoods: List[gpytorch.likelihoods.Likelihood] = []
        for k in range(self.rank):
            target = torch.as_tensor(residuals[k], dtype=self.dtype, device=device)
            noise_prior = self.noise_prior
            noise_prior_mean = None
            noise_constraint = None
            if self.noise_prior_from_data_scale is not None:
                if self.noise_prior_loo_cv:
                    noise_prior_mean = self._noise_prior_mean_from_loo_cv(
                        residuals[k], times, self.prior_variance
                    )
                else:
                    noise_prior_mean = self._noise_prior_mean_from_residuals(residuals[k])
                noise_prior = self._build_gamma_prior(
                    noise_prior_mean,
                    device=device,
                    dtype=self.dtype,
                )
                lower = min(1.0e-4, noise_prior_mean * 0.1)
                if self.noise_constraint_lower is not None:
                    lower = max(lower, float(self.noise_constraint_lower))
                lower = max(lower, 1.0e-12)
                noise_constraint = gpytorch.constraints.GreaterThan(lower)
            elif self.noise_constraint_lower is not None:
                noise_constraint = gpytorch.constraints.GreaterThan(
                    float(self.noise_constraint_lower)
                )
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=noise_constraint,
            ).to(device=device, dtype=self.dtype)
            if noise_prior_mean is not None and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "GP[%d] noise prior mean=%.6f (mode=%s, scale=%.3f)",
                    k,
                    float(noise_prior_mean),
                    self.noise_prior_from_data_mode,
                    float(self.noise_prior_from_data_scale),
                )
                if noise_constraint is not None:
                    logger.info(
                        "GP[%d] noise constraint lower=%.6e",
                        k,
                        float(noise_constraint.lower_bound),
                    )
            if self.noise_initializer is not None:
                noise_init = float(self.noise_initializer(residuals[k], k))
                if noise_init <= 0:
                    raise ValueError("noise_initializer must return a positive value")
            elif noise_prior_mean is not None:
                noise_init = float(noise_prior_mean)
            else:
                noise_init = float(self.noise_scale_init)
            min_noise = None
            if noise_constraint is not None:
                min_noise = float(noise_constraint.lower_bound)
            elif hasattr(likelihood, "noise_covar") and hasattr(
                likelihood.noise_covar, "raw_noise_constraint"
            ):
                lower = likelihood.noise_covar.raw_noise_constraint.lower_bound
                if torch.is_tensor(lower):
                    min_noise = float(lower.detach().cpu().reshape(-1)[0].item())
                else:
                    min_noise = float(lower)
            if min_noise is not None and noise_init < min_noise:
                if logger.isEnabledFor(logging.INFO):
                    logger.info(
                        "GP[%d] noise init %.6f below constraint %.6f; clamping",
                        k,
                        noise_init,
                        min_noise,
                    )
                noise_init = min_noise
            likelihood.noise = torch.tensor(
                noise_init,
                dtype=self.dtype,
                device=device,
            )
            kernel = self.kernel_spec.build(
                outputscale=float(scales[k]),
                parameter_overrides=parameter_overrides,
                device=device,
                dtype=self.dtype,
            )
            mean_module = self._clone_mean_module()
            model = _ScalarTimeGP(
                train_x=train_x,
                train_y=target,
                likelihood=likelihood,
                kernel=kernel,
                mean_module=mean_module,
            ).to(device=device, dtype=self.dtype)
            models.append(model)
            likelihoods.append(likelihood)
        for idx, (model, likelihood) in enumerate(zip(models, likelihoods)):
            self._train_model(model, likelihood, model_idx=idx)
        return GPyTorchHilbertPredictive(
            basis=self.basis,
            kernel_spec=self.kernel_spec,
            scales=scales,
            prior_variance=self.prior_variance,
            mean_function=self.mean_function,
            models=models,
            likelihoods=likelihoods,
            device=device,
            dtype=self.dtype,
            jitter=self.jitter,
            warp=warp,
            input_scaler=input_scaler,
            output_scaler=output_scaler,
        )

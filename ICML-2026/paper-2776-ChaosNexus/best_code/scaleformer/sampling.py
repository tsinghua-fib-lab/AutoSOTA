import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from dysts.base import BaseDyn
from dysts.sampling import BaseSampler
from dysts.systems import _resolve_event_signature
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SignedGaussianParamSampler(BaseSampler):
    """Sample gaussian perturbations for system parameters with sign control

    Args:
        scale: std (isotropic) of gaussian used for sampling
        sign_match_probability: probability of matching perturbation sign with parameter sign
            defaults to 0.0 (never match signs, matching GaussianParamSampler behavior)
    """

    scale: float = 1e-2
    eps: float = 1e-6
    sign_match_probability: float = 0.0
    ignore_probability: float = 0.0
    verbose: bool = False

    def __call__(
        self, name: str, param: NDArray, system: BaseDyn | None = None
    ) -> NDArray | float:
        # exponentially reduce ignore probability for low parameter count
        if system:
            param_count = len(system.param_list)
            effective_ignore_prob = self.ignore_probability * (1 - np.exp(-param_count))
        else:
            effective_ignore_prob = self.ignore_probability

        if self.rng.random() < effective_ignore_prob:
            return param

        # scale each parameter relatively
        shape = 1 if np.isscalar(param) else param.shape

        # avoid shape errors
        flat_param = np.array(param, dtype=np.float32).flatten()
        scale = np.abs(flat_param) * self.scale
        cov = np.diag(np.square(scale) + self.eps)
        perturbation = self.rng.multivariate_normal(
            mean=np.zeros_like(flat_param), cov=cov
        ).reshape(shape)

        if np.isscalar(param):
            perturbation = perturbation.item()

        # match sign with specified probability
        if self.rng.random() < self.sign_match_probability:
            perturbation = np.sign(param) * np.abs(perturbation)

        param = np.array(param)
        perturbed_param = param + perturbation

        if self.verbose:
            if system is not None:
                logger.info(
                    f"System: {system.name} | param {name}: {param} -> {perturbed_param}"
                )
            else:
                logger.info(f"Parameter {name}: {param} -> {perturbed_param}")
        return perturbed_param


@dataclass
class OnAttractorInitCondSampler(BaseSampler):
    """
    Sample points from the attractor of a system

    WARNING: This is multiprocessing-unsafe, use a Manager to share the cache across processes.

    Args:
        reference_traj_length: Length of the reference trajectory to use for sampling ic on attractor.
        reference_traj_transient: Transient length to ignore for the reference trajectory
        trajectory_cache: Cache of reference trajectories for each system.
        events: integration events to pass to solve_ivp
    """

    # TODO: these arguments are not used
    reference_traj_length: int = 4096
    reference_traj_transient: float = 0.2
    reference_traj_n_periods: int = 40
    reference_traj_atol: float = 1e-7
    reference_traj_rtol: float = 1e-6
    trajectory_cache: dict[str, NDArray | None] = field(default_factory=dict)
    silence_integration_errors: bool = False
    recompute_standardization: bool = False
    events: list[Callable] | None = None
    verbose: int = 0

    def __post_init__(self):
        super().__post_init__()
        assert 0 < self.reference_traj_transient < 1, (
            "Transient must be a fraction of the trajectory length"
        )
        self.transient = int(self.reference_traj_length * self.reference_traj_transient)

    def clear_cache(self):
        """Clear the trajectory cache. Creates new dict if previous was a managed dict."""
        self.trajectory_cache.clear()

    def __call__(self, ic: NDArray, system: BaseDyn) -> NDArray | None:
        if system.name is None:
            raise ValueError("System must have a name")

        # make reference trajectory if not already cached
        cache_traj = self.trajectory_cache.get(system.name)
        if cache_traj is None:
            # resolve event signatures and reset events attributes if applicable
            events = [
                _resolve_event_signature(system, event) for event in self.events or []
            ]
            for event in events:
                if hasattr(event, "reset") and callable(event.reset):
                    event.reset()

            try:
                pts_per_period = (
                    self.reference_traj_length // self.reference_traj_n_periods
                )
                reference_traj = system.make_trajectory(
                    self.reference_traj_length,
                    pts_per_period=pts_per_period,
                    events=events,
                    standardize=False,
                    atol=self.reference_traj_atol,
                    rtol=self.reference_traj_rtol,
                )
            except Exception as e:
                if self.verbose > 0:
                    logger.error(f"Error integrating {system.name}: {str(e)}")
                if not self.silence_integration_errors:
                    raise e
                return None

            # if integrate fails, resulting in an incomplete trajectory
            if reference_traj is None:
                if self.verbose > 0:
                    logger.warning(
                        f"On-attractor sampling failed integration for {system.name}"
                    )
                return None

            # renormalize with respect to reference trajectory
            # this should work since system is passed by reference
            if self.recompute_standardization:
                system.mean = reference_traj.mean(axis=0)
                system.std = reference_traj.std(axis=0)

            reference_traj = reference_traj[self.transient :]
            self.trajectory_cache[system.name] = reference_traj.copy()
            cache_traj = reference_traj

        # Sample a new initial condition from the cached trajectory
        new_ic = self.rng.choice(cache_traj)

        if self.verbose > 1:
            logger.info(f"System: {system.name} ic: {ic} -> {new_ic}")

        return new_ic


@dataclass
class GaussianInitialConditionSampler(BaseSampler):
    """
    Sample gaussian perturbations for each initial condition in a given system list
    """

    scale: float = 1e-4
    verbose: bool = False  # for testing purposes

    def __call__(self, ic: NDArray, system: BaseDyn) -> NDArray:
        """
        Sample a new initial condition from a multivariate isotropic Gaussian.

        Args:
            ic (NDArray): The current initial condition.

        Returns:
            NDArray: A resampled version of the initial condition.
        """
        # Scale the covariance relative to each dimension
        scaled_cov = np.diag(np.square(ic * self.scale))
        perturbed_ic = self.rng.multivariate_normal(mean=ic, cov=scaled_cov)

        if self.verbose:
            if system is not None:
                logger.info(f"System: {system.name} \nIC: {ic} -> {perturbed_ic}")
            else:
                logger.info(f"IC: {ic} -> {perturbed_ic}")

        return perturbed_ic

"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import logging
from typing import Callable

import numpy as np
from dysts.base import DynSys
from scaleformer.coupling_maps import RandomAdditiveCouplingMap

logger = logging.getLogger(__name__)


class SkewProduct(DynSys):
    def __init__(
        self,
        driver: DynSys,
        response: DynSys,
        coupling_map: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        _default_random_seed: int | None = None,
        **kwargs,
    ):
        """A skew-product dynamical system composed of a driver and response system.

        The driver system evolves independently while the response system is influenced by the driver
        through a coupling map. Inherits from dysts.base.DynSys to enable trajectory generation.

        Args:
            driver (DynSys): The autonomous driver dynamical system
            response (DynSys): The response dynamical system that is influenced by the driver
            coupling_map (Callable, optional):
                Function defining how driver and response systems are coupled.
                If None, defaults to RandomAdditiveCouplingMap.
            _default_random_seed (int, optional):
                Random seed for default coupling map initialization
        """
        # default to (randomly) additively forcing the response with the driver
        if coupling_map is None:
            self.coupling_map = RandomAdditiveCouplingMap(
                driver.dimension, response.dimension, random_seed=_default_random_seed
            )
        else:
            self.coupling_map = coupling_map

        super().__init__(
            metadata_path=None,
            parameters={},  # dummy: parameters are handled in the overwritten methods below
            dt=min(driver.dt, response.dt),
            period=max(driver.period, response.period),
            metadata={
                "name": f"{driver.name}_{response.name}",
                "dimension": driver.dimension + response.dimension,
                "driver": driver,
                "response": response,
                "driver_dim": driver.dimension,
                "response_dim": response.dimension,
            },
            **kwargs,
        )

        # hack: set a dummy param list for param count checks
        n_params = len(driver.parameters) + len(response.parameters)
        if hasattr(self.coupling_map, "n_params"):
            n_params += self.coupling_map.n_params
        self.param_list = [0 for _ in range(n_params)]

        assert hasattr(driver, "ic") and hasattr(response, "ic"), (
            "Driver and response must have default initial conditions"
            "and must be of the same dimension"
        )

        # manually set the unbounded indices derived from the coupling map
        self.unbounded_indices = self._update_unbounded_indices()

        self.ic = np.concatenate([self.driver.ic, self.response.ic])
        self.mean = np.concatenate([self.driver.mean, self.response.mean])
        self.std = np.concatenate([self.driver.std, self.response.std])

    def _update_unbounded_indices(self) -> list[int]:
        """Update the unbounded indices (which may depend on the coupling map)"""
        driver_inds = self.driver.unbounded_indices
        if hasattr(self.coupling_map, "unbounded_indices") and callable(
            self.coupling_map.unbounded_indices
        ):
            coupled_inds = self.coupling_map.unbounded_indices(
                driver_inds, self.response.unbounded_indices
            )
            return driver_inds + [i + self.driver_dim for i in coupled_inds]
        return driver_inds

    def transform_params(self, param_transform: Callable) -> bool:
        """Transform parameters of the driver & response systems and coupling map"""
        driver_success = self.driver.transform_params(param_transform)
        response_success = self.response.transform_params(param_transform)
        success = driver_success and response_success

        if hasattr(self.coupling_map, "transform_params"):
            coupling_success = self.coupling_map.transform_params(param_transform)

            if coupling_success:  # update upon successful coupling map transform
                self.unbounded_indices = self._update_unbounded_indices()

            success &= coupling_success

        return success

    def has_jacobian(self) -> bool:
        return self.driver.has_jacobian() and self.response.has_jacobian()

    def rhs(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Flow of the skew product system

        NOTE: the coupled RHS is always assumed to have the dimension of the response system
        """
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_rhs = np.asarray(self.driver.rhs(driver, t))
        response_rhs = np.asarray(self.response.rhs(response, t))
        coupled_rhs = self.coupling_map(driver_rhs, response_rhs)
        return np.concatenate([driver_rhs, coupled_rhs])

    def jac(self, X: np.ndarray, t: float) -> np.ndarray:
        """
        Jacobian of the skew product system, computed via chain rule

        NOTE: the coupling map jacobian implements the jacobian wrt to the driver or the response flow
        """
        driver, response = X[: self.driver_dim], X[self.driver_dim :]

        driver_jac = np.asarray(self.driver.jac(driver, t))
        coupling_jac_driver = self.coupling_map.jac(driver, response, wrt="driver")

        response_jac = np.asarray(self.response.jac(response, t))
        coupling_jac_response = self.coupling_map.jac(driver, response, wrt="response")

        return np.block(
            [
                [driver_jac, np.zeros((self.driver_dim, self.response_dim))],
                [
                    coupling_jac_driver @ driver_jac,
                    coupling_jac_response @ response_jac,
                ],
            ]
        )

    def __call__(self, X: np.ndarray, t: float) -> np.ndarray:
        return self.rhs(X, t)

    def _postprocessing(self, *X: np.ndarray) -> np.ndarray:
        driver, response = X[: self.driver_dim], X[self.driver_dim :]
        driver_postprocess_fn = (
            None
            if not hasattr(self.driver, "_postprocessing")
            else self.driver._postprocessing
        )
        response_postprocess_fn = (
            None
            if not hasattr(self.response, "_postprocessing")
            else self.response._postprocessing
        )

        if driver_postprocess_fn is not None:
            driver = driver_postprocess_fn(*driver)
        if hasattr(self.coupling_map, "_postprocessing"):
            response = self.coupling_map._postprocessing(
                np.asarray(response),
                driver_postprocess_fn,
                response_postprocess_fn,
                self.response.unbounded_indices,
                self.driver.unbounded_indices,
            )

        return np.concatenate([np.asarray(driver), np.asarray(response)])

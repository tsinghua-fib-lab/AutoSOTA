# Copyright 2023 Multiscale Modeling of Fluid Materials, TU Munich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Documents commonly used types in chemtrain."""

import typing
from typing import Callable, Any, Optional, Protocol, TypedDict, Dict, TypeAlias
from typing_extensions import NotRequired

try:
    from jax.typing import ArrayLike
except:
    ArrayLike = Any

import jax_md_mod
from jax_md.energy import NeighborList

# Energy Functions

class EnergyFn(Protocol):
    def __call__(self,
                 position: ArrayLike,
                 neighbor: NeighborList=None,
                 **kwargs) -> ArrayLike:
        """Computes the energy for a given conformation.

        Args:
            position: Positions of the particles
            neighbor: Updated neighborlist
            **kwargs: Additional parameters to the energy function, e.g., the
                thermostat temperature ``"kbt"``.

        Returns:
            Returns the potential energy of the system.

        """

class EnergyFnTemplate(Protocol):
    def __call__(self, energy_params: Any) -> EnergyFn:
        """Initialies the energy function with parameters.

        Args:
            energy_params: Parameters for the energy function.

        Returns:
            Returns a concrete potential energy function.

        """



class ErrorFn(Protocol):
    def __call__(self,
                 predictions: ArrayLike,
                 targets: ArrayLike,
                 mask: ArrayLike = None,
                 weights: ArrayLike = None) -> ArrayLike:
        """Computes the error of the predictions.

        Args:
            predictions: Predicted values with same shape as targets
            targets: Target values
            mask: Masks out invalid predictions along the first axis.
            weights: Weights for the error calculation.

        Returns:
            Returns the masked error value.

        """

# Quantities

class TrajFn(Protocol):
    def __call__(self, quantity_trajs: Dict[str, ArrayLike], weights: ArrayLike = None) -> ArrayLike: ...


class SingleTarget(TypedDict):
    traj_fn: TrajFn
    loss_fn: NotRequired[Callable[[ArrayLike, ArrayLike], ArrayLike]]
    target: NotRequired[ArrayLike]
    gamma: NotRequired[ArrayLike]


class QuantityComputeFunction(Protocol):
    def __call__(self, state: Any, **kwargs) -> ArrayLike: ...


QuantityDict: TypeAlias = Dict[str, QuantityComputeFunction]

TargetDict: TypeAlias = Dict[str, SingleTarget]


class ComputeFn(Protocol):

    @typing.overload
    def __call__(self, state, neighbor: NeighborList = None, **kwargs) -> Any: ...
    def __call__(self, state, **kwargs) -> Any: ...

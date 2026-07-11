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

"""Pre-defined simulator templates."""

import jax_md_mod
from jax_md import simulate
from jax_md.partition import NeighborList

from chemtrain.ensemble import sampling

from typing_extensions import TypedDict, Optional
from jax.typing import ArrayLike


class StatePoint(TypedDict, total=False):
    """Dictionary defining properties of the thermodynamic statepoint.

    Args:
        kT: External temperature
        pressure: External pressure
        dof: Degrees of freedom of the system. Reduced, e.g., for constrained
            bonds.
        box: Simulation box
        volume: Volume of the simulation box
        species: Species of the particles
        mask: Mask for the particles

    """

    kT: Optional[ArrayLike]
    pressure: Optional[ArrayLike]
    dof: Optional[ArrayLike]
    species: Optional[ArrayLike]
    mask: Optional[ArrayLike]
    box: Optional[ArrayLike]
    volume: Optional[ArrayLike]


def init_nvt_langevin_simulator_template(shift_fn,
                                         nbrs: NeighborList = None,
                                         dt: float = 0.1,
                                         kT: float = 2.56,
                                         gamma: float = 100.):
    """Initializes a NVT Langevin simulator template."""
    extra_kwargs = dict(dt=dt, kT=kT, gamma=gamma)
    return sampling.initialize_simulator_template(
        init_simulator_fn=simulate.nvt_langevin,
        shift_fn=shift_fn, nbrs=nbrs, init_with_PRNGKey=True,
        extra_simulator_kwargs=extra_kwargs
    )

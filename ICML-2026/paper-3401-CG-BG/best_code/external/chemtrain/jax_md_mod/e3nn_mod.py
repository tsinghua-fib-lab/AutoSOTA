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

"""Patches the e3nn_jax library."""

import jax
from jax import lax, numpy as jnp


def _distinct_but_small(x: jax.Array):
    """Maps the entries of x into integers from 0 to n-1 denoting unique values.

    Note:
        This implementation replaces the original e3nn_jax implementation
        and allows to use Shape Polymorphism for exporting.

    """

    shape = x.shape
    x = x.ravel()
    # We sort the array
    sorted_idx = jnp.argsort(x)

    # We assign indices to the sorted array
    new_group = jnp.concat([jnp.zeros(1), jnp.diff(x[sorted_idx]) > 0], axis=0)
    group_idx = jnp.cumsum(new_group)

    # We assign the unique indices
    x = x.at[sorted_idx].set(group_idx)
    return x.reshape(shape)

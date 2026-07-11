# MIT License
#
# Copyright (c) 2025 tummfm
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Patches the e3nn_jax library."""

import jax
from jax import lax, numpy as jnp


def _distinct_but_small(x: jax.Array):
    """Maps the entries of x into integers from 0 to n-1 denoting unique values.

    Note:
        This implementation replaces the original e3nn_jax implementation
        and allows to use Shape Polymorphism for exporting.

    """
    print(f"Use a custom scatter implementation")

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

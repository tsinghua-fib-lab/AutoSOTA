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

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class MonotonicInterpolate:
    """
    Piecewise cubic, monotonic interpolation via Steffens method [#Steffen]_.

    The interpolation curve is monotonic within each interval such that extrema
    can only occur at grid points. Guarantees continuous first derivatives
    of the spline. Is applicable to arbitrary data; not restricted to monotonic data.

    References:

       .. [#Steffen] Steffen, M.,
          “A simple method for monotonic interpolation in one dimension.”,
          Astronomy and Astrophysics, vol. 239, pp. 443–450, 1990.

    Attributes:
        a, b, c, d: Piecewise coefficients for the cubic sections
        x: grid points

    Args:
        x : x-value of grid points -- must be strictly increasing
        y : y-value of grid points
        coefficients: Necessary for tree_unflatten

    Returns:
        A function that takes x values and returns spline values at these points

    """
    def __init__(self, x, y, coefficients=None):

        assert len(x) > 3, "Not enough input values for spline"
        assert len(x) == len(y), "x and y must have the same length"
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."

        if coefficients is None:

            h = jnp.diff(x)
            k = jnp.diff(y)
            s = k/h
            p = (s[0:-1] * h[1:] + s[1:] * h[0:-1]) / (h[0:-1] + h[1:])

            # Build coefficient pairs
            s0s1 = s[0:-1] * s[1:]
            a = jnp.sign(s[0:-1])
            cond1 = jnp.logical_or(jnp.abs(p) > 2 * jnp.abs(s[0:-1]), jnp.abs(p) > 2 * jnp.abs(s[1:]))
            tmp = jnp.where(cond1, 2 * a * jnp.where(jnp.abs(s[0:1]) > jnp.abs(s[1:]), jnp.abs(s[1:]),
                                                     jnp.abs(s[0:-1])), p)
            slopes = jnp.where(s0s1 <= 0, 0.0, tmp)

            p0 = s[0]*(1+h[0]/(h[0]+h[1]))-s[1]*(h[0]/(h[0]+h[1]))
            pn = s[-1]*(1+h[-1]/(h[-1]+h[-2])) - s[-2]*(h[-1]/(h[-1]+h[-2]))

            tmp0 = jnp.where(jnp.abs(p0) > 2 * jnp.abs(s[0]), 2 * s[0], p0)
            tmpn = jnp.where(jnp.abs(pn) > 2 * jnp.abs(s[-1]), 2 * s[-1], pn)

            yp0 = jnp.where(p0 * s[0] <= 0.0, 0.0, tmp0)
            ypn = jnp.where(pn * s[-1] <= 0.0, 0.0, tmpn)
            slopes = jnp.concatenate((jnp.array([yp0]), slopes, jnp.array([ypn])))

            # Build the coefficients and store properties
            a = (slopes[0:-1] + slopes[1:] - 2 * s) / jnp.square(h)
            b = (3 * s - 2 * slopes[0:-1] - slopes[1:]) / h
            c = slopes
            d = y[0:-1]

            coefficients = (a, b, c, d)

        self.x = x
        self.y = y
        self.coefficients = coefficients

    def __call__(self, x_new):
        """
        Evaluate spline at new data points.

        Args:
            x_new: Evaluation points

        Returns:
            Returns the interpolated values y_new corresponding to y_new.
        """

        a, b, c, d = self.coefficients

        x_new_idx = jnp.searchsorted(self.x, x_new, side="right") - 1  # Find the indexes of the reference

        # avoid out of bound indexing
        x_new_idx = jnp.where(x_new_idx < 0, 0, x_new_idx)
        x_new_idx = jnp.where(x_new_idx > len(self.x) - 2, len(self.x) - 2, x_new_idx)

        # Return the interpolated values
        a = a[x_new_idx]
        b = b[x_new_idx]
        c = c[x_new_idx]
        d = d[x_new_idx]

        x = self.x[x_new_idx]
        y_new = a * jnp.power(x_new - x, 3) + b * jnp.power(x_new - x, 2) + c * (x_new - x) + d

        return y_new

    def tree_flatten(self):
        children = (self.x, self.y, self.coefficients)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        x, y, coefficients = children
        return cls(x, y, coefficients=coefficients)

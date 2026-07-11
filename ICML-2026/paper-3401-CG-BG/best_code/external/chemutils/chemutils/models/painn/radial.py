# MIT License
#
# Copyright (c) 2023 Gianluca Galletti
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


from typing import Callable

import haiku as hk
import jax.numpy as jnp


def gaussian_rbf(
    n_rbf: int,
    cutoff: float,
    start: float = 0.0,
    centered: bool = False,
    learnable: bool = False,
) -> Callable[[jnp.ndarray], Callable]:
    r"""Gaussian radial basis functions.

    Args:
        n_rbf: total number of Gaussian functions, :math:`N_g`.
        cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
        start: center of first Gaussian function, :math:`\mu_0`.
        learnable: If True, widths and offset of Gaussian functions learnable.
    """
    if centered:
        width = jnp.linspace(start, cutoff, n_rbf)
        offset = jnp.zeros_like(width)
    else:
        offset = jnp.linspace(start, cutoff, n_rbf)
        width = jnp.abs(cutoff - start) / n_rbf * jnp.ones_like(offset)

    if learnable:
        widths = hk.get_parameter(
            "widths", width.shape, width.dtype, init=lambda *_: width
        )
        offsets = hk.get_parameter(
            "offset", offset.shape, offset.dtype, init=lambda *_: offset
        )
    else:
        hk.set_state("widths", jnp.array([width]))
        hk.set_state("offsets", jnp.array([offset]))
        widths = hk.get_state("widths")
        offsets = hk.get_state("offsets")

    def _rbf(x: jnp.ndarray) -> jnp.ndarray:
        coeff = -0.5 / jnp.power(widths, 2)
        diff = x[..., jnp.newaxis] - offsets
        return jnp.exp(coeff * jnp.power(diff, 2))

    return _rbf


def bessel_rbf(n_rbf: int, cutoff: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Sine for radial basis functions with coulomb decay (0th order bessel).

    Args:
        n_rbf: total number of Bessel functions, :math:`N_g`.
        cutoff: center of last Bessel function, :math:`\mu_{N_g}`

    References:
        [#dimenet] Klicpera, Groß, Günnemann:
        Directional message passing for molecular graphs.
        ICLR 2020
    """
    # compute offset and width of Gaussian functions
    freqs = jnp.arange(1, n_rbf + 1) * jnp.pi / cutoff
    hk.set_state("freqs", freqs)
    freqs = hk.get_state("freqs")

    def _rbf(self, x: jnp.ndarray) -> jnp.ndarray:
        ax = x[..., None] * self.freqs
        sinax = jnp.sin(ax)
        norm = jnp.where(x == 0, 1.0, x)
        y = sinax / norm[..., None]
        return y

    return _rbf


EXISTING_RBF = {
    "gaussian": gaussian_rbf,
    "bessel": bessel_rbf,
}

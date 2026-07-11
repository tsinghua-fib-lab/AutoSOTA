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


def cosine_cutoff(cutoff: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Behler-style cosine cutoff.

    .. math::
        f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
            & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff (float): cutoff radius.
    """
    hk.set_state("cutoff", cutoff)
    cutoff = hk.get_state("cutoff")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        cuts = 0.5 * (jnp.cos(x * jnp.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        mask = jnp.array(x < cutoff, dtype=jnp.float32)
        return cuts * mask

    return _cutoff


def mollifier_cutoff(cutoff: float, eps: float) -> Callable[[jnp.ndarray], Callable]:
    r"""Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.
    """
    hk.set_state("cutoff", jnp.array([cutoff]))
    hk.set_state("eps", jnp.array([eps]))
    cutoff = hk.get_state("cutoff")
    eps = hk.get_state("eps")

    def _cutoff(x: jnp.ndarray) -> jnp.ndarray:
        # Compute values of cutoff function
        mask = (x + eps < cutoff).float()
        cuts = jnp.exp(1.0 - 1.0 / (1.0 - jnp.power(x * mask / cutoff, 2)))
        return cuts * mask

    return _cutoff

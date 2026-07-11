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

from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph


class LinearXav(hk.Linear):
    """Linear layer with Xavier init. Avoid distracting 'w_init' everywhere."""

    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        if w_init is None:
            w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform")
        super().__init__(output_size, with_bias, w_init, b_init, name)


class GatedEquivariantBlock(hk.Module):
    """Gated equivariant block (restricted to vectorial features). FIG 3 in the paper.

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021
    """

    def __init__(
        self,
        hidden_size: int,
        scalar_out_channels: int,
        vector_out_channels: int,
        activation: Callable = jax.nn.silu,
        scalar_activation: Callable = None,
        eps: float = 1e-8,
        name: str = "gated_equivariant_block",
    ):
        super().__init__(name)

        assert scalar_out_channels > 0 and vector_out_channels > 0
        self._scalar_out_channels = scalar_out_channels
        self._vector_out_channels = vector_out_channels
        self._eps = eps

        self.vector_mix_net = LinearXav(
            2 * vector_out_channels,
            with_bias=False,
            name="vector_mix_net",
        )
        self.gate_block = hk.Sequential(
            [
                LinearXav(hidden_size),
                activation,
                LinearXav(scalar_out_channels + vector_out_channels),
            ],
            name="scalar_gate_net",
        )
        self.scalar_activation = scalar_activation

    def __call__(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        v_l, v_r = jnp.split(self.vector_mix_net(v), 2, axis=-1)

        v_r_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2) + self._eps)
        gating_scalars = jnp.concatenate([s, v_r_norm], axis=-1)
        s, _, v_gate = jnp.split(
            self.gate_block(gating_scalars),
            [self._scalar_out_channels, self._vector_out_channels],
            axis=-1,
        )
        # scale the vectors by the gating scalars
        v = v_l * v_gate[:, jnp.newaxis]

        if self.scalar_activation:
            s = self.scalar_activation(s)

        return s, v


def pooling(
    graph: jraph.GraphsTuple,
    aggregate_fn: Callable = jraph.segment_sum,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pools over graph nodes with the specified aggregation.

    Args:
        graph: Input graph
        aggregate_fn: function used to update pool over the nodes

    Returns:
        The pooled graph nodes.
    """
    n_graphs = graph.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs)
    # Equivalent to jnp.sum(n_node), but jittable
    sum_n_node = tree.tree_leaves(graph.nodes)[0].shape[0]
    batch = jnp.repeat(graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node)

    s, v = None, None
    if graph.nodes.s is not None:
        s = aggregate_fn(graph.nodes.s, batch, n_graphs)
    if graph.nodes.v is not None:
        v = aggregate_fn(graph.nodes.v, batch, n_graphs)

    return s, v

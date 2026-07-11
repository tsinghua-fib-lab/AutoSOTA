# MIT License
#
# Copyright (c) 2023 Gianluca Galletti
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


"""PaiNN Model

Implementation adapted from https://github.com/gerkone/painn-jax (MIT-License).

"""

from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Callable, Any, Tuple, Optional
from typing import NamedTuple

import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
import numpy as onp
from chemutils.models.layers import AtomicEnergyLayer
from jax import numpy as jnp
from jax_md import space, partition, nn
from jax_md import util as md_util
from jax_md_mod.model import layers, sparse_graph

from .blocks import GatedEquivariantBlock, LinearXav, pooling

class NodeFeatures(NamedTuple):
    """Simple container for scalar and vectorial node features."""

    s: jnp.ndarray = None
    v: jnp.ndarray = None


ReadoutFn = Callable[[jraph.GraphsTuple], Tuple[jnp.ndarray, jnp.ndarray]]
ReadoutBuilderFn = Callable[..., ReadoutFn]


def PaiNNReadout(
    hidden_size: int,
    task: str,
    pool: str,
    out_channels: int = 1,
    activation: Callable = jax.nn.silu,
    blocks: int = 2,
    eps: float = 1e-8,
) -> ReadoutFn:
    """
    PaiNN readout block.

    Args:
        hidden_size: Number of hidden channels.
        task: Task to perform. Either "node" or "graph".
        pool: pool method. Either "sum" or "avg".
        scalar_out_channels: Number of scalar/vector output channels.
        activation: Activation function.
        blocks: Number of readout blocks.
    """

    assert task in ["node", "graph"], "task must be node or graph"
    assert pool in ["sum", "avg"], "pool must be sum or avg"
    if pool == "avg":
        pool_fn = jraph.segment_mean
    if pool == "sum":
        pool_fn = jraph.segment_sum

    def _readout(graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        s, v = graph.nodes
        s = jnp.squeeze(s, axis=1)
        for i in range(blocks - 1):
            ith_hidden_size = hidden_size // 2 ** (i + 1)
            s, v = GatedEquivariantBlock(
                hidden_size=ith_hidden_size * 2,
                scalar_out_channels=ith_hidden_size,
                vector_out_channels=ith_hidden_size,
                activation=activation,
                eps=eps,
                name=f"readout_block_{i}",
            )(s, v)

        if task == "graph":
            graph = graph._replace(nodes=NodeFeatures(s, v))
            s, v = pooling(graph, aggregate_fn=pool_fn)

        s, v = GatedEquivariantBlock(
            hidden_size=ith_hidden_size,
            scalar_out_channels=out_channels,
            vector_out_channels=out_channels,
            activation=activation,
            eps=eps,
            name="readout_block_out",
        )(s, v)
        print(f"Readout output: {s.shape}, {v.shape}")
        return jnp.squeeze(s, axis=1), jnp.squeeze(v, axis=2)

    return _readout


class PaiNNLayer(hk.Module):
    """PaiNN interaction block."""

    def __init__(
        self,
        hidden_size: int,
        layer_num: int,
        *,
        epsilon: float,
        activation: Callable = jax.nn.silu,
        blocks: int = 2,
        aggregate_fn: Callable = jraph.segment_sum,
        eps: float = 1e-8,
    ):
        """
        Initialize the PaiNN layer, made up of an interaction block and a mixing block.

        Args:
            hidden_size: Number of node features.
            activation: Activation function.
            layer_num: Numbering of the layer.
            blocks: Number of layers in the context networks.
            aggregate_fn: Function to aggregate the neighbors.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__(f"layer_{layer_num}")
        self._hidden_size = hidden_size
        self._eps = eps
        self._aggregate_fn = aggregate_fn

        # inter-particle context net
        self.interaction_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="interaction_block",
        )

        # intra-particle context net
        self.mixing_block = hk.Sequential(
            [LinearXav(hidden_size), activation] * (blocks - 1)
            + [LinearXav(3 * hidden_size)],
            name="mixing_block",
        )

        self.epsilon = epsilon

        # vector channel mix
        self.vector_mixing_block = LinearXav(
            2 * hidden_size,
            with_bias=False,
            name="vector_mixing_block",
        )

    def _message(
        self,
        s: jnp.ndarray,
        v: jnp.ndarray,
        dir_ij: jnp.ndarray,
        Wij: jnp.ndarray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Message/interaction. Inter-particle.

        Args:
            s (jnp.ndarray): Input scalar features (n_nodes, 1, hidden_size).
            v (jnp.ndarray): Input vector features (n_nodes, 3, hidden_size).
            dir_ij (jnp.ndarray): Direction of the edge (n_edges, 3).
            Wij (jnp.ndarray): Filter (n_edges, 1, 3 * hidden_size).
            senders (jnp.ndarray): Index of the sender node.
            receivers (jnp.ndarray): Index of the receiver node.

        Returns:
            Aggregated messages after interaction.
        """
        x = self.interaction_block(s)

        xj = x[receivers]
        vj = v[receivers]

        ds, dv1, dv2 = jnp.split(Wij * xj, 3, axis=-1)  # (n_edges, 1, hidden_size)
        n_nodes = tree.tree_leaves(s)[0].shape[0]
        dv = dv1 * dir_ij[..., jnp.newaxis] + dv2 * vj  # (n_edges, 3, hidden_size)
        # aggregate scalars and vectors
        ds = self._aggregate_fn(ds, senders, n_nodes)
        dv = self._aggregate_fn(dv, senders, n_nodes)

        s = (s + ds) * self.epsilon
        v = (v + dv) * self.epsilon

        return s, v

    def _update(
        self, s: jnp.ndarray, v: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update/mixing. Intra-particle.

        Args:
            s (jnp.ndarray): Input scalar features (n_nodes, 1, hidden_size).
            v (jnp.ndarray): Input vector features (n_nodes, 3, hidden_size).

        Returns:
            Node features after update.
        """
        v_l, v_r = jnp.split(self.vector_mixing_block(v), 2, axis=-1)
        v_norm = jnp.sqrt(jnp.sum(v_r**2, axis=-2, keepdims=True) + self._eps)

        ts = jnp.concatenate([s, v_norm], axis=-1)  # (n_nodes, 1, 2 * hidden_size)
        ds, dv, dsv = jnp.split(self.mixing_block(ts), 3, axis=-1)
        dv = v_l * dv
        dsv = dsv * jnp.sum(v_r * v_l, axis=1, keepdims=True)

        s = (s + ds + dsv) * self.epsilon
        v = (v + dv) * self.epsilon
        return s, v

    def __call__(
        self,
        graph: jraph.GraphsTuple,
        Wij: jnp.ndarray,
    ):
        """Compute interaction output.

        Args:
            graph (jraph.GraphsTuple): Input graph.
            Wij (jnp.ndarray): Filter.

        Returns:
            atom features after interaction
        """
        s, v = graph.nodes
        s, v = self._message(s, v, graph.edges, Wij, graph.senders, graph.receivers)
        s, v = self._update(s, v)
        return graph._replace(nodes=NodeFeatures(s=s, v=v))


class PaiNN(hk.Module):
    """PaiNN - polarizable interaction neural network.

    References:
        [#painn1] Schütt, Unke, Gastegger:
        Equivariant message passing for the prediction of tensorial properties and
        molecular spectra.
        ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html
    """

    def __init__(
        self,
        hidden_size: int,
        n_layers: int,
        *args,
        avg_num_neighbors: float,
        n_rbf: int = 20,
        activation: Callable = jax.nn.silu,
        node_type: str = "discrete",
        task: str = "node",
        pool: str = "sum",
        out_channels: Optional[int] = None,
        readout_fn: ReadoutBuilderFn = PaiNNReadout,
        num_species: int = 100,
        shared_interactions: bool = False,
        shared_filters: bool = False,
        eps: float = 1e-8,
        **kwargs,
    ):
        """Initialize the model.

        Args:
            hidden_size: Determines the size of each embedding vector.
            n_layers: Number of interaction blocks.
            radial_basis_fn: Expands inter-particle distances in a basis set.
            cutoff_fn: Cutoff method. None means no cutoff.
            radius: Cutoff radius.
            n_rbf: Number of radial basis functions.
            activation: Activation function.
            node_type: Type of node features. Either "discrete" or "continuous".
            task: Regression task to perform. Either "node"-wise or "graph"-wise.
            pool: Node readout pool method. Only used in "graph" tasks.
            out_channels: Number of output scalar/vector channels. Used in readout.
            readout_fn: Readout function. If None, use default PaiNNReadout.
            max_z: Maximum atomic number. Used in discrete node feature embedding.
            shared_interactions: If True, share the weights across interaction blocks.
            shared_filters: If True, share the weights across filter networks.
            eps: Constant added in norm to prevent derivation instabilities.
        """
        super().__init__("painn")

        assert node_type in [
            "discrete",
            "continuous",
        ], "node_type must be discrete or continuous"
        assert task in ["node", "graph"], "task must be node or graph"

        self._hidden_size = hidden_size
        self._n_layers = n_layers
        self._eps = eps
        self._node_type = node_type
        self._shared_filters = shared_filters
        self._shared_interactions = shared_interactions

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )
        epsilon = 1 / jnp.sqrt(1 + epsilon ** 2)

        self.cutoff_fn = layers.SmoothingEnvelope(1.0)
        self.radial_basis_fn = layers.RadialBesselLayer(1.0, n_rbf)

        if node_type == "discrete":
            self.scalar_emb = hk.Embed(
                num_species,
                hidden_size,
                w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
                name="scalar_embedding",
            )
        else:
            self.scalar_emb = LinearXav(hidden_size, name="scalar_embedding")
        # mix vector channels (only used if vector features are present in input)
        self.vector_emb = LinearXav(
            hidden_size, with_bias=False, name="vector_embedding"
        )

        if shared_filters:
            self.filter_net = LinearXav(3 * hidden_size, name="filter_net")
        else:
            self.filter_net = LinearXav(n_layers * 3 * hidden_size, name="filter_net")

        if self._shared_interactions:
            self.layers = [
                PaiNNLayer(
                    hidden_size, 0, epsilon=epsilon,
                    activation=activation, eps=eps
                )
            ] * n_layers
        else:
            self.layers = [
                PaiNNLayer(
                    hidden_size, i, epsilon=epsilon, activation=activation,
                    eps=eps
                ) for i in range(n_layers)
            ]

        self.readout = None
        if out_channels is not None and readout_fn is not None:
            self.readout = readout_fn(
                *args,
                hidden_size,
                task,
                pool,
                out_channels=out_channels,
                activation=activation,
                eps=eps,
                **kwargs,
            )

    def _embed(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Embed the input nodes."""
        # embeds scalar features
        s = graph.nodes.s
        if self._node_type == "continuous":
            # e.g. velocities
            s = jnp.asarray(s, dtype=jnp.float32)
            if len(s.shape) == 1:
                s = s[:, jnp.newaxis]
        if self._node_type == "discrete":
            # e.g. atomic numbers
            s = jnp.asarray(s, dtype=jnp.int32)
        s = self.scalar_emb(s)[:, jnp.newaxis]  # (n_nodes, 1, hidden_size)

        # embeds vector features
        if graph.nodes.v is not None:
            # initialize the vector with the global positions
            v = graph.nodes.v
            v = self.vector_emb(v)  # (n_nodes, 3, hidden_size)
        else:
            # if no directional info, initialize the vector with zeros (as in the paper)
            v = jnp.zeros((s.shape[0], 3, s.shape[-1]))

        return graph._replace(nodes=NodeFeatures(s=s, v=v))

    def _get_filters(self, norm_ij: jnp.ndarray) -> jnp.ndarray:
        r"""Compute the rotationally invariant filters :math:`W_s`.

        .. math::
            W_s = MLP(RBF(\|\vector{r}_{ij}\|)) * f_{cut}(\|\vector{r}_{ij}\|)
        """
        phi_ij = self.radial_basis_fn(norm_ij)
        if self.cutoff_fn is not None:
            norm_ij = self.cutoff_fn(norm_ij)
        # compute filters
        filters = (
            self.filter_net(phi_ij) * norm_ij[:, jnp.newaxis]
        )  # (n_edges, 1, n_layers * 3 * hidden_size)
        # split into layer-wise filters
        if self._shared_filters:
            filter_list = [filters] * self._n_layers
        else:
            filter_list = jnp.split(filters, self._n_layers, axis=-1)
        return filter_list

    def __call__(self, graph: jraph.GraphsTuple) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute representations/embeddings.

        Args:
            inputs: GraphsTuple. The nodes should cointain a NodeFeatures object with
                - scalar feature of the shape (n_atoms, n_features)
                - vector feature of the shape (n_atoms, 3, n_features)

        Returns:
            Tuple with scalar and vector representations/embeddings.
        """
        # compute atom and pair features
        norm_ij = jnp.sqrt(jnp.sum(graph.edges**2, axis=1, keepdims=True) + self._eps)
        # edge directions
        # NOTE: assumes edge features are displacement vectors.
        dir_ij = graph.edges / (norm_ij + self._eps)
        graph = graph._replace(edges=dir_ij)

        # compute filters (r_ij track in message block from the paper)
        filter_list = self._get_filters(norm_ij)  # list (n_edges, 1, 3 * hidden_size)

        # embeds node scalar features (and vector, if present)
        graph = self._embed(graph)

        # message passing
        for n, layer in enumerate(self.layers):
            graph = layer(graph, filter_list[n])

        if self.readout is not None:
            # return decoded representations
            s, v = self.readout(graph)
        else:
            # return representations (last layer embedding)
            s, v = jnp.squeeze(graph.nodes.s), jnp.squeeze(graph.nodes.v)
        return s, v


def painn_neighborlist_pp(displacement: space.DisplacementFn,
                          r_cutoff: float,
                          n_species: int = 100,
                          positions_test: jnp.ndarray = None,
                          neighbor_test: partition.NeighborList = None,
                          max_edge_multiplier: float = 1.25,
                          max_edges=None,
                          avg_num_neighbors: float = None,
                          mode: str = "energy",
                          per_particle: bool = False,
                          positive_species: bool = False,
                          **painn_kwargs
                          ) -> Tuple[nn.InitFn, Callable[[Any, md_util.Array],
                                                                 md_util.Array]]:
    """Allegro model for property prediction.

    Args:
        displacement: Jax_md displacement function
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        nequip_escn: Use NequIPESCN instead of NequIP (more computational
            efficient).
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        nequip_kwargs: Kwargs to change the default structure of NequIP.
            For definition of the kwargs, see NequIP.


    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """

    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    # Checking only necessary if neighbor list is dense
    _avg_num_neighbors = None
    if positions_test is not None and neighbor_test is not None:
        if neighbor_test.format == partition.Dense:
            print('Capping edges and triplets. Beware of overflow, which is'
                  ' currently not being detected.')

            testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
                displacement, positions_test, neighbor_test, r_cutoff)
            max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

            # cap maximum edges and angles to avoid overflow from multiplier
            n_particles, n_neighbors = neighbor_test.idx.shape
            max_edges = min(max_edges, n_particles * n_neighbors)

            print(f"Estimated max. {max_edges} edges.")

            _avg_num_neighbors = testgraph.n_edges / n_particles
        else:
            n_particles = neighbor_test.idx.shape[0]
            _avg_num_neighbors = onp.sum(neighbor_test.idx[0] < n_particles)
            _avg_num_neighbors /= n_particles

    if avg_num_neighbors is None:
        avg_num_neighbors = _avg_num_neighbors
    assert avg_num_neighbors is not None, (
        "Average number of neighbors not set and no test graph was provided."
    )

    @hk.without_apply_rng
    @hk.transform
    def model(position: md_util.Array,
              neighbor: partition.NeighborList,
              species: md_util.Array = None,
              mask: md_util.Array = None,
              **dynamic_kwargs):
        if species is None:
            print(f"[Allegro] Use default species")
            species = jnp.zeros(position.shape[0], dtype=jnp.int32)
        elif positive_species:
            species -= 1
        if mask is None:
            print(f"[Allegro] Use default mask")
            mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

        # Compute the displacements for all edges
        dyn_displacement = partial(displacement, **dynamic_kwargs)

        if neighbor.format == partition.Dense:
            graph, _ = sparse_graph.sparse_graph_from_neighborlist(
                dyn_displacement, position, neighbor, r_cutoff,
                species, max_edges=max_edges, species_mask=mask
            )
            senders = graph.idx_i
            receivers = graph.idx_j
        else:
            assert neighbor.idx.shape == (2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
            senders, receivers = neighbor.idx

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < position.shape[0],
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))
        vectors /= r_cutoff

        graph = jraph.GraphsTuple(
            nodes=NodeFeatures(species, None),
            edges=vectors,
            senders=senders,
            receivers=receivers,
            n_node=jnp.asarray([position.shape[0]]),
            n_edge=jnp.asarray([vectors.shape[0]]),
            globals=None,
        )

        net = PaiNN(
            out_channels=1,
            avg_num_neighbors=avg_num_neighbors,
            **painn_kwargs
        )

        features = net(graph)

        if mode == "energy":
            per_node_energies, _ = features

            print(f"PaiNN output: {per_node_energies.shape}")

            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "energy_and_charge":
            raise NotImplementedError("Mode 'energy_and_charge' not implemented.")
        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)

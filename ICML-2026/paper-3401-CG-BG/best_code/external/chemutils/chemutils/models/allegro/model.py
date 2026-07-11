# MIT License
#
# Copyright (c) 2024 Mario Geiger
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

"""Custom implementation of Allegro-Jax based on the implementation available
at https://github.com/mariogeiger/allegro-jax.
 """

from functools import partial
from typing import Callable, Any, Tuple, List, Union, Iterable

import haiku as hk
import jax
import jaxopt
from jax import random, lax, numpy as jnp, nn as jax_nn, tree_util
from jax_md import space, partition, nn, util, energy, smap
import e3nn_jax as e3nn

import numpy as onp

from jax_md_mod.model import layers, sparse_graph
from jax_md import util as md_util

from chemutils.models.layers import AtomicEnergyLayer, CELLI


class Allegro(hk.Module):
    """Allegro for molecular property prediction.

    This model takes as input a sparse representation of a molecular graph
    - consisting of pairwise distances and angular triplets - and predicts
    pairwise properties. Global properties can be obtained by summing over
    pairwise predictions.

    This custom implementation follows the original Allegro-Jax
    (https://github.com/mariogeiger/allegro-jax).
    """
    def __init__(self,
                 avg_num_neighbors: float,
                 max_ell: int = 3,
                 hidden_irreps: e3nn.Irreps = 128 * e3nn.Irreps("0o + 1o + 1e + 2e + 2o + 3o + 3e"),
                 output_irreps: e3nn.Irreps = e3nn.Irreps("0e"),
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 mlp_n_hidden: int = 1024,
                 mlp_n_layers: int = 3,
                 embed_n_hidden: Iterable[int] = (64, 128, 256),
                 species_embed: int = None,
                 num_species: int = 100,
                 envelope_p: int = 6,
                 n_radial_basis: int = 8,
                 num_layers: Union[int, Iterable[int]] = 1, #3,
                 charge_embed_n_hidden: int = 16,
                 charge_embed_n_layers: int = 1,
                 name: str = 'Allegro',
                 charge_eq_fn: Callable = None,
                 *args, **kwargs):
        """Initializes the Allegro model

        Args:
            avg_num_neighbors: Average number of neighboring atoms.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.
            irreps: Irreducible representations to consider.
            mlp_activation: Activation function in MLPs.
            mlp_n_hidden: Number of nodes in hidden layers of two-body MLP and
                          latent MLP.
            mlp_n_layers: Number of hidden layers in MLPs.
            embed_n_hidden: Number of nodes in hidden layers of embedding MLP.
            species_embed: Dimension of species embedding. If None, use half
                the size of the first hidden layer.
            p: Polynomial order of polynomial envelope for weighting two-body
               features by atomic distance.
            n_radial_basis: Number of Bessel basis functions.
            radial_cutoff: Radial cut-off distance of edges.
            output_irreps: Irreducible representations in output layer.
            num_layers: Number of tensor product layers.
            name: Name of Allegro model.
        """
        super().__init__(name=name)
        self.output_irreps = output_irreps
        self.mlp_n_hidden = mlp_n_hidden

        epsilon = hk.get_parameter(
            "varepsilon", shape=(),
            init=hk.initializers.Constant(jnp.sqrt(avg_num_neighbors))
        )

        self.alpha = hk.get_parameter(
            "residual_alpha", shape=(), init=hk.initializers.Constant(0.0)
        )

        self.particle_energy = hk.Embed(num_species, 1)

        self.envelope_fn = layers.SmoothingEnvelope(envelope_p)

        # Learnable normalization depending on the number of neighbors
        self.epsilon = 1 / jnp.sqrt(1 + jax_nn.softplus(epsilon))

        if charge_eq_fn is not None:
            try:
                pre_qeq, post_qeq = num_layers
            except TypeError:
                pre_qeq = num_layers
                post_qeq = num_layers

        else:
            pre_qeq = num_layers - 1
            post_qeq = 0

        self.irreps_layers = [hidden_irreps] * (pre_qeq + post_qeq + 1) + [output_irreps]
        self.irreps_layers = self.filter_layers(self.irreps_layers, max_ell)
        self.irreps_layers = [e3nn.Irreps(irreps) for irreps in self.irreps_layers]

        self.embedding_layer = AllegroEmbedding(
            num_species=num_species,
            embed_n_hidden=embed_n_hidden,
            species_embed=species_embed,
            n_radial_basis=n_radial_basis,
            envelope_p=envelope_p,
            mlp_n_hidden=mlp_n_hidden,
            mlp_activation=mlp_activation,
            irreps=self.irreps_layers[0]
        )

        self.layers = []
        for idx, irreps in enumerate(self.irreps_layers[1:]):
            if charge_eq_fn is not None and idx == pre_qeq:
                self.layers.append(
                    CELLI(
                        charge_embed_n_hidden=charge_embed_n_hidden,
                        charge_embed_n_layers=charge_embed_n_layers,
                        num_species=num_species,
                        mlp_n_hidden=mlp_n_hidden,
                        mlp_n_layers=mlp_n_layers,
                        mlp_activation=mlp_activation,
                        envelope_p=envelope_p,
                        charge_eq_fn=charge_eq_fn,
                    )
                )
            else:
                self.layers.append(
                    AllegroLayer(
                        epsilon=self.epsilon,
                        max_ell=max_ell,
                        output_irreps=irreps,
                        mlp_activation=mlp_activation,
                        mlp_n_hidden=mlp_n_hidden,
                        mlp_n_layers=mlp_n_layers,
                        p=envelope_p,
                    )
                )

        self.readout_layer = AllegroReadout(
            output_n_hidden=mlp_n_hidden, output_n_layers=1,
            output_activation=mlp_activation, envelope_p=envelope_p,
            output_irreps=e3nn.Irreps(output_irreps)
        )


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,  # [n_edges, 3]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        species: jnp.ndarray,  # [n_nodes]
    ) -> e3nn.IrrepsArray:
        """Predicts pairwise quantities for a given conformation.

        Args:
            node_attrs: Species information (jax.nn.one_hot(z, num_species)).
            vectors: Relative displacement vectors r_{ij} between neighboring
                     atoms from i to j.
            senders: Indices of central atoms i.
            receivers: Indices of neighboring atoms j.
            edge_feats:
            is_training: Set model into training mode (True: dropout is applied).

        Returns:
            An array of predicted pairwise quantities in irreducible representations.
        """
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)

        assert vectors.irreps in ["1o", "1e"]

        # Embedding layer
        x, V = self.embedding_layer(vectors, senders, receivers, species)

        charge_out = None
        for layer in self.layers:
            out = layer(vectors, x, V, senders, species)
            if len(out) == 3:
                y, V, charge_out = out
            else:
                y, V = out

            # Perform residual update through a weighted sum
            x = (x + jax_nn.softplus(self.alpha) * y)
            x /= (1 + jax_nn.softplus(self.alpha))

        # Readout layer

        xV = self.readout_layer(vectors, x, V)

        if charge_out is None:
            return xV
        else:
            return xV, charge_out


    def filter_layers(self, layer_irreps: List[e3nn.Irreps], max_ell: int) -> List[e3nn.Irreps]:
        """Shape irreducible representations of tensor product layers in order
        to match desired output irreps via tensor products.

        Args:
            layer_irreps: Irreducible representations of tensor product layers.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.

        Returns:
            Updated list of irreducible representations of tensor product layers.
        """
        layer_irreps = list(layer_irreps)
        # initialize filtered as last layer from layer_irreps
        filtered = [e3nn.Irreps(layer_irreps[-1])]
        # propagate through network from output to input
        for irreps in reversed(layer_irreps[:-1]):
            irreps = e3nn.Irreps(irreps)
            # filter all irreps, remaining should satisfy tensor product with consecutive layer
            irreps = irreps.filter(
                # tensor product of consecutive layers in the nn structure
                keep=e3nn.tensor_product(
                    # irreps of subsequent layer in nn structure
                    filtered[0],
                    # only irreps considering spherical harmonics (eg. '1x0e+1x1o+1x2e+1x3o' for lmax=3)
                    e3nn.Irreps.spherical_harmonics(lmax=max_ell),
                ).regroup() # regroup the same irreps together
            )
            filtered.insert(0, irreps)
        return filtered


class AllegroEmbedding(hk.Module):

    def __init__(self,
                 num_species: int = 100,
                 embed_n_hidden: Iterable[int] = (64, 128, 256),
                 species_embed: int = None,
                 n_radial_basis: int = 8,
                 envelope_p: int = 6,
                 mlp_n_hidden: int = 64,
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 *,
                 irreps: e3nn.Irreps,
                 ):
        super().__init__()

        if species_embed is None:
            species_embed = list(embed_n_hidden)[0] // 2

        self.species_embedding = hk.Embed(num_species, species_embed)
        self.radial_basis = layers.RadialBesselLayer(
            cutoff=1.0, num_radial=n_radial_basis, envelope_p=envelope_p
        )

        self.irreps = irreps

        self.envelope_p = envelope_p

        self.embed_n_hidden = list(embed_n_hidden) + [mlp_n_hidden]
        self.mlp_activation = mlp_activation

    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        species: jnp.ndarray,
    ) -> e3nn.IrrepsArray:

        # Shape checks
        num_edges = vectors.shape[0]

        assert vectors.shape == (num_edges, 3)
        assert senders.shape == (num_edges,)
        assert receivers.shape == (num_edges,)
        assert vectors.irreps in ["1o", "1e"]

        # Two-body graph embedding
        d = e3nn.norm(vectors).array.squeeze(1)
        x = jnp.concatenate(
            [
                self.radial_basis(d),
                self.species_embedding(species[senders]),
                self.species_embedding(species[receivers]),
            ],
            axis=1,
        )

        x = e3nn.haiku.MultiLayerPerceptron(
            self.embed_n_hidden,
            self.mlp_activation,
            output_activation=False,
        )(x)

        # Smooth truncation of two-body features
        x = layers.SmoothingEnvelope(self.envelope_p)(d)[:, None] * x

        # Tensorial embedding. Keep only irreps from first layer that satisfy
        # mirroring of input vectors and apply spherical harmonics according to
        # irreps of first layer to vectors
        irreps_Y = self.irreps.filter(
            keep=lambda mir: vectors.irreps[0].ir.p ** mir.ir.l == mir.ir.p
        )
        V = e3nn.spherical_harmonics(irreps_Y, vectors, True)
        V = e3nn.concatenate([
            V,
            self.species_embedding(species[senders]),
            self.species_embedding(species[receivers]),
        ])

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.num_irreps,),None)(x)
        V = w * V / V.irreps.num_irreps
        assert V.shape == (num_edges, V.irreps.dim)

        return x, V


class AllegroReadout(hk.Module):

    def __init__(self,
                 output_n_hidden: int = 64,
                 output_n_layers: int = 1,
                 output_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax_nn.silu,
                 envelope_p: int = 6,
                 *,
                 output_irreps: e3nn.Irreps,
                 ):
        super().__init__()

        self.mlp = e3nn.haiku.MultiLayerPerceptron(
            (output_n_hidden,) * output_n_layers,
            output_activation, output_activation=False)
        self.linear = e3nn.haiku.Linear(output_irreps)


        self.output_irreps = output_irreps
        self.envelope_fn = layers.SmoothingEnvelope(envelope_p)


    def __call__(self, vectors, x: jnp.ndarray, V: e3nn.IrrepsArray) -> jnp.ndarray:

        x = self.mlp(x)
        xV = self.linear(e3nn.concatenate([x, V]))

        if xV.irreps != self.output_irreps:
            raise ValueError(
                f"output_irreps {self.output_irreps} is not reachable from "
                f"irreps {xV.irreps} ({xV.irreps == self.output_irreps}) and ({xV.irreps != self.output_irreps})."
            )

        lengths = jnp.sum(vectors.array ** 2, axis=-1) ** 0.5
        xV = self.envelope_fn(lengths)[:, None] * xV

        return xV


class AllegroLayer(hk.Module):
    """Tensor product layer of Allegro.

    This model updates invariant two-body features and equivariant latent
    features by applying tensor prodict operations.

    This custom implementation follows the original Allegro-Jax
    (https://github.com/mariogeiger/allegro-jax).
    """
    def __init__(self,
                 epsilon: float,
                 max_ell: int = 3,
                 output_irreps: e3nn.Irreps = 64 * e3nn.Irreps("0e + 1o + 2e"),
                 mlp_activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.silu,
                 mlp_n_hidden: int = 64,
                 mlp_n_layers: int = 3,
                 p: int = 6,
                 name: str = 'TensorProduct'):
        """Tensor product layer.

        Args:
            avg_num_neighbors: Average number of neighboring atoms.
            max_ell: Maximum rotation order (l_{max}) to which features in the
                     network are truncated.
            irreps: Irreducible representations to consider.
            mlp_activation: Activation function in MLPs.
            mlp_n_hidden: Number of nodes in hidden layers of two-body MLP and
                          latent MLP.
            mlp_n_layers: Number of hidden layers in MLPs.
            p: Polynomial order of polynomial envelope for weighting two-body
               features by atomic distance d.
            name: Name of Embedding block.
        """
        super().__init__(name=name)
        self.epsilon = epsilon
        self.max_ell = max_ell
        self.output_irreps = output_irreps
        self.mlp_activation = mlp_activation
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_n_layers = mlp_n_layers
        self.envelope_p = p


    def __call__(
        self,
        vectors: e3nn.IrrepsArray,
        x: jnp.ndarray,
        V: e3nn.IrrepsArray,
        senders: jnp.ndarray,
        species: jnp.ndarray,
    ) -> e3nn.IrrepsArray:
        """Returns output of the Tensor product layer."""
        num_edges = vectors.shape[0]
        assert vectors.shape == (num_edges, 3)
        assert x.shape == (num_edges, x.shape[-1])
        assert V.shape == (num_edges, V.irreps.dim)
        assert senders.shape == (num_edges,)

        irreps_out = e3nn.Irreps(self.output_irreps)

        w = e3nn.haiku.MultiLayerPerceptron((V.irreps.mul_gcd,),None)(x)
        Y = e3nn.spherical_harmonics(range(self.max_ell + 1), vectors, True)
        wY = e3nn.scatter_sum(
            w[:, :, None] * Y[:, None, :], dst=senders, map_back=True
        ) * self.epsilon
        assert wY.shape == (num_edges, V.irreps.mul_gcd, wY.irreps.dim)

        V = e3nn.tensor_product(
            wY, V.mul_to_axis(), filter_ir_out="0e" + irreps_out
        ).axis_to_mul()

        print(f"Out irreps {V.irreps}")

        if "0e" in V.irreps:
            x = jnp.concatenate([x, V.filter(keep="0e").array], axis=1)
            V = V.filter(drop="0e")

        x = e3nn.haiku.MultiLayerPerceptron(
            (self.mlp_n_hidden,) * self.mlp_n_layers,
            self.mlp_activation,
            output_activation=False,
        )(x)

        lengths = e3nn.norm(vectors).array
        x = layers.SmoothingEnvelope(self.envelope_p)(lengths) * x
        assert x.shape == (num_edges, self.mlp_n_hidden)

        V = e3nn.haiku.Linear(irreps_out)(V)
        assert V.shape == (num_edges, V.irreps.dim)
        print(f"Irreps after layer are {V.irreps}")

        return (x, V)


def allegro_qeq_neighborlist_pp(displacement: space.DisplacementFn,
                                charge_eq_fn: Callable,
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
                                learn_radius: bool = False,
                                **allegro_kwargs
                                ) -> Tuple[nn.InitFn,
                                           Callable[[Any, md_util.Array],
                                                    md_util.Array]]:
    """Allegro model for property prediction.

    Args:
        displacement: Jax_md displacement function
        charge_eq_fn: Function computing charges via the QEM method based
            on electronegativities, hardnesses, and radii predicted by
            the neural network.
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list
        n_species: Number of different atom species the network is supposed
            to process.
        positions_test: Sample positions to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        neighbor_test: Sample neighborlist to estimate max_edges / max_angles.
            Needs to be provided to enable capping.
        max_edge_multiplier: Multiplier for initial estimate of maximum edges.
        max_edges: Expected maximum of valid edges.
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        learn_radius: True if the radius of the atom for electrostatic
            interactions should be learned.
        allegro_kwargs: Kwargs to overwrite default values of the Allegro model.
            See :class:`Allegro` for available kwargs.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    r_cutoff = jnp.array(r_cutoff, dtype=md_util.f32)

    assert not per_particle, "Per-particle energies not yet implemented."

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
    assert max_edges is not None, (
        "Requires maximum number of edges within NN cutoff."
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

        # Remove all edges between replicated atoms
        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff)
        vectors /= r_cutoff

        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

        vectors = e3nn.IrrepsArray(
            e3nn.Irreps("1o"), vectors
        )

        def _charge_eq_fn(gammas, chis, hardness):
            assert "radius" in dynamic_kwargs.keys(), "Radius not in dynamic_kwargs."

            if learn_radius:
                gammas *= dynamic_kwargs["radius"]
            else:
                gammas = dynamic_kwargs["radius"]

            _, charges = charge_eq_fn(
                position, neighbor, gammas, chi=chis, idmp=hardness,
                mask=mask, **dynamic_kwargs
            )

            # Do not optimize hardness and gammas on energy and force (only
            # indirectly through charges)
            pot = charge_eq_fn(
                position, neighbor, jax.lax.stop_gradient(gammas), mask=mask,
                charge=charges, **dynamic_kwargs
            )

            return charges, pot

        net = Allegro(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            mlp_activation=jax_nn.mish,
            charge_eq_fn=_charge_eq_fn,
            **allegro_kwargs
        )

        features, qeq_features = net(vectors, senders, receivers, species)

        if mode in ["energy", "energy_and_charge"]:
            per_edge_energies, = features.array.T

            per_node_energies = layers.high_precision_segment_sum(
                per_edge_energies, senders, num_segments=position.shape[0])

            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies *= mask

            charges, elec_pot = qeq_features
            total_pot = elec_pot + md_util.high_precision_sum(per_atom_energies)

            if mode == "energy_and_charge":
                return total_pot, charges
            else:
                return total_pot

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)


def allegro_neighborlist_pp(displacement: space.DisplacementFn,
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
                            **allegro_kwargs
                            ) -> Tuple[nn.InitFn,
                                       Callable[[Any, md_util.Array],
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
        avg_num_neighbors: Average number of neighbors per particle. Guessed
            if positions_test and neighbor_test are provided.
        mode: Prediction mode of the model. If "property_prediction" (default),
            returns the learned node features. If "energy_prediction", returns
            the total energy of the system.
        per_particle: Return per-particle energies instead of total energy.
        positive_species: True if the smallest occurring species is 1, e.g., in
            case of atomic numbers.
        allegro_kwargs: Kwargs to overwrite default values of the Allegro model.
            See :class:`Allegro` for available kwargs.

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
        if species is not None:
            print(f"[Allegro] Use two atom species for oxygen and hydroge.")
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

        # Remove all edges between replicated atoms
        invalid_idx = position.shape[0]

        # Set invalid edges to the cutoff to avoid numerical issues
        vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
        vectors = jnp.where(
            jnp.logical_and(
                senders < invalid_idx,
                receivers < position.shape[0]
            )[:, jnp.newaxis], vectors, r_cutoff)
        vectors /= r_cutoff

        vectors = e3nn.IrrepsArray(e3nn.Irreps("1o"), vectors)

        net = Allegro(
            avg_num_neighbors=avg_num_neighbors,
            num_species=n_species,
            mlp_activation=jax_nn.mish,
            **allegro_kwargs
        )

        features = net(vectors, senders, receivers, species)

        if mode == "energy":
            per_edge_energies, = features.array.T

            per_node_energies = layers.high_precision_segment_sum(
                per_edge_energies, senders, num_segments=position.shape[0])

            per_atom_energies = AtomicEnergyLayer(n_species)(per_node_energies, species)
            per_atom_energies *= mask

            if per_particle:
                return per_atom_energies
            else:
                return md_util.high_precision_sum(per_atom_energies)

        elif mode == "property_prediction":
            return features
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

    return jax.jit(model.init), jax.jit(model.apply)

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

"""Neural network models for potential energy and molecular property
prediction.
 """
from functools import partial
from typing import Callable, Dict, Any, Tuple, Union

import haiku as hk
from jax import numpy as jnp, nn as jax_nn
from jax_md import smap, space, partition, nn, util

from jax_md_mod.model import dropout, layers, sparse_graph


class DimeNetPP(hk.Module):
    """DimeNet++ for molecular property prediction.

    This model takes as input a sparse representation of a molecular graph
    - consisting of pairwise distances and angular triplets - and predicts
    per-atom properties. Global properties can be obtained by summing over
    per-atom predictions.

    The default values correspond to the orinal values of DimeNet++.

    This custom implementation follows the original DimeNet
    [#Gasteiger2022Dimenet]_ / DimeNet++ [#Gasteiger2022Dimenetpp]_
    (Directional Message Passing Neural Network), while correcting for known
    issues (see https://github.com/klicperajo/dimenet).

    References:
       .. [#Gasteiger2022Dimenet] Gasteiger, J.; Groß, J.; Günnemann,
          S. Directional Message Passing for Molecular Graphs. arXiv April 5,
          2022. https://doi.org/10.48550/arXiv.2003.03123.
       .. [#Gasteiger2022Dimenetpp] Gasteiger, J.; Giri, S.; Margraf, J. T.;
          Günnemann, S. Fast and Uncertainty-Aware Directional Message Passing
          for Non-Equilibrium Molecules. http://arxiv.org/abs/2011.14115.

    """
    def __init__(self,
                 r_cutoff: float,
                 n_species: int,
                 num_targets: int,
                 kbt_dependent: bool = False,
                 embed_size: int = 128,
                 n_interaction_blocks: int = 4,
                 num_residual_before_skip: int = 1,
                 num_residual_after_skip: int = 2,
                 out_embed_size: int = None,
                 type_embed_size: int = None,
                 angle_int_embed_size: int = None,
                 basis_int_embed_size: int = 8,
                 num_dense_out: int = 3,
                 num_rbf: int = 6,
                 num_sbf: int = 7,
                 activation: Callable = jax_nn.swish,
                 envelope_p: int = 6,
                 init_kwargs: Dict[str, Any] = None,
                 dropout_mode: Dict[str, Any] = None,
                 name: str = 'DimeNetPP',
                 outscale: bool = False):
        """Initializes the DimeNet++ model

        The default values correspond to the orinal values of DimeNet++.

        Args:
            r_cutoff: Radial cut-off distance of edges
            n_species: Number of different atom species the network is supposed
                to process.
            num_targets: Number of different atomic properties to predict
            kbt_dependent: True, if DimeNet explicitly depends on temperature.
                In this case 'kT' needs to be provided as a kwarg during the
                model call to the energy_fn. Default False results in a model
                independent of temperature.
            embed_size: Size of message embeddings. Scale interaction and output
                embedding sizes accordingly, if not specified explicitly.
            n_interaction_blocks: Number of interaction blocks
            num_residual_before_skip: Number of residual blocks before the skip
                connection in the Interaction block.
            num_residual_after_skip: Number of residual blocks after the skip
                connection in the Interaction block.
            out_embed_size: Embedding size of output block. If None is set to
                2 * embed_size.
            type_embed_size: Embedding size of atom type embeddings. If None is
                set to 0.5 * embed_size.
            angle_int_embed_size: Embedding size of Linear layers for
                down-projected triplet interation. If None is 0.5 * embed_size.
            basis_int_embed_size: Embedding size of Linear layers for interation
                of RBS/ SBF basis in interaction block
            num_dense_out: Number of final Linear layers in output block
            num_rbf: Number of radial Bessel embedding functions
            num_sbf: Number of spherical Bessel embedding functions
            activation: Activation function
            envelope_p: Power of envelope polynomial
            init_kwargs: Kwargs for initializaion of Linear layers
            dropout_mode: A dict defining which fully connected layers to apply
                dropout and at which rate (see dropout.dimenetpp_setup).
                If None, no Dropout is applied.
            name: Name of DimeNet++ model
            outscale: Scale the output initially return zero energy
        """
        super().__init__(name=name)
        self.dropout_setup = dropout.dimenetpp_setup(dropout_mode,
                                                     num_dense_out,
                                                     n_interaction_blocks,
                                                     num_residual_before_skip,
                                                     num_residual_after_skip)
        if init_kwargs is None:
            init_kwargs = {
                'w_init': layers.OrthogonalVarianceScalingInit(scale=1.),
                'b_init': hk.initializers.Constant(0.),
            }

        # input representation:
        self.r_cutoff = r_cutoff
        self._rbf_layer = layers.RadialBesselLayer(r_cutoff, num_rbf,
                                                   envelope_p)
        self._sbf_layer = layers.SphericalBesselLayer(r_cutoff, num_sbf,
                                                      num_rbf, envelope_p)

        # build GNN structure
        self._n_interactions = n_interaction_blocks
        self._output_blocks = []
        self._int_blocks = []
        self._embedding_layer = layers.EmbeddingBlock(
            embed_size, n_species, type_embed_size, activation, init_kwargs,
            kbt_dependent)
        self._output_blocks.append(layers.OutputBlock(
            embed_size, out_embed_size, num_dense_out, num_targets, activation,
            init_kwargs, outscale=outscale)
        )

        for _ in range(n_interaction_blocks):
            self._int_blocks.append(layers.InteractionBlock(
                embed_size, num_residual_before_skip, num_residual_after_skip,
                activation, init_kwargs, angle_int_embed_size,
                basis_int_embed_size)
            )
            self._output_blocks.append(layers.OutputBlock(
                embed_size, out_embed_size, num_dense_out, num_targets,
                activation, init_kwargs, outscale=outscale)
            )

    def __call__(self,
                 graph: sparse_graph.SparseDirectionalGraph,
                 **dyn_kwargs) -> jnp.ndarray:
        """Predicts per-atom quantities for a given molecular graph.

        Args:
            graph: An instance of sparse_graph.SparseDirectionalGraph defining
                the molecular graph connectivity.
            **dyn_kwargs: Kwargs supplied on-the-fly, such as 'kT' for
                temperature-dependent models or 'dropout_key' for Dropout.

        Returns:
            An (n_partciles, num_targets) array of predicted per-atom quantities
        """
        dropout_key = dyn_kwargs.get('dropout_key', None)
        dropout_params = dropout.construct_dropout_params(dropout_key,
                                                          self.dropout_setup)
        n_particles = graph.species.size
        # cutoff all non-existing edges: are encoded as 0 by rbf envelope
        # non-existing triplets will be masked explicitly in DimeNet++
        pair_distances = jnp.where(graph.edge_mask, graph.distance_ij,
                                   2. * self.r_cutoff)

        rbf = self._rbf_layer(pair_distances)
        # explicitly masked via mask array in angular_connections
        sbf = self._sbf_layer(pair_distances, graph.angles, graph.triplet_mask,
                              graph.expand_to_kj)

        messages = self._embedding_layer(rbf, graph.species, graph.idx_i,
                                         graph.idx_j, dropout_params,
                                         **dyn_kwargs)
        per_atom_quantities = self._output_blocks[0](
            messages, rbf, graph.idx_i, n_particles, dropout_params)

        for i in range(self._n_interactions):
            messages = self._int_blocks[i](
                messages, rbf, sbf, graph.reduce_to_ji, graph.expand_to_kj,
                dropout_params)
            per_atom_quantities += self._output_blocks[i + 1](
                messages, rbf, graph.idx_i, n_particles, dropout_params)
        return per_atom_quantities


def dimenetpp_neighborlist(displacement: space.DisplacementFn,
                           r_cutoff: float,
                           n_species: int = 100,
                           positions_test: jnp.ndarray = None,
                           neighbor_test: partition.NeighborList = None,
                           max_triplet_multiplier: float = 1.25,
                           max_edge_multiplier: float = 1.25,
                           max_edges=None,
                           max_triplets=None,
                           mode: Union[str, Tuple[int]] = "energy",
                           per_particle: bool = False,
                           **dimenetpp_kwargs
                           ) -> Tuple[nn.InitFn, Callable[[Any, util.Array],
                                                          Tuple[util.Array]]]:
    """DimeNet++ energy function for Jax, M.D.

    This function provides an interface for the :class:`DimeNetPP` haiku model
    to be used as a jax_md energy_fn. Analogous to jax_md energy_fns, the
    initialized DimeNet++ energy_fn requires particle positions and a dense
    neighbor list as input - plus an array for species or other dynamic kwargs,
    if applicable.

    From particle positions and neighbor list, the sparse graph representation
    with edges and angle triplets is computed. Due to the constant shape
    requirement of jit of the neighborlist in jax_md, the neighbor list contains
    many masked edges, i.e. pairwise interactions that only "fill" the neighbor
    list, but are set to 0 during computation. This translates to masked edges
    and triplets in the sparse graph representation.

    For improved computational efficiency during jax_md simulations, the
    maximum number of edges and triplets can be estimated during model
    initialization. Edges and triplets beyond this maximum estimate can be
    capped to reduce computational and memory requirements. Capping is enabled
    by providing sample inputs (positions_test and neighbor_test) at
    initialization time. However, beware that currently, an overflow of
    max_edges and max_angles is not caught, as this requires passing an error
    code throgh jax_md simulators - analogous to the overflow detection in
    jax_md neighbor lists. If in doubt, increase the max edges/angles
    multipliers or disable capping.

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
        max_triplet_multiplier: Multiplier for initial estimate of maximum
            triplets.
        max_edges: Expected maximum of valid edges.
        max_triplets: Expected maximum of valid triplets.
        mode: Select what the model should return. If set to 'energy' (default),
            returns a single scalar quantity, i.e., the potential energy.
            If set to tuple ``(n_global, n_local)``, returns a tuple of global
            and local atomic properties.
        per_particle: If True, returns the energy per particle.
        dimenetpp_kwargs: Kwargs to change the default structure of DimeNet++.
            For definition of the kwargs, see DimeNetPP.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters. The energy function requires the same input as
        other energy functions with neighbor lists in jax_md.energy.
    """
    r_cutoff = jnp.array(r_cutoff, dtype=util.f32)

    if positions_test is not None and neighbor_test is not None:
        print('Capping edges and triplets. Beware of overflow, which is'
              ' currently not being detected.')

        testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
            displacement, positions_test, neighbor_test, r_cutoff)
        max_triplets = jnp.int32(jnp.ceil(testgraph.n_triplets
                                          * max_triplet_multiplier))
        max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

        # cap maximum edges and angles to avoid overflow from multiplier
        n_particles, n_neighbors = neighbor_test.idx.shape
        max_edges = min(max_edges, n_particles * n_neighbors)
        max_triplets = min(max_triplets, n_particles * n_neighbors**2)

        print(f"Estimated max. {max_edges} edges and max. {max_triplets} triplets.")

    @hk.without_apply_rng
    @hk.transform
    def model(positions: jnp.ndarray,
              neighbor: partition.NeighborList,
              species: jnp.ndarray = None,
              **dynamic_kwargs) -> Union[jnp.ndarray,
                                         Tuple[jnp.ndarray, jnp.ndarray]]:
        """Evalues the DimeNet++ model and predicts the potential energy.

        Args:
            positions: Jax_md state-position. (N_particles x dim) array of
                particle positions
            neighbor: Jax_md dense neighbor list corresponding to positions
            species: (N_particles,) Array encoding atom types. If None, assumes
                all particles to belong to the same species
            **dynamic_kwargs: Dynamic kwargs, such as 'box' or 'kT'.

        Returns:
            If ``n_global = 1`` and ``n_local = 0`` (default), returns the
            potential energy.
            If ``n_global > 0`` xor ``n_local > 0``, returns a vector of
            global, respectively local, features.
            Otherwise, returns the tuple ``(global_feats, local_feats)``.
        """
        # dynamic box necessary for pressure computation
        dynamic_displacement = partial(displacement, **dynamic_kwargs)

        graph_rep, overflow = sparse_graph.sparse_graph_from_neighborlist(
            dynamic_displacement, positions, neighbor, r_cutoff, species,
            max_edges, max_triplets
        )
        # TODO: return overflow to detect possible overflow
        del overflow

        if mode == "energy":
            n_global = 1
            n_local = 0
        else:
            n_global, n_local = mode

        net = DimeNetPP(r_cutoff, n_species, num_targets=n_global + n_local, **dimenetpp_kwargs)
        features = net(graph_rep, **dynamic_kwargs)

        if "mask" in dynamic_kwargs:
            features *= dynamic_kwargs["mask"][:, jnp.newaxis]

        # Default behaviour
        if mode == "energy":
            if per_particle:
                return features.squeeze(axis=-1)
            else:
                return util.high_precision_sum(features)

        global_features = util.high_precision_sum(
            features[:, :n_global], axis=0)
        local_features = features[:, n_global:]

        return global_features, local_features

    return dropout.model_init_apply(model, dimenetpp_kwargs)


def dimenetpp_property_prediction(
        r_cutoff: float,
        n_targets: int = 1,
        n_species: int = 100,
        n_per_atom: int = None,
        **model_kwargs) -> Tuple[nn.InitFn, Callable[[Any, jnp.ndarray],
                                                     jnp.ndarray]]:
    """Initializes DimeNet++ to predict molecular properties.

    This function provides an interface to the :class:`DimeNetPP` haiku model.

    Args:
        r_cutoff: Radial cut-off distance of DimeNetPP and the neighbor list.
        n_targets: Number of different molecular properties to predict.
        n_species: Number of different atom species the network is supposed
            to process.
        n_per_atom: Number of per-atom predictions, i.e., predicted quantity
            equals the number of atoms. The remaining predictions are considered
            global, i.e., one prediction per molecule.
        **model_kwargs: Kwargs to change the default structure of DimeNet++.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an apply_function that predicts global molecular properties.
    """
    @hk.without_apply_rng
    @hk.transform
    def property_predictor(
            mol_graph: sparse_graph.SparseDirectionalGraph,
            **dynamic_kwargs):
        """Predicts global quantities for a given molecular graph.

        Args:
            mol_graph: An instance of sparse_graph.SparseDirectionalGraph
                defining the molecular graph connectivity.
            dynamic_kwargs: Dynamic kwargs for DimeNet++.

        Returns:
            An (n_targets,) array of predicted global quantities.
        """
        model = DimeNetPP(r_cutoff, n_species, n_targets, **model_kwargs)
        per_atom_predictions = model(mol_graph, **dynamic_kwargs)

        # Split the global predictions from the per-atom predictions.
        # Assume that the global predictions are a sum of the per-atom
        # predictions.
        n_predicted = per_atom_predictions.shape[1]
        n_global = n_predicted - n_per_atom
        per_atom_properties = per_atom_predictions[:, n_global:]
        global_properties = jnp.sum(per_atom_predictions[:, :n_global], axis=0)
        return global_properties, per_atom_properties
    return dropout.model_init_apply(property_predictor, model_kwargs)


class PairwiseNN(hk.Module):
    """A neural network predicting pairwise edge quantities

     Can be used for energy prediction for pairwise interactions.
     """
    def __init__(self,
                 r_cutoff: float,
                 hidden_layers,
                 init_kwargs: Dict = None,
                 activation: Callable = jax_nn.swish,
                 num_rbf: int = 6,
                 envelope_p: int = 6,
                 name: str = 'PairNN'):
        super().__init__(name=name)

        if init_kwargs is None:
            init_kwargs = {
                'w_init': layers.OrthogonalVarianceScalingInit(scale=1.),
                'b_init': hk.initializers.Constant(0.),
            }

        assert num_rbf > 0, 'The number of RBF embeddings but be at least 1.'
        self.cutoff = r_cutoff
        self.embedding = layers.RadialBesselLayer(r_cutoff, num_radial=num_rbf,
                                                  envelope_p=envelope_p)
        self.pair_nn = hk.nets.MLP(hidden_layers, activation=activation,
                                   **init_kwargs)
        self.rbf_transform = hk.Linear(1, with_bias=False, name='RBF_Transform',
                                       **init_kwargs)

    def __call__(self, distances, species=None, **kwargs):
        # ensure differentiability construction: rbf=0 for r > r_cut
        rbf = self.embedding(distances)
        predicted_energies = self.pair_nn(rbf)

        # rbf_transform has no bias: masked pairs remain 0 and counteract
        # possible non-zero contribution from biases in MPL (in a
        # continuously differentiable manner)
        per_pair_energy = predicted_energies * self.rbf_transform(rbf)
        return per_pair_energy


def pair_interaction_nn(displacement: space.DisplacementFn,
                        r_cutoff: float,
                        hidden_layers,
                        **pair_net_kwargs):
    """An MLP acting on pairwise distances independently and
    summing the contributions.

    Embeds pairwise distances via radial Bessel functions (RBF). The
    RBF is also used to enforce a differentiable cut-off.

    Args:
        displacement: Displacement function
        r_cutoff: Radial cut-off of pairwise interactions and neighbor list
        hidden_layers: A list (or scalar in the case of a single hidden layer)
            of number of neurons for each hidden layer in the MLP
        pair_net_kwargs: Kwargs to change the default structure of PairwiseNN.
            For definition of the kwargs, see PairwiseNN.

    Returns:
        A tuple of 2 functions: A init_fn that initializes the model parameters
        and an energy function that computes the energy for a particular state
        given model parameters.
    """

    if jnp.isscalar(hidden_layers):
        hidden_layers = [hidden_layers]
    hidden_layers.append(1)  # output layer is scalar energy

    @hk.without_apply_rng
    @hk.transform
    def model(position, neighbor, species=None, **dynamic_kwargs):
        n_particles, _ = neighbor.idx.shape
        if species is not None:
            smap._check_species_dtype(species)  # assert species are int
            raise NotImplementedError('Add species embedding to distance '
                                      'embedding.')

        dynamic_displacement = partial(displacement, **dynamic_kwargs)
        dyn_neighbor_displacement_fn = space.map_neighbor(dynamic_displacement)

        # compute pairwise distances
        neighbor_mask = neighbor.idx != n_particles
        r_neigh = position[neighbor.idx]
        pair_displacement = dyn_neighbor_displacement_fn(position, r_neigh)
        pair_distances = space.distance(pair_displacement)
        pair_distances = jnp.where(neighbor_mask, pair_distances,
                                   2. * r_cutoff)

        net = PairwiseNN(r_cutoff, hidden_layers, **pair_net_kwargs)
        per_pair_energy = net(pair_distances, species, **dynamic_kwargs)
        # pairs are counted twice
        pot_energy = util.high_precision_sum(per_pair_energy) / 2.
        return pot_energy

    return model.init, model.apply

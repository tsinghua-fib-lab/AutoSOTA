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

"""Functions to extract the sparse directional graph representation of a
molecular state.

The :class:`SparseDirectionalGraph` is the input to
:class:`~chemtrain.neural_networks.DimeNetPP`.
"""
import inspect
from typing import Optional, Callable, Tuple, TYPE_CHECKING

import chex
import numpy as onp
from jax import numpy as jnp, vmap, lax
from jax_md import space, partition, smap

from jax_md_mod import custom_partition, custom_space

if TYPE_CHECKING:
    from jax_md_mod.model.prior import Topology


@chex.dataclass
class SparseDirectionalGraph:
    """Sparse directial graph representation of a molecular state.

     Required arguments are necessary inputs for DimeNet++.
     If masks are not provided, all entities are assumed to be present.

     Attributes:
         distance_ij: A (N_edges,) array storing for each the radial distances
            between particle i and j
         idx_i: A (N_edges,) array storing for each edge particle index i
         idx_j: A (N_edges,) array storing for each edge particle index j
         angles: A (N_triplets,) array storing for each triplet the angle formed
            by the 3 particles
         reduce_to_ji: A (N_triplets,) array storing for each triplet kji edge
            index j->i to aggregate messages via a segment_sum: each m_ji is a
            distinct segment containing all incoming m_kj.
         expand_to_kj: A (N_triplets,) array storing for each triplet kji edge
            index k->j to gather all incoming edges for message passing.
         edge_mask: A (N_edges,) boolean array storing for each edge whether the
            edge exists. By default, all edges are considered.
         triplet_mask: A (N_triplets,) boolean array storing for each triplet
            whether the triplet exists. By default, all triplets are considered.
         n_edges: Number of non-masked edges in the graph. None assumes all
            edges are real.
         n_triplets: Number of non-masked triplets in the graph. None assumes
            all triplets are real.
         n_particles: Number of non-masked species in the graph.
    """
    species: jnp.ndarray
    distance_ij: jnp.ndarray
    idx_i: jnp.ndarray
    idx_j: jnp.ndarray
    angles: jnp.ndarray
    reduce_to_ji: jnp.ndarray
    expand_to_kj: jnp.ndarray
    species_mask: Optional[jnp.ndarray] = None
    edge_mask: Optional[jnp.ndarray] = None
    triplet_mask: Optional[jnp.ndarray] = None
    n_edges: Optional[int] = None
    n_triplets: Optional[int] = None

    def __post_init__(self):
        if self.species_mask is None:
            self.species_mask = jnp.ones_like(self.species, dtype=bool)
        if self.edge_mask is None:
            self.edge_mask = jnp.ones_like(self.distance_ij, dtype=bool)
        if self.triplet_mask is None:
            self.triplet_mask = jnp.ones_like(self.angles, dtype=bool)

    @property
    def n_particles(self):
        return jnp.sum(self.species_mask)

    def to_dict(self):
        """Returns the stored graph data as a dictionary of arrays.
        This format is often beneficial for dataloaders.
        """
        return {
            'species': self.species,
            'distance_ij': self.distance_ij,
            'idx_i': self.idx_i,
            'idx_j': self.idx_j,
            'angles': self.angles,
            'reduce_to_ji': self.reduce_to_ji,
            'expand_to_kj': self.expand_to_kj,
            'species_mask': self.species_mask,
            'edge_mask': self.edge_mask,
            'triplet_mask': self.triplet_mask
        }

    @classmethod
    def from_dict(cls, graph_dict):
        """Initializes instance from dictionary containing all necessary keys
        for initialization.
        """
        return cls(**{
            key: value for key, value in graph_dict.items()
            if key in inspect.signature(cls).parameters
        })

    def cap_exactly(self):
        """Deletes all non-existing edges and triplets from the stored graph.

        This is a non-pure function and hence not available in a jit-context.
        Returning the capped graph does not solve the problem when n_edges
        and n_triplets are computed within the jit-compiled function.
        """
        # edges are sorted, hence all non-existing edges are at the end
        self.species = self.species[:self.n_particles]
        self.species_mask = self.species_mask[:self.n_particles]

        self.distance_ij = self.distance_ij[:self.n_edges]
        self.idx_i = self.idx_i[:self.n_edges]
        self.idx_j = self.idx_j[:self.n_edges]
        self.edge_mask = self.edge_mask[:self.n_edges]

        self.angles = self.angles[:self.n_triplets]
        self.reduce_to_ji = self.reduce_to_ji[:self.n_triplets]
        self.expand_to_kj = self.expand_to_kj[:self.n_triplets]
        self.triplet_mask = self.triplet_mask[:self.n_triplets]


def angle(r_ij, r_kj, eps=1.e-8):
    """Computes the angle (kj, ij) from vectors r_kj and r_ij,
    correctly selecting the quadrant.

    Based on
    :math:`\\tan(\\theta) = |(r_{ji} \\times r_{kj})| / (r_{ji} \\cdot r_{kj})`.
    Beware the non-differentability of arctan2(0,0).

    Args:
        r_ij: Vector pointing to position of particle i from particle j
        r_kj: Vector pointing to position of particle k from particle j
        eps: If the vectors are exactly parallel, the gradient of the angle
            is undefined. Therefore, we set the angles to 0 or pi when the norm
            of the cross product is smaller than eps.

    Returns:
        Angle between vectors
    """
    safe_ji = jnp.array([1., 0., 0.], dtype=jnp.float32)
    safe_kj = jnp.array([0., 1., 0.], dtype=jnp.float32)

    dot = jnp.dot(r_ij, r_kj)
    cross = jnp.linalg.norm(jnp.cross(r_ij, r_kj))

    mask = (cross == 0.)
    r_ij = jnp.where(mask, safe_ji, r_ij)
    r_kj = jnp.where(mask, safe_kj, r_kj)

    safe_cross = jnp.linalg.norm(jnp.cross(r_ij, r_kj))
    theta = jnp.atan2(safe_cross, dot)

    return theta


def safe_angle_mask(r_ji, r_kj, angle_mask):
    """Sets masked angles to pi/2 to ensure differentiablility.

    Args:
        r_ji: Array (N_triplets, dim) of vectors  pointing to position of
            particle i from particle j
        r_kj: Array (N_triplets, dim) of vectors pointing to position of
            particle k from particle j
        angle_mask: (N_triplets, ) or (N_triplets, 1) Boolean mask for each
            triplet, which is False for triplets that need to be masked.

    Returns:
        A tuple (r_ji_safe, r_kj_safe) of vectors r_ji and r_kj, where masked
        triplets are replaced such that the angle between them is pi/2.
    """
    if angle_mask.ndim == 1:  # expand for broadcasing, if necessary
        angle_mask = jnp.expand_dims(angle_mask, -1)
    safe_ji = jnp.array([1., 0., 0.], dtype=jnp.float32)
    safe_kj = jnp.array([0., 1., 0.], dtype=jnp.float32)
    r_ji_safe = jnp.where(angle_mask, r_ji, safe_ji)
    r_kj_safe = jnp.where(angle_mask, r_kj, safe_kj)
    return r_ji_safe, r_kj_safe


def angle_triplets(positions, displacement_fn, angle_idxs, angle_mask):
    """Computes the angle for all triplets between 0 and pi.

     Masked angles are set to pi/2.

    Args:
        positions: Array pf particle positions (N_particles x 3)
        displacement_fn: Jax_md displacement function
        angle_idxs: Array of particle indeces that form a triplet
            (N_triples x 3)
        angle_mask: Boolean mask for each triplet, which is False for triplets
            that need to be masked.

    Returns:
        A (N_triples,) array with the angle for each triplet.
    """
    r_i = positions[angle_idxs[:, 0]]
    r_j = positions[angle_idxs[:, 1]]
    r_k = positions[angle_idxs[:, 2]]

    # Note: The original DimeNet implementation uses R_ji, however r_ij is the
    #       correct vector to get the angle between both vectors. This is a
    #       known issue in DimeNet. We apply the correct angle definition.
    r_ij = vmap(displacement_fn)(r_i, r_j)  # r_i - r_j respecting periodic BCs
    r_kj = vmap(displacement_fn)(r_k, r_j)
    # we need to mask as the case where r_ij is co-linear with r_kj.
    # Otherwise, this generates NaNs on the backward pass
    r_ij_safe, r_kj_safe = safe_angle_mask(r_ij, r_kj, angle_mask)
    angles = vmap(angle)(r_ij_safe, r_kj_safe)
    return angles


def _flatten_sort_and_capp(matrix, sorting_args, cap_size):
    """Helper function that takes a 2D array, flattens it, sorts it using the
    args (usually provided via argsort) and capps the end of the resulting
    vector. Used to delete non-existing edges and returns the capped vector.
    """
    vect = jnp.ravel(matrix)
    sorted_vect = vect[sorting_args]
    capped_vect = sorted_vect[0:cap_size]
    return capped_vect


def sparse_graph_from_neighborlist(displacement_fn: Callable,
                                   positions: jnp.ndarray,
                                   neighbor: partition.NeighborList,
                                   r_cutoff: jnp.array,
                                   species: jnp.array = None,
                                   max_edges: Optional[int] = None,
                                   max_triplets: Optional[int] = None,
                                   species_mask: jnp.array = None,
                                   ) -> Tuple[SparseDirectionalGraph, bool]:
    """Constructs a sparse representation of graph edges and angles to save
    memory and computations over neighbor list.

    The speed-up over simply using the dense jax_md neighbor list is
    significant, particularly regarding triplets. To allow for a representation
    of constant size required by jit, we pad the resulting vectors.

    Args:
        displacement_fn: Jax_MD displacement function encoding box dimensions
        positions: (N_particles, dim) array of particle positions
        neighbor: Jax_MD neighbor list that is in sync with positions
        r_cutoff: Radial cutoff distance, below which 2 particles are considered
            to be connected by an edge.
        species: (N_particles,) array encoding atom types. If None, assumes type
            0 for all atoms.
        max_edges: Maximum number of edges storable in the graph. Can be used to
            reduce the number of padded edges, but should be used carefully,
            such that no existing edges are capped. Default None uses the
            maximum possible number of edges as given by the dense neighbor
            list.
        max_triplets: Maximum number of triplets storable in the graph. Can be
            used to reduce the number of padded triplets, but should be used
            carefully, such that no existing triplets are capped. Default None
            uses the maximum possible number of triplets as given by the dense
            neighbor list.
        species_mask: (N_particles,) array encoding atom types. Default None,
            assumes no masking necessary.

    Returns:
        Tuple (sparse_graph, too_many_edges_error_code) containing the
        SparseDirectionalGraph and whether max_edges or max_triplets overflowed.
    """
    # TODO might be worth updating this function to the new sparse-style
    #  neighborlist in jax_md
    assert neighbor.format.name == 'Dense', ('Currently only dense neighbor'
                                             ' lists supported.')
    n_particles, max_neighbors = neighbor.idx.shape
    species = _canonicalize_species(species, n_particles)

    neighbor_displacement_fn = space.map_neighbor(displacement_fn)

    # compute pairwise distances
    pos_neigh = positions[neighbor.idx]
    pair_displacement = neighbor_displacement_fn(positions, pos_neigh)
    pair_distances = space.distance(pair_displacement)

    # compute adjacency matrix via neighbor_list, then build sparse graph
    # representation to avoid part of padding overhead in dense neighborlist
    # adds all edges > cut-off to masked edges
    edge_idx_ji = jnp.where(pair_distances < r_cutoff, neighbor.idx,
                            n_particles)
    # neighbor.idx: an index j in row i encodes a directed edge from
    #               particle j to particle i.
    # edge_idx[i, j]: j->i. if j == N: encodes masked edge.
    # Index N would index out-of-bounds, but in jax the last element is
    # returned instead

    # conservative estimates for initialization run
    # use guess from initialization for tighter bound to save memory and
    # computations during production runs
    if max_edges is None:
        max_edges = n_particles * max_neighbors
    max_edges = onp.min([max_edges, n_particles * max_neighbors])
    if max_triplets is None:
        max_triplets = max_edges * max_neighbors
    max_triplets = onp.min([max_triplets, max_edges * max_neighbors])

    # sparse edge representation:
    # construct vectors from adjacency matrix and only keep existing edges
    # Target node (i) and source (j) of edges
    pair_mask = edge_idx_ji != n_particles  # non-existing neighbor encoded as N
    # due to undirectedness, each edge is included twice
    n_edges = jnp.count_nonzero(pair_mask)
    pair_mask_flat = jnp.ravel(pair_mask)
    # non-existing edges are sorted to the end for capping
    sorting_idxs = jnp.argsort(~pair_mask_flat)
    _, yy = jnp.meshgrid(jnp.arange(max_neighbors), jnp.arange(n_particles))  # pylint: disable=unbalanced-tuple-unpacking
    idx_i = _flatten_sort_and_capp(yy, sorting_idxs, max_edges)
    idx_j = _flatten_sort_and_capp(edge_idx_ji, sorting_idxs, max_edges)
    d_ij = _flatten_sort_and_capp(pair_distances, sorting_idxs, max_edges)
    sparse_pair_mask = _flatten_sort_and_capp(pair_mask_flat, sorting_idxs,
                                              max_edges)

    # build sparse angle combinations from adjacency matrix:
    # angle defined for 3 particles with connections k->j and j->i
    # directional message passing accumulates all k->j to update each m_ji
    idx3_i = jnp.repeat(idx_i, max_neighbors)
    idx3_j = jnp.repeat(idx_j, max_neighbors)
    # retrieves for each j in idx_j its neighbors k: stored in 2nd axis
    idx3_k_mat = edge_idx_ji[idx_j]
    idx3_k = idx3_k_mat.ravel()
    angle_idxs = jnp.column_stack([idx3_i, idx3_j, idx3_k])

    # masking:
    # k and j are different particles, by edge_idx_ji construction.
    # The same applies to j - i, except for masked ones
    mask_i_eq_k = idx3_i != idx3_k
    # mask for ij known a priori
    mask_ij = jnp.repeat(sparse_pair_mask, max_neighbors)
    mask_k = idx3_k != n_particles
    angle_mask = mask_ij * mask_k * mask_i_eq_k  # union of masks
    angle_mask, sorting_idx3 = lax.top_k(angle_mask, max_triplets)
    angle_idxs = angle_idxs[sorting_idx3]
    n_triplets = jnp.count_nonzero(angle_mask)
    angles = angle_triplets(positions, displacement_fn, angle_idxs, angle_mask)

    # retrieving edge_id m_ji from nodes i and j:
    # idx_i < N by construction, but idx_j can be N: will override
    # lookup[i, N-1], which is problematic if [i, N-1] is an existing edge.
    # Hence, the lookup table is extended by 1.
    edge_id_lookup = jnp.zeros([n_particles, n_particles + 1], dtype=jnp.int32)
    edge_id_lookup_direct = edge_id_lookup.at[(idx_i, idx_j)].set(
        jnp.arange(max_edges))

    # stores for each angle kji edge index j->i to aggregate messages via a
    # segment_sum: each m_ji is a distinct segment containing all incoming m_kj
    reduce_to_ji = edge_id_lookup_direct[(angle_idxs[:, 0], angle_idxs[:, 1])]
    # stores for each angle kji edge index k->j to gather all incoming edges
    # for message passing
    expand_to_kj = edge_id_lookup_direct[(angle_idxs[:, 1], angle_idxs[:, 2])]

    too_many_edges_error_code = lax.cond(
        jnp.bitwise_or(n_edges > max_edges, n_triplets > max_triplets),
        lambda _: True, lambda _: False, n_edges
    )

    sparse_graph = SparseDirectionalGraph(
        species=species, distance_ij=d_ij, idx_i=idx_i, idx_j=idx_j,
        angles=angles, reduce_to_ji=reduce_to_ji, expand_to_kj=expand_to_kj,
        edge_mask=sparse_pair_mask, triplet_mask=angle_mask, n_edges=n_edges,
        n_triplets=n_triplets, species_mask=species_mask
    )
    return sparse_graph, too_many_edges_error_code


def _pad_graph(final_size, quantities, connectivities):
    """Helper function that returns padded edges or triplets, while
    differentiating between quantities (distances, angles) and adge / triplet
    connectivity.
    """
    # Everything can be padded with 0, because 0 corresponds to False
    # and the edge/triplet will hence have no effect
    padded_quantities, padded_connectivities = [], []
    for (quantity, connectivity) in zip(quantities, connectivities):
        pad_size = final_size - quantity.shape[0]
        connectivity_pad = jnp.zeros((pad_size, 3), dtype=jnp.int32)
        quantity_pad = jnp.zeros(pad_size, dtype=jnp.float32)
        padded_connectivities.append(jnp.vstack((connectivity,
                                                 connectivity_pad)))
        padded_quantities.append(jnp.concatenate((quantity, quantity_pad)))
    return padded_quantities, padded_connectivities


def convert_dataset_to_graphs(r_cutoff, position_data, box, species,
                              padding=True):
    """Converts input consisting of particle poistions and boxes to a dataset
    of sparse graph representations.

    Due to the high memory cost of saving padded graphs, this preprocessing
    step is only recomended for small datasets and only slightly changing
    number of particles per box.

    This function tackles the general case, where the number of particles and
    boxes vary across different snapshots, introducing some overhead if particle
    number and the box is fixed. Due to this general setting, this function is
    not jittable.

    Args:
        r_cutoff: Radial cut-off distance below which 2 particles form an edge
        position_data: Either a list of (N_particles, dim) arrays of particle
            positions in case N_particles is not constant accross snapshots or a
            (N_snapshots, N_particles, dim) array. The positions need to be
            given in real (non-fractional) coordinates.
        box: Either a single 1 or 2-dimensional box (if the box is constant
            across snapshots) or an (N_snapshots, dim) or
            (N_snapshots, dim, dim) array of boxes.
        species: Either a list of (N_particles,) arrays of atom types in case
            N_particles is not constant accross snapshots or a single
            (N_particles,) array.
        padding: If True, pads resulting edges and triplets to the maximum
            across the input data to allow for straightforward batching
            without re-compilation. If False, returns edges and triplets with
            varying shapes, but to-be-masked non-existing edges / triplets.

    Returns:
        With padding, a SparseDirectionalGraph pytree containing all graphs of
        the dataset, stacked along axis 0. Without padding, a dictionary
        containing the whole definitions of the sparse molecular graph, given
        as Lists. Refer to :class:`SparseDirectionalGraph` for respective
        definitions.
    """
    # canonicalize inputs to lists
    if not isinstance(position_data, list):
        n_snapshots = position_data.shape[0]
        position_data = [position_data[i] for i in range(n_snapshots)]
    else:
        n_snapshots = len(position_data)
    if box.shape[0] == n_snapshots:  # array of boxes
        box = [box[i] for i in range(n_snapshots)]
    else:  # a single box
        box = [box for _ in range(n_snapshots)]
    if not isinstance(species, list):
        species = [species for _ in range(n_snapshots)]

    max_edges = 0
    max_triplets = 0
    dists, angles, edges, triplets = [], [], [], []

    for (positions, cur_box) in zip(position_data, box):
        box_tensor, scale_fn = custom_space.init_fractional_coordinates(cur_box)
        displacement_fn, _ = space.periodic_general(box_tensor)
        positions = scale_fn(positions)  # to fractional coordinates
        neighbor_fn = partition.neighbor_list(  # only required for 1 state
            displacement_fn, box_tensor, r_cutoff, dr_threshold=0.01,
            capacity_multiplier=1.01, fractional_coordinates=True
        )
        nbrs = neighbor_fn.allocate(positions)  # pylint: disable=not-callable
        graph, _ = sparse_graph_from_neighborlist(displacement_fn, positions,
                                                  nbrs, r_cutoff)
        graph.cap_exactly()

        max_edges = max(max_edges, graph.n_edges)
        max_triplets = max(max_triplets, graph.n_triplets)

        # build arrays for edges and angles. Needs to be stored in lists due
        # to different edge and angle count across snapshots in general
        # Boolean mask arrays are converted to int32 1.
        dists.append(graph.distance_ij)
        angles.append(graph.angles)
        edges.append(jnp.stack((graph.idx_i, graph.idx_j, graph.edge_mask),
                               axis=-1))
        triplets.append(jnp.stack((graph.reduce_to_ji, graph.expand_to_kj,
                                   graph.triplet_mask), axis=-1))

    if padding:
        dists, edges = _pad_graph(max_edges, dists, edges)
        angles, triplets = _pad_graph(max_triplets, angles, triplets)
        species, species_mask = pad_per_atom_quantities(species)
    else:
        species_mask = [jnp.ones_like(species_arr) for species_arr in species]

    # save in dict for better transparency
    graph_rep = {
        'species': species,
        'distance_ij': dists,
        'idx_i': [edge[:, 0] for edge in edges],
        'idx_j': [edge[:, 1] for edge in edges],
        'angles': angles,
        'reduce_to_ji': [triplet[:, 0] for triplet in triplets],
        'expand_to_kj': [triplet[:, 1] for triplet in triplets],
        'species_mask': species_mask,
        'edge_mask': [onp.array(edge[:, 2], dtype=bool) for edge in edges],
        'triplet_mask': [onp.array(triplet[:, 2], dtype=bool)
                         for triplet in triplets]
    }

    if padding:  # when padded, we can return arrays instead of lists
        graph_rep = {key: onp.array(value) for key, value in graph_rep.items()}
        graph_rep = SparseDirectionalGraph(**graph_rep)

    return graph_rep


def pad_per_atom_quantities(per_atom_data):
    """Pads list arrays containing per-atom quantities (e.g. species,
    partial charges, ...).

    Allows for straightforward batching without re-compilations in case of
    non-constant number of particles across snapshots.

    Args:
        per_atom_data: List of (N_particles,) arrays containing a scalar
            quantity of each particle.

    Returns:
        A (N_snapshots, N_particles) array and corresponding mask array.
    """
    max_particles = max([species.size for species in per_atom_data])
    n_snapshots = len(per_atom_data)
    padded_quantity = onp.zeros((n_snapshots, max_particles),
                                dtype=per_atom_data[0].dtype)
    quantity_mask = onp.zeros((n_snapshots, max_particles), dtype=bool)
    for i, quantity in enumerate(per_atom_data):
        padded_quantity[i, :quantity.size] = quantity
        quantity_mask[i, :quantity.size] = True
    return padded_quantity, quantity_mask


def _canonicalize_species(species, n_particles):
    """Ensures species are integer and initializes species to 0 if species=None.

    Args:
        species: (N_particles,) array of atom types or None
        n_particles: Number of particles

    Returns:
        Integer species array.
    """
    if species is None:
        species = jnp.zeros(n_particles, dtype=jnp.int32)
    else:
        smap._check_species_dtype(species)  # assert species are int
    return species


def subtract_topology_from_neighbor_list(nbrs: partition.NeighborList,
                                         topology: "Topology",
                                         bond_mask = None,
                                         angle_mask = None,
                                         dihedral_mask = None):
    """Removes all neighbors connected via bonds and angles from the neighbor list. """

    bond_idx, _, bond_mask_2 = topology.get_bonds()
    if bond_mask is not None:
        bond_mask = jnp.logical_and(bond_mask, bond_mask_2)
    else:
        bond_mask = bond_mask_2

    angle_idx, _, angle_mask_2 = topology.get_angles()
    if angle_mask is not None:
        angle_mask = jnp.logical_and(angle_mask, angle_mask_2)
    else:
        angle_mask = angle_mask_2

    dihedral_idx, _, dihedral_mask_2 = topology.get_dihedrals()
    # The dihedral mask might be vectorized. It sufficies, if a single
    # dihedral term for that dihedral is nonzero
    dihedral_mask_2 = jnp.any(dihedral_mask_2, axis=-1)
    if dihedral_mask is not None:
        dihedral_mask = jnp.logical_and(dihedral_mask, dihedral_mask_2)
    else:
        dihedral_mask = dihedral_mask_2

    if bond_mask is not None:
        nbrs = custom_partition.exclude_from_neighbor_list(
            nbrs, bond_idx[:, (0, 1)], bond_mask)
    if angle_mask is not None:
        nbrs = custom_partition.exclude_from_neighbor_list(
            nbrs, angle_idx[:, (0, 2)], angle_mask)
    if dihedral_mask is not None:
        nbrs = custom_partition.exclude_from_neighbor_list(
            nbrs, dihedral_idx[:, (0, 3)], dihedral_mask)

    return nbrs

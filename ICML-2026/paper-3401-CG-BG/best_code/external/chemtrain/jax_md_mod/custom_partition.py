# Copyright 2019 Google LLC
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

"""Custom functions to analyze the neighborlist graph."""
import importlib
import warnings
from typing import Union

import functools

import jax
import jax.numpy as jnp
from jax import Array, export

import numpy as onp

from jax_md import partition, space
from jax_md_mod.model import sparse_graph


def mask_dense(idx, mask=None):
    # Mask out edges to self
    self_mask = (idx == jnp.arange(idx.shape[0])[:, jnp.newaxis])

    # Only mask edges to self
    if mask is None:
        return jnp.where(self_mask, idx.shape[0], idx)

    # Mask out all senders
    sender_mask = jnp.logical_or(
        jnp.logical_not(mask)[:, jnp.newaxis], self_mask
    )

    # Mask out all receivers
    total_mask = jnp.logical_or(
        sender_mask, jnp.logical_not(mask[idx])
    )

    return jnp.where(total_mask, idx.shape[0], idx)


def partition_neighbor_list(nbrs: partition.NeighborList,
                            partitions: Array = None,
                            max_capacity: int = None) -> partition.NeighborList:
    """Removes edges between particles of different partitions.

    Args:
        nbrs: Dense or sparse neighbor list.
        partitions: Array containing the partition indices of each particle.
        max_capacity: Maximum capacity of the masked neighbor list.
            If not specified, the capacity of the original neighbor list is used.

    Returns:
        Returns a neighbor list without edges to invalid particles.

    """

    def mask_sparse(idx):
        # Mask out all invalid edges
        senders, receivers = idx

        edge_mask = partitions[senders] != partitions[receivers]

        return jnp.where(edge_mask[jnp.newaxis, :], nbrs.reference_position.shape[0], idx)

    if nbrs.format == partition.NeighborListFormat.Dense:
        raise NotImplementedError(
            "Partitioning is not yet implemented for dense neighbor lists."
        )

    else:
        new_idx = mask_sparse(nbrs.idx)
        if max_capacity is not None:
            invalid_idx = nbrs.reference_position.shape[0]
            valid = jnp.logical_and(
                new_idx[0, :] < invalid_idx,
                new_idx[1, :] < invalid_idx
            )
            _, select = lax.top_k(valid, k=max_capacity)
            new_idx = jnp.take(new_idx, select, axis=1)

    return nbrs.set(
        idx=new_idx, max_occupancy=max_capacity
        if max_capacity is not None else nbrs.max_occupancy,
    )


def mask_neighbor_list(nbrs: partition.NeighborList,
                       mask: Array = None) -> partition.NeighborList:
    """Masks the neighbor list indices.

    Args:
        nbrs: Dense or sparse neighbor list.
        mask: Boolean array masking valid particles (True). Edges from and to
            invalid particles (False) are removed from the neighbor list.

    Returns:
        Returns a neighbor list without edges to invalid particles.

    """

    def mask_sparse(idx, mask):
        # Mask out all invalid edges
        senders, receivers = idx

        edge_mask = jnp.logical_or(
            jnp.logical_not(mask[senders]),
            jnp.logical_not(mask[receivers])
        )

        return jnp.where(edge_mask[jnp.newaxis, :], nbrs.reference_position.shape[0], idx)

    if nbrs.format == partition.NeighborListFormat.Dense:
        new_idx = mask_dense(nbrs.idx, mask)
    else:
        new_idx = mask_sparse(nbrs.idx, mask)

    new_position = jnp.where(mask[:, jnp.newaxis], nbrs.reference_position, 0.0)

    return nbrs.set(idx=new_idx, reference_position=new_position)


def masked_neighbor_list(displacement_or_metric,
                         r_cutoff: float,
                         dr_threshold: Union[float, None] = None,
                         capacity_multiplier: float = 1.25,
                         format = partition.NeighborListFormat.Dense,
                         ) -> partition.NeighborFn:
    """Returns a function that builds a list neighbors for collections of points.

    Adapts the JAX, M.D. neighbor list :func:`jax_md.partition.neighbor_list`
    to allow for masking and to enforce rebuilding of the neighbor list.

    Args:
      displacement_or_metric: A function `d(R_a, R_b)` that computes the displacement
          between pairs of points.
      r_cutoff: A scalar specifying the neighborhood radius.
      dr_threshold: A scalar specifying the maximum distance particles can move
          before rebuilding the neighbor list. If specified to None, the neighbor
          list will always be rebuilt.
      capacity_multiplier: A floating point scalar specifying the fractional
          increase in maximum neighborhood occupancy we allocate compared with the
          maximum in the example positions.
      format: The format of the neighbor list; see the :meth:`NeighborListFormat` enum
        for details about the different choices for formats. Defaults to `Dense`.

    Returns:
        A NeighborListFns object that contains a method to allocate a new neighbor
        list and a method to update an existing neighbor list.
    """
    always_recompute = dr_threshold is None
    if always_recompute:
        dr_threshold = 0.0

    partition.is_format_valid(format)
    r_cutoff = jax.lax.stop_gradient(r_cutoff)
    dr_threshold = jax.lax.stop_gradient(dr_threshold)

    cutoff = r_cutoff + dr_threshold
    cutoff_sq = cutoff ** 2
    threshold_sq = (dr_threshold / partition.f32(2)) ** 2
    metric_sq = partition._displacement_or_metric_to_metric_sq(displacement_or_metric)

    @functools.partial(jax.jit, static_argnums=0)
    def candidate_fn(positionShape) -> Array:
        candidates = jnp.arange(positionShape[0])
        return jnp.broadcast_to(candidates[None, :],
                                (positionShape[0], positionShape[0]))

    @jax.jit
    def mask_self_fn(idx: Array) -> Array:
        self_mask = idx == jnp.reshape(jnp.arange(idx.shape[0], dtype=partition.i32),
                                       (idx.shape[0], 1))
        return jnp.where(self_mask, idx.shape[0], idx)

    @jax.jit
    def prune_neighbor_list_dense(position: Array, idx: Array, **kwargs
                                  ) -> Array:
        d = functools.partial(metric_sq, **kwargs)
        d = space.map_neighbor(d)

        N = position.shape[0]
        neigh_position = position[idx]
        dR = d(position, neigh_position)

        mask = (dR < cutoff_sq) & (idx < N)
        out_idx = N * jnp.ones(idx.shape, partition.i32)

        cumsum = jnp.cumsum(mask, axis=1)
        index = jnp.where(mask, cumsum - 1, idx.shape[1] - 1)
        p_index = jnp.arange(idx.shape[0])[:, None]
        out_idx = out_idx.at[p_index, index].set(idx)
        max_occupancy = jnp.max(cumsum[:, -1])

        return out_idx, max_occupancy

    @jax.jit
    def prune_neighbor_list_sparse(position: Array, idx: Array, **kwargs
                                   ) -> Array:
        d = functools.partial(metric_sq, **kwargs)
        d = space.map_bond(d)

        N = position.shape[0]
        sender_idx = jnp.broadcast_to(jnp.arange(N)[:, None], idx.shape)

        sender_idx = jnp.reshape(sender_idx, (-1,))
        receiver_idx = jnp.reshape(idx, (-1,))
        dR = d(position[sender_idx], position[receiver_idx])

        mask = (dR < cutoff_sq) & (receiver_idx < N)
        if format is partition.NeighborListFormat.OrderedSparse:
            mask = mask & (receiver_idx < sender_idx)

        out_idx = N * jnp.ones(receiver_idx.shape, partition.i32)

        cumsum = jnp.cumsum(mask)
        index = jnp.where(mask, cumsum - 1, len(receiver_idx) - 1)
        receiver_idx = out_idx.at[index].set(receiver_idx)
        sender_idx = out_idx.at[index].set(sender_idx)
        max_occupancy = cumsum[-1]

        return jnp.stack((receiver_idx, sender_idx)), max_occupancy

    def neighbor_list_fn(position: Array,
                         neighbors = None,
                         extra_capacity: int = 0,
                         **kwargs) -> partition.NeighborList:
        N = position.shape[0]
        mask = kwargs.get("mask", jnp.ones(N, dtype=jnp.bool_))
        position = jnp.where(mask[:, jnp.newaxis], position, jnp.inf)
        def neighbor_fn(position_and_error, max_occupancy=None):
            position, err = position_and_error
            idx = candidate_fn(position.shape)
            idx = mask_dense(idx, mask=mask)

            if partition.is_sparse(format):
                idx, occupancy = prune_neighbor_list_sparse(position, idx, **kwargs)
            else:
                idx, occupancy = prune_neighbor_list_dense(position, idx, **kwargs)

            if max_occupancy is None:
                _extra_capacity = (extra_capacity if not partition.is_sparse(format)
                                   else N * extra_capacity)
                max_occupancy = int(occupancy * capacity_multiplier + _extra_capacity)
                if max_occupancy > idx.shape[-1]:
                    max_occupancy = idx.shape[-1]
                if not partition.is_sparse(format):
                    capacity_limit = N - 1
                elif format is partition.NeighborListFormat.Sparse:
                    capacity_limit = N * (N - 1)
                else:
                    capacity_limit = N * (N - 1) // 2
                if max_occupancy > capacity_limit:
                    max_occupancy = capacity_limit
            idx = idx[:, :max_occupancy]
            update_fn = (neighbor_list_fn if neighbors is None else
                         neighbors.update_fn)
            return partition.NeighborList(
                idx,
                position,
                err.update(partition.PEC.NEIGHBOR_LIST_OVERFLOW, occupancy > max_occupancy),
                None,
                max_occupancy,
                format,
                None,
                None,
                update_fn)  # pytype: disable=wrong-arg-count

        nbrs = neighbors
        if nbrs is None:
            return neighbor_fn((position, partition.PartitionError(jnp.zeros((), jnp.uint8))))

        neighbor_fn = functools.partial(neighbor_fn, max_occupancy=nbrs.max_occupancy)

        d = functools.partial(metric_sq, **kwargs)
        d = jax.vmap(d)

        if always_recompute:
            print(f"Always recompute the neighbor list!")
            return neighbor_fn((position, nbrs.error))
        else:
            return jax.lax.cond(
                jnp.logical_or(
                    jnp.any(d(position, nbrs.reference_position) > threshold_sq),
                    jnp.any(jnp.logical_xor(position == jnp.inf, nbrs.reference_position == jnp.inf))
                ),
                (position, nbrs.error), neighbor_fn,
                nbrs, lambda x: x)

    def allocate_fn(position: Array, extra_capacity: int = 0, **kwargs
                    ):
        return neighbor_list_fn(position, extra_capacity=extra_capacity, **kwargs)

    def update_fn(position: Array, neighbors, **kwargs
                  ):
        return neighbor_list_fn(position, neighbors, **kwargs)

    return partition.NeighborListFns(allocate_fn, update_fn)  # pytype: disable=wrong-arg-count


def exclude_from_neighbor_list(neighbor: partition.NeighborList,
                               exclude_idx,
                               exclude_mask) -> partition.NeighborList:
    """Function excluding edges from the neighbor list.

    Args:
        neighbor: Neighbor list
        exclude_idx: Indices of the edges that should be excluded, if contained.
        exclude_mask: Boolean array whether specified edges should be excluded.

    Example:

        >>> from pathlib import Path
        >>> root = Path.cwd().parent

        >>> import mdtraj
        >>> from jax_md import space
        >>> from jax import numpy as jnp
        >>> from jax_md_mod.custom_partition import masked_neighbor_list

        >>> pdb = mdtraj.load(root / "examples/data/ethane.pdb")
        >>> r_init = jnp.asarray(pdb.xyz[0], dtype=jnp.float32)
        >>> box = jnp.array(1.0)

        >>> displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=True)
        >>> neighbor_fns = masked_neighbor_list(
        ...     displacement_fn, r_cutoff=1.0, dr_threshold=0.05,
        ... )

        We can now exclude, e.g., the first C atom from the neighbor list

        >>> mask = jnp.array([0, 1, 1, 1, 1, 1, 1, 1])
        >>> nbrs_init = neighbor_fns.allocate(r_init, mask=mask)
        >>>
        >>> print(nbrs_init.idx)
        [[8 8 8 8 8 8 8]
         [2 3 4 5 6 7 8]
         [1 3 4 5 6 7 8]
         [1 2 4 5 6 7 8]
         [1 2 3 5 6 7 8]
         [1 2 3 4 6 7 8]
         [1 2 3 4 5 7 8]
         [1 2 3 4 5 6 8]]


        Whenever the neighbor list must be recomputed (dR threshold), a new
        mask is applied

        >>> mask = jnp.array([1, 0, 1, 1, 1, 1, 1, 1])
        >>> print(neighbor_fns.update(r_init, nbrs_init, mask=mask).idx)
        [[2 3 4 5 6 7 8]
         [8 8 8 8 8 8 8]
         [0 3 4 5 6 7 8]
         [0 2 4 5 6 7 8]
         [0 2 3 5 6 7 8]
         [0 2 3 4 6 7 8]
         [0 2 3 4 5 7 8]
         [0 2 3 4 5 6 8]]


    Returns:
        Returns a neighbor list of the same format with excluded edges.

    """

    @functools.partial(jax.vmap, in_axes=(0, 0, None, None, None, None))
    def _exclude(particle_nbrs, idx, invalid, mask, ref_i, ref_j):
        # Check for all neighbors whether they are already part of the bond
        # list.
        exclude = jax.vmap(
            # Map over all pairs i, j that are part of the neighbor list
            lambda i: jnp.any(
                # Check whether these pairs appear in the bond list
                jnp.logical_and(
                    mask,
                    jnp.logical_or(
                        jnp.logical_and(i == ref_i, idx == ref_j),
                        jnp.logical_and(i == ref_j, idx == ref_i)
                    )
                )
            )
        )(particle_nbrs)
        return jnp.where(exclude, invalid, particle_nbrs)

    @functools.partial(jax.vmap, in_axes=(1, None, None, None, None),
                       out_axes=1)
    def _exclude_sparse(nbr_idx, invalid, ref_i, ref_j, mask):
        # Map through the entries in the neighbor list and exclude them if
        # they are part of the bond list
        exclude = jnp.logical_or(
            jnp.logical_and(nbr_idx[0] == ref_i, nbr_idx[1] == ref_j),
            jnp.logical_and(nbr_idx[0] == ref_j, nbr_idx[1] == ref_i)
        )
        # Mask out invalid bonds or angles
        exclude = jnp.any(jnp.logical_and(exclude, mask))
        return jnp.where(exclude, invalid, nbr_idx)

    # Call the respective function depending on the format of the neighbor list

    invalid_idx = neighbor.idx.shape[0]
    if neighbor.format == partition.NeighborListFormat.Dense:
        new_idx = _exclude(
            neighbor.idx, jnp.arange(invalid_idx), invalid_idx, exclude_mask,
            exclude_idx[:, 0], exclude_idx[:, 1]
        )
    else:
        new_idx = _exclude_sparse(
            neighbor.idx, invalid_idx, exclude_idx[:, 0], exclude_idx[:, 1],
            exclude_mask
        )

    return neighbor.set(idx=new_idx)

def get_triplet_indices(neighbor: partition.NeighborList):
    """Returns indices for all triplets of the neighbor list."""

    @functools.partial(jax.vmap, in_axes=(None, 0), out_axes=-1)
    def _get_triplets(idx, j):
        max_nbrs = idx.shape[1]

        # Return all bonds idx to j
        to_j = idx[j, :]

        # All permutations (remove diagonal entries)
        diagonal_mask = onp.mod(onp.arange(max_nbrs ** 2), max_nbrs + 1) != 0
        ik_to_j = jnp.stack(jnp.meshgrid(to_j, to_j, indexing='ij'), axis=0)
        ik_to_j = ik_to_j.reshape((2, -1))[:, diagonal_mask]

        # Add the reference to the center atom
        ij = ik_to_j.at[1, :].set(j)
        jk = ik_to_j.at[0, :].set(j)

        return ij, jk

    if neighbor.format == neighbor.format.Dense:
        invalid_idx = neighbor.idx.shape[0]
        ij, jk = _get_triplets(neighbor.idx, jnp.arange(neighbor.idx.shape[0]))
        ij = ij.reshape((2, -1)).swapaxes(0, 1)
        jk = jk.reshape((2, -1)).swapaxes(0, 1)

        # Mask out all invalid triplets
        mask = jax.vmap(jnp.logical_and)(
            jnp.all(ij != invalid_idx, axis=-1),
            jnp.all(jk != invalid_idx, axis=-1),
        )

        # Sort (simpler to later prune the triplet array)
        order = jnp.argsort(-1.0 * mask)
        ij = ij[order, :]
        jk = jnp.flip(jk, axis=-1)[order, :]
        mask = mask[order]

        return ij, jk, mask
    else:
        raise NotImplementedError(
            f"Neighbor list format {neighbor.format} not yet supported."
        )


def check_connectivity(neighbor: partition.NeighborList, mask=None):
    """Check the connectivity of the neighbor list.

    Args:
        neighbor: Neighbor list

    Returns:
        Returns True if a connection between any nodes exists.

    """
    if mask is None:
        mask = jnp.ones(neighbor.reference_position.shape[0], dtype=bool)

    def _update_connectivity(state):
        reachable, idx = state

        if neighbor.format == partition.NeighborListFormat.Dense:
            pass
        elif neighbor.format == partition.NeighborListFormat.Sparse:
            senders, receivers = neighbor.idx

            # Propagate reachable state from senders to receivers
            reachable = jax.ops.segment_sum(
                jnp.int_(reachable[senders]), receivers, reachable.size)
            reachable = jnp.logical_and(reachable > 0, mask)
        else:
            raise NotImplementedError(
                f"Neighbor list format {neighbor.format} not yet supported."
            )

        return reachable, idx + 1

    def _search(state):
        reachable, idx = state
        # We stop when one of the following conditions is met:
        # 1. Iterations equal to the number of actual particles. Worst case
        #    scenario when graph is line
        # 2. All valid particles are reachable
        return jnp.logical_and(idx < jnp.sum(mask), jnp.sum(reachable) < jnp.sum(mask))

    # Find one non-masked particle and start the search from there
    first_nonzero = jnp.argmax(mask)
    reachable = jnp.logical_and(mask, jnp.arange(mask.size) == first_nonzero)

    reachable, _ = jax.lax.while_loop(
        _search, _update_connectivity, (reachable, 0)
    )

    return jnp.sum(reachable) >= jnp.sum(mask)


def find_clusters(neighbor: partition.NeighborList, mask=None):
    """Discovers separate subgraphs in the neighbor list.

    Args:
        neighbor: Neighbor list
        mask: Mask indicating whether particles are real or padded

    Returns:
        Returns a vector with unique cluster-ids to which a particle belongs
        to and the number of discovered separate subgraphs.

    """
    if mask is None:
        mask = jnp.ones(neighbor.reference_position.shape[0], dtype=bool)

    def _update_connectivity(clusters, _):
        # Particles propagate their cluster information
        if neighbor.format == partition.NeighborListFormat.Dense:
            pass
        elif neighbor.format == partition.NeighborListFormat.Sparse:
            senders, receivers = neighbor.idx

            # Propagate cluster state from senders to receivers
            clusters = jax.ops.segment_min(
                jnp.int_(clusters[senders]), receivers, clusters.size)
        else:
            raise NotImplementedError(
                f"Neighbor list format {neighbor.format} not yet supported."
            )

        return clusters, jnp.sum(jnp.diff(jnp.sort(clusters) * mask) > 0) + 1

    # Each valid particle gets its own cluster in the beginning
    clusters = jnp.where(mask, jnp.arange(mask.size), mask.size)
    clusters -= jnp.min(clusters) # Start the cluster counter with 0
    _, nclusters = jax.lax.scan(_update_connectivity, clusters, jnp.arange(clusters.size))

    return clusters, nclusters[-1]


def to_networkx(neighbor: partition.NeighborList):
    nx = importlib.import_module('networkx')

    graph = nx.Graph()

    if neighbor.format == partition.NeighborListFormat.Dense:
        num_particles, max_neighbors = neighbor.idx.shape
        for i in range(num_particles):
            for j in range(max_neighbors):
                if neighbor.idx[i, j] == num_particles: continue

                graph.add_edge(int(i), int(neighbor.idx[i, j]))
    else:
        for i, j in neighbor.idx.T:
            if i == neighbor.reference_position.shape[0]: continue
            if j == neighbor.reference_position.shape[0]: continue

            graph.add_edge(int(i), int(j))

    return graph


def test_graph_statistics(displacement_fn: space.DisplacementFn,
                          position: Array,
                          neighbor: partition.NeighborList,
                          r_cutoff: Union[float, Array],
                          max_edge_multiplier: float = 1.5,
                          ):
    """Computes neighbor list statistics for test conformation.

    Args:
        displacement_fn: Displacement function
        position: Particle position for a representative configuration
        neighbor: Neighbor list for a representative configuration
        r_cutoff: Cutoff radius of edges.
        max_edge_multiplier: Multiplier to increase maximum number of edges
            based on edges found in representative configuration.

    Returns:
        Returns a tuple of average number of neighbors and maximum number of
        edges.

    """

    # Checking only necessary if neighbor list is dense
    if neighbor.format == partition.Dense:
        print('Capping edges and triplets. Beware of overflow, which is '
              'currently not being detected.')

        testgraph, _ = sparse_graph.sparse_graph_from_neighborlist(
            displacement_fn, position, neighbor, r_cutoff)
        max_edges = jnp.int32(jnp.ceil(testgraph.n_edges * max_edge_multiplier))

        # cap maximum edges and angles to avoid overflow from multiplier
        n_particles, n_neighbors = neighbor.idx.shape
        max_edges = min(max_edges, n_particles * n_neighbors)

        print(f"Estimated max. {max_edges} edges.")

        avg_num_neighbors = testgraph.n_edges / n_particles
    else:
        n_particles = neighbor.reference_position.shape[0]
        max_edges = neighbor.idx.shape[0]
        avg_num_neighbors = onp.sum(neighbor.idx[0] < n_particles)
        avg_num_neighbors /= n_particles

    return avg_num_neighbors, max_edges


def readout_vectors(displacement_fn: space.DisplacementFn,
                    r_cutoff: Union[float, Array],
                    position: Array,
                    neighbor: partition.NeighborList,
                    species: Array = None,
                    mask: Array = None,
                    max_edges = None,
                    edges_per_particle: None | float = None,
                    **kwargs
                    ):
    """Computes neighbor list statistics for test conformation.

    Args:
        displacement_fn: Displacement function
        r_cutoff: Cutoff radius of edges.
        position: Particle position for a representative configuration
        neighbor: Neighbor list for a representative configuration
        species: Species of atoms.
        mask: Mask indicating whether particles are real or padded.
        max_edges: Maximum number of edges to consider.
        edges_per_particle: Limit the number of edges to a value proportional
            to the number of particles.
        kwargs: Keyword arguments passed to the displacement function.

    Returns:
        Returns the extracted vectors, senders, and receivers.

    """
    dyn_displacement = functools.partial(displacement_fn, **kwargs)

    if edges_per_particle is not None:
        if max_edges is not None:
            raise ValueError(
                "Either edges_per_particle or max_edges can be specified, not both."
            )
        
        # Restrict to two digits after the decimal point. This step is required
        # because JAX symbolic dimensions support only integer operations.
        factor = int(edges_per_particle * 1000)
        gcd = onp.gcd(factor, 1000)
        max_edges = (factor // gcd) * position.shape[0] // (1000 // gcd)

        print(f"Limit the maximum number of edges to {max_edges} "
              f"({factor // gcd} / {1000 // gcd} edges per particle).")


    if species is None:
        species = jnp.ones(position.shape[0], dtype=jnp.int32)
    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=jnp.bool_)

    if max_edges is not None and export.is_symbolic_dim(position.shape[0]):
        if not export.is_symbolic_dim(max_edges):
            raise TypeError(
            "max_edges must be symbolic if used in export."
        )

    if neighbor.format == partition.Dense:
        graph, _ = sparse_graph.sparse_graph_from_neighborlist(
            dyn_displacement, position, neighbor, r_cutoff,
            species, max_edges=max_edges, species_mask=mask
        )
        senders = graph.idx_i
        receivers = graph.idx_j
    else:
        assert neighbor.idx.shape == (
        2, neighbor.idx.shape[1]), "Neighbor list has wrong shape."
        senders, receivers = neighbor.idx

    # Set invalid edges to the cutoff to avoid numerical issues
    vectors = jax.vmap(dyn_displacement)(position[senders], position[receivers])
    vectors = jnp.where(
        jnp.logical_and(
            jnp.logical_and(senders < position.shape[0], mask[senders]),
            jnp.logical_and(receivers < position.shape[0], mask[receivers])
        )[:, jnp.newaxis], vectors, r_cutoff / jnp.sqrt(3))

    if max_edges is not None:
        # Sort vectors by length and remove up to max_edges edges
        lengths = jnp.linalg.norm(vectors, axis=-1)
        sort_idx = jnp.argsort(lengths)
        vectors = vectors[sort_idx][:max_edges]
        senders = senders[sort_idx][:max_edges]
        receivers = receivers[sort_idx][:max_edges]

    return vectors, senders, receivers

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

"""Graphs for exporting potential and force models."""

import abc
import functools
import typing
from itertools import product

import numpy as onp

import jax
from jax import export, numpy as jnp, lax

import jax_md_mod
from jax_md import partition, dataclasses, smap, space

from typing import NamedTuple, Tuple

from . import util
from ._protobuf import model_pb2 as model_proto


# Does not have to be typed
ListStatistics = typing.Dict


@dataclasses.dataclass
class NeighborList(metaclass=abc.ABCMeta):
    """Abstract class for neighbor list graphs."""

    @staticmethod
    @abc.abstractmethod
    def set_properties(proto: model_proto.Model):
        """Assigns the graph type to the protobuf message."""
        pass

    @staticmethod
    @util.define_symbols("")
    @abc.abstractmethod
    def create_symbolic_input_format(*args, **kwargs):
        """Creates a symbolic representation of the graph.

        Args:
            max_atoms: The maximum number of atoms, including ghost atoms and
                padding atoms.
            scope: The scope to add more symbolic variables.

        The variables should begin with "graph_".

        Returns:
            Returns a symbolic representation of the graph.

        """

    @staticmethod
    def create_from_args(displacement_fn,
                         r_cutoff,
                         num_mpl,
                         position,
                         species,
                         ghost_mask,
                         valid_mask,
                         newton,
                         *args,
                         half=True
                         ) -> Tuple["NeighborList", "ListStatistics"]:
        """Creates the neighbor list from inputs to the exported function."""


@dataclasses.dataclass
class SimpleSparseNeighborList(NeighborList):
    """Simple neighbor list representation using precomputed neighbor list.

    This neighbor list is a sparse representation of a graph.
    It does not infer the neighbor list from the positions but acts as an
    interface between the precomputed neighbor list, e.g., from LAMMPS, and
    the exported model.

    Nevertheless, this class increases the efficiency of the exported model
    while reducing necessary data transfer by pruning the neighbor list.
    Therefore, the class filters out all edges that are longer than the
    specified model cutoff distance. Moreover, it prunes all edges between
    ghost atoms (atoms not in the local domain) that are not relevant for
    a correct force computation.

    Attributes:
        senders: The sender indices of the edges.
        receivers: The receiver indices of the edges.
        max_edges: The maximum number of relevant edges in the neighbor list.

    """

    senders: jax.Array
    receivers: jax.Array

    max_edges: jax.Array

    @staticmethod
    def set_properties(proto: model_proto.Model):
        proto.neighbor_list.type = model_proto.Model.NeighborListType.SIMPLE_SPARSE
        proto.neighbor_list.half_list = True

    @staticmethod
    @util.define_symbols(
        "max_buffers, max_edges",
        ["max_edges <= 2 * max_buffers"]
    )
    def create_symbolic_input_format(max_buffers, max_edges, **kwargs):

        senders = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        receivers = jax.ShapeDtypeStruct((max_buffers,), jnp.int32)
        buffer = jax.ShapeDtypeStruct((max_edges,), jnp.bool_)

        return senders, receivers, buffer

    @staticmethod
    def create_from_args(r_cutoff,
                         nbr_order,
                         position,
                         species,
                         ghost_mask,
                         valid_mask,
                         newton,
                         *args) -> Tuple["SimpleSparseNeighborList",
                                         "NeighborListStatistics"]:
        # Make edges undirected by adding their counterpart
        invalid_idx = species.size

        # If newton is true, the transferred neighbor list is a full list.
        # Therefore, we need to set half of the edges to invalid to avoid
        # double counting.
        senders, receivers, m = args
        max_edges = m.size

        # Remove all edges that are longer than the cutoff distance
        dists = jnp.linalg.norm(position[senders] - position[receivers], axis=-1)
        invalid = dists > r_cutoff

        vs = jnp.where(invalid, invalid_idx, senders)
        vr = jnp.where(invalid, invalid_idx, receivers)

        # Prune all irrelevant edges. In the newton setting, the provided
        # neighbor list is a full list.
        graph = SimpleSparseNeighborList(vs, vr, m)
        graph, max_neighbors = lax.cond(
            newton,
            functools.partial(prune_neighbor_list, max_edges=max_edges, nbr_order=nbr_order[0], half_list=False),
            functools.partial(prune_neighbor_list, max_edges=max_edges, nbr_order=nbr_order[1], half_list=True),
            graph, ghost_mask
        )

        statistics = NeighborListStatistics(
            max_neighbors=max_neighbors,
            overlong=jnp.sum(~invalid)
        )

        return graph, statistics

    def to_neighborlist(self):
        idx = jnp.stack([self.senders, self.receivers], axis=0)
        nbrs = partition.NeighborList(
            idx, None, None, None, None, partition.Sparse, None, None, None)
        return nbrs


@dataclasses.dataclass
class SimpleDenseNeighborList(NeighborList):
    """Simple dense neighbor list representation using precomputed neighbor list.

    This neighbor list is a semi-sparse representation of a graph.
    It does not infer the neighbor list from the positions but acts as an
    interface between the precomputed neighbor list, e.g., from LAMMPS, and
    the exported model.

    This class increases the efficiency of the exported model
    while reducing necessary data transfer by pruning the neighbor list.
    Therefore, the class filters out all edges that are longer than the
    specified model cutoff distance. Moreover, it prunes all edges between
    ghost atoms (atoms not in the local domain) that are not relevant for
    a correct force computation.

    Attributes:
        senders: The sender indices of the edges.
        receivers: The receiver indices of the edges.
        max_edges: The maximum number of relevant edges in the neighbor list.

    """

    nbrs: jax.Array

    max_edges: jax.Array
    max_triplets: jax.Array

    @staticmethod
    def set_properties(proto: model_proto.Model):
        proto.neighbor_list.type = proto.NeighborListType.SIMPLE_DENSE
        proto.neighbor_list.half_list = False

    @staticmethod
    @util.define_symbols(
        "max_nbrs, max_edges, max_triplets",
        [
            "max_nbrs <= n_atoms",
            "max_edges <= n_atoms * max_nbrs",
            "max_triplets <= max_edges * max_nbrs"
        ]
    )
    def create_symbolic_input_format(max_nbrs, max_edges, max_triplets, **kwargs):

        nbrs = jax.ShapeDtypeStruct((kwargs["n_atoms"], max_nbrs), jnp.int32)
        max_edges = jax.ShapeDtypeStruct((max_edges,), jnp.bool_)
        max_triplets = jax.ShapeDtypeStruct((max_triplets,), jnp.bool_)

        return nbrs, max_edges, max_triplets

    @staticmethod
    def create_from_args(r_cutoff,
                         nbr_order,
                         position,
                         species,
                         ghost_mask,
                         valid_mask,
                         newton,
                         *args) -> Tuple["SimpleDenseNeighborList",
                                         "NeighborListStatistics"]:
        # Make edges undirected by adding their counterpart
        invalid_idx = species.size

        # If newton is true, the transferred neighbor list is a full list.
        # Therefore, we need to set half of the edges to invalid to avoid
        # double counting.
        nbrs, max_edges, max_triplets = args

        # Remove all edges that are longer than the cutoff distance
        dists = jax.vmap(
            jax.vmap(
                lambda i, j: jnp.linalg.norm(position[i] - position[j]),
                in_axes=(None, 0)
            ), in_axes=(0, 0)
        )(jnp.arange(nbrs.shape[0]), nbrs)
        invalid = dists > r_cutoff

        nbrs = jnp.where(invalid, invalid_idx, nbrs)

        # Prune all irrelevant edges. In the newton setting, the provided
        # neighbor list is a full list.
        graph = SimpleDenseNeighborList(nbrs, max_edges, max_triplets)
        graph, (max_edges, max_triplets) = lax.cond(
            newton,
            functools.partial(prune_neighbor_list_dense, nbr_order=nbr_order[0]),
            functools.partial(prune_neighbor_list_dense, nbr_order=nbr_order[1]),
            graph, ghost_mask
        )

        statistics = NeighborListStatistics(
            max_neighbors=max_edges,
            overlong=max_triplets
        )

        return graph, statistics

    def to_neighborlist(self):
        nbrs = partition.NeighborList(
            self.nbrs, None, None, None, None, partition.Dense, None, None, None)
        return nbrs



class DeviceSparseNeighborListArgs(NamedTuple):
    update: jax.Array | jax.ShapeDtypeStruct

    xcells: jax.Array | jax.ShapeDtypeStruct
    ycells: jax.Array | jax.ShapeDtypeStruct
    zcells: jax.Array | jax.ShapeDtypeStruct
    capacity: jax.Array | jax.ShapeDtypeStruct

    # ref_pos: jax.Array | jax.ShapeDtypeStruct

    # cutoff: jax.Array | jax.ShapeDtypeStruct
    # skin: jax.Array | jax.ShapeDtypeStruct

    senders: jax.Array | jax.ShapeDtypeStruct
    receivers: jax.Array | jax.ShapeDtypeStruct


@dataclasses.dataclass
class DeviceSparseNeighborList(NeighborList):
    """Creates the neighbor list graph on the device using a cell list.

    Warning: This implementation is experimental and work in progress.

    """

    @staticmethod
    def set_properties(proto: model_proto.Model):
        proto.neighbor_list.type = proto.NeighborListType.DEVICE_SPARSE

    @staticmethod
    @util.define_symbols(
        "max_neighbors, nx, ny, nz, c",
        ["c <= n_atoms", "27*c^2*nx*ny*nz >= max_neighbors"]
    )
    def create_symbolic_input_format(max_neighbors, nx, ny, nz, c, *, n_atoms, **kwargs):

        # Currently, JAX can only infer dimensions from array shapes but not the
        # input
        update = jax.ShapeDtypeStruct((1,), jnp.bool)

        xcells = jax.ShapeDtypeStruct((nx,), jnp.bool)
        ycells = jax.ShapeDtypeStruct((ny,), jnp.bool)
        zcells = jax.ShapeDtypeStruct((nz,), jnp.bool)

        capacity = jax.ShapeDtypeStruct((c,), jnp.bool)

        # We pass reference positions from the previous build to skip the
        # neighbor list construction if smaller than the input
        # ref_pos = jax.ShapeDtypeStruct((n_atoms, 3), jnp.float32)

        # Increase cutoff by this value to reuse neighbor list when particle
        # move less than half this distance
        # skin = jax.ShapeDtypeStruct(tuple(), jnp.float32)
        # cutoff = skin

        senders = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)
        # receivers = jax.ShapeDtypeStruct((max_neighbors,), jnp.int32)

        return (
            update, xcells, ycells, zcells, capacity, senders, senders
        )

    @staticmethod
    def create_from_args(r_cutoff, num_mpl, positions, species, ghost_mask, valid_mask, *args):
        nargs = DeviceSparseNeighborListArgs(*args)

        buffer = jnp.zeros(
            (
                nargs.xcells.size,
                nargs.ycells.size,
                nargs.zcells.size,
                nargs.capacity.size
            ),
            dtype=jnp.int32
        )

        # TODO: Skip the recomputation for now
        # recompute = jnp.max(
        #     jnp.sum((positions - nargs.ref_pos) ** 2.0, axis=-1)
        # ) < (nargs.skin / 2) ** 2

        update_fn = functools.partial(
            compute_neighbor_list, positions, buffer, nargs.senders,
            cutoff=r_cutoff + 2.0, mask=valid_mask # Hard-coded skin size
        )

        def reuse_fn():
            # Return the statistics from the previous build
            statistics = NeighborListStatistics(
                min_cell_capacity=nargs.capacity.size,
                cell_too_small=0,
                max_neighbors=nargs.senders.size)

            return (nargs.senders, nargs.receivers), statistics


        graph, statistics = lax.cond(nargs.update.squeeze(), update_fn, reuse_fn)

        return SimpleSparseNeighborList(*graph), (*statistics.tuple, *graph)


class DeviceListStatistics(typing.TypedDict, total=True):
    """Statistics for the :class:`DeviceSparseNeighborList`."""
    min_cell_capacity: typing.Required[int]
    cell_too_small: typing.Required[int]
    max_neighbors: typing.Required[int]


class NeighborListStatistics(typing.TypedDict, total=True):
    """Statistics for the :class:`SimpleSparseNeighborList`."""
    max_neighbors: typing.Required[int]
    overlong: typing.Optional[int]


@jax.jit
def compute_cell_list(position, id_buffer, cutoff, mask=None, eps=1e-3):
    """Assigns particle IDs into a 3D grid.

    This implementation follows the JAX, M.D. implementation, but aims to
    support building a cell list by only using shape information from the
    input arguments.

    Args:
        position: The position of the atom.
        id_buffer: Determines the dimensions of the grid and the cell
            capacities. Shape (nx, ny, nz, c) correponds to the numbers of
            cells in x,y,z dimensions and the maximum capacity per cell c.
        cutoff: Cutoff to check the dimensions of the cells. If the cell
            dimensions are smaller than the cutoff, increases the box size
            to enlarge the cells. Has the downside that cells will get fuller
            than usual, but will still yield correct neighbor list results.
        mask: Specifies whether particles should be ignored (mask = 0)
        eps: Tolerance increasing the box and cells to avoid wrong classification

    Returns:
        Returns a tuple with updated particle ids per grid and a dataclass
        containing statistics of the build.

    """
    assert mask is not None, "Requires mask argument!"

    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=bool)

    *cell_counts, capacity = id_buffer.shape

    # Shift the positions to be in the range [0, box]. First, we shift
    # the masked particles positions to not have an influence on the range.
    # Then we shift the positions to be positive.
    mean_position = jnp.mean(mask[:, jnp.newaxis] * position, axis=0, keepdims=True)
    position = jnp.where(mask[:, jnp.newaxis], position, mean_position)
    position -= jnp.min(position, axis=0, keepdims=True)

    # TODO: How big should the tolerance be?
    box = jnp.diag(jnp.max(position, axis=0) + 0.5 * cutoff)

    # Generally, the minimum cell dimension must be larger than the cutoff,
    # such that all potential neighbors are contained in the neighboring cells.
    # Potential workaround: Increase box dimension such that smallest cell size
    # is as large as the cutoff. Will work if cell capacity is big enough
    cell_sizes = jnp.diag(box) / jnp.asarray(cell_counts)
    cell_too_small = jnp.sum((cell_sizes < cutoff) * 2 ** jnp.arange(3))

    cell_too_small = jnp.sum(1 - mask)

    # Scale the box dimensions such that all cell sizes are larger than the cutoff
    cell_sizes *= 1 + (cell_sizes < cutoff) * ((cutoff - cell_sizes) / cell_sizes)

    # Get the cell ids for each particle in every dimension (n, x_id, y_id, z_id)
    # and transfrom into flat ids. Assign invalid particles an invalid
    # cell id such that they are not member to any of the cells
    nx, ny, nz = cell_counts
    max_cell_ids = 1
    for n_in_dim in cell_counts:
        max_cell_ids *= n_in_dim

    cell_ids = jnp.int32(jnp.floor(position / cell_sizes[jnp.newaxis, :]))
    cell_ids = jnp.sum(cell_ids * jnp.asarray([[nz * ny, nz, 1]]), axis=-1)
    cell_ids = jnp.where(mask, cell_ids, max_cell_ids)

    # We can now count how often a particle appears in each cell
    cell_occupancy = jax.ops.segment_sum(jnp.int32(mask), cell_ids, cell_ids.size + 1)
    min_cell_capacity = jnp.max(cell_occupancy)

    # We sort the particles along their cell id to obtain, e.g.
    # the cell id array (0, 0, 0, 1, 1, 2, 3, ...). If the capacity is
    # sufficiently large, each segment should be no longer than the capacity.
    # We now create a second array that with repeating numbers 0 ... capacity,
    # such that within segment each number appears at most once.
    sort_idx = jnp.argsort(cell_ids)
    particle_ids = jnp.arange(position.shape[0])
    unique_id_per_segment = jnp.mod(lax.iota(jnp.int32, position.shape[0]), capacity)

    new_id_buffer = jnp.full((max_cell_ids + 1, capacity), position.shape[0])
    new_id_buffer = new_id_buffer.at[cell_ids[sort_idx], unique_id_per_segment].set(particle_ids[sort_idx])
    new_id_buffer = new_id_buffer[:-1, :].reshape(id_buffer.shape)

    statistics = DeviceListStatistics(min_cell_capacity, cell_too_small, 0)
    return new_id_buffer, statistics


@jax.jit
def compute_neighbor_list(position, id_buffer, senders, cutoff, mask=None, eps=1e-3):
    """Computes a sparse neighbor list using a cell list.

    Args:
        position: The positions of the atoms.
        id_buffer: Determines the dimensions of the grid and the cell capacity.
        senders: Determines the maximum number of edges.
        cutoff: Includes neighbor up to this distance.
        mask: Specifies whether particles should be ignored (mask = 0)
        eps: Tolerance increasing the box and cells to avoid wrong classification.

    Returns:
        Returns a tuple with sender-receiver pairs and statistics of the
        neighbor list construction.

    """
    assert mask is not None, "Requires mask argument!"

    if mask is None:
        mask = jnp.ones(position.shape[0], dtype=bool)

    invalid_idx = position.shape[0]

    # Compute the offsets of all neighboring cells
    offset_in_dim = jnp.arange(3) - 1
    xn, yn, zn = jnp.meshgrid(offset_in_dim, offset_in_dim, offset_in_dim, indexing='ij')
    nx, ny, nz, capacity = id_buffer.shape

    total_edges = 27 * (nx * ny * nz) * (capacity ** 2)

    id_buffer, statistics = compute_cell_list(
        position, id_buffer, cutoff, mask=mask, eps=eps)

    # Build the neighbor list for all cells
    @functools.partial(jax.vmap, in_axes=(0, None, None))
    @functools.partial(jax.vmap, in_axes=(None, 0, None))
    @functools.partial(jax.vmap, in_axes=(None, None, 0))
    def cell_candidate_fn(cx, cy, cz):
        # Get the ids of all neighboring cells. For at least
        # three cells, this should not count edges double
        all_cx = jnp.mod(cx + xn, nx).ravel()
        all_cy = jnp.mod(cy + yn, ny).ravel()
        all_cz = jnp.mod(cz + zn, nz).ravel()

        # These are the indices of all particles that could be neighbors.
        # Senders are only local atoms such that no directed edges will be
        # coundted double
        receiver_idxs = id_buffer[all_cx, all_cy, all_cz, :]
        sender_idxs = id_buffer[cx, cy, cz, :]

        # Transform to sparse list
        cell_senders, cell_receivers = jnp.meshgrid(
            sender_idxs, receiver_idxs.ravel(), indexing='ij')
        cell_senders = cell_senders.ravel()
        cell_receivers = cell_receivers.ravel()

        sender_pos = position[cell_senders, :]
        receiver_pos = position[cell_receivers, :]

        # Compute all the distances (senders, receivers)
        dist_sq = jnp.sum((receiver_pos - sender_pos) ** 2, axis=-1)
        cut_sq = jnp.square(cutoff)

        # Select valid neighbors within cutoff that are not self
        cell_mask = dist_sq < cut_sq

        # Remove edges from or to invalid receivers
        cell_mask = jnp.logical_and(cell_mask, mask[cell_senders])
        cell_mask = jnp.logical_and(cell_mask, mask[cell_receivers])

        # Remove edges to self
        cell_mask = jnp.logical_and(cell_mask, cell_senders != cell_receivers)

        # Apply invalid indices form senders to receivers and vice versa
        cell_mask = jnp.logical_and(cell_mask, cell_senders < invalid_idx)
        cell_mask = jnp.logical_and(cell_mask, cell_receivers < invalid_idx)

        # Apply mask to neighbor list
        cell_senders = jnp.where(cell_mask, cell_senders, invalid_idx)
        cell_receivers = jnp.where(cell_mask, cell_receivers, invalid_idx)

        print(
            f"Senders: {cell_senders.shape}, Receivers: {cell_receivers.shape}")

        return cell_senders, cell_receivers

    new_senders, new_receivers = cell_candidate_fn(
        jnp.arange(nx), jnp.arange(ny), jnp.arange(nz)
    )
    new_senders, new_receivers = new_senders.ravel(), new_receivers.ravel()

    max_neighbors = senders.size
    valid_neighbors = jnp.sum(new_receivers < invalid_idx)

    _, prune_idx = lax.top_k(-new_receivers, max_neighbors)

    valid_pruned_neighbors = jnp.sum(new_receivers[prune_idx] < invalid_idx)


    statistics = statistics.set(
        max_neighbors=valid_neighbors, cell_too_small=valid_pruned_neighbors)

    return (new_senders[prune_idx], new_receivers[prune_idx]), statistics


def prune_neighbor_list(list, local, max_edges, nbr_order: int, half_list: bool = False):
    """Prunes the neighbor list by removing edges irrelevant to local atoms.

    For simplicity, a neighbor list might be built for all atoms within a, e.g.,
    rectangular domain. However, this list can contain atoms that are not
    relevant for the force computation of local atoms.
    Therefore, this function prunes the neighbor list by removing all edges
    that are not relevant for the local atoms. For example, given a simple
    lennard-jones potential, the neighbor list should only contain atoms that
    are first-order neighbors to any local atoms.

    Args:
        list: Sparse neighbor list to prune.
        local: Mask specifying the local atoms.
        max_edges: Maximum number of edges in the pruned list.
        nbr_order: Maximum order of neighbors required for the force computation.
        half_list: If True, the neighbor list is a half list. This means that
            an edge from i to j implies an edge from j to i.

    Returns:
        Returns the pruned neighbor list and the number of valid edges.

    """

    if half_list:
        # Make a full list from the half list
        senders = jnp.concat([list.senders, list.receivers], axis=0)
        receivers = jnp.concat([list.receivers, list.senders], axis=0)
    else:
        # Fill up the list with invalid indices. Required to ensure consistency
        # with half list setting
        invalid_fill = jnp.full_like(list.senders, local.size)
        senders = jnp.concat([list.senders, invalid_fill], axis=0)
        receivers = jnp.concat([list.receivers, invalid_fill], axis=0)
    list = list.set(senders=senders, receivers=receivers)

    def _update(reachable, _):
        # Pruning is deliberately independent of a model's aggregation
        # convention. Some models aggregate source -> receiver, while others
        # scatter onto the first endpoint. Expand through both endpoints, but
        # preserve the original directed edges passed to the model.
        reachable_from_receivers = jax.ops.segment_max(
            reachable[list.receivers], list.senders, reachable.size)
        reachable_from_senders = jax.ops.segment_max(
            reachable[list.senders], list.receivers, reachable.size)
        reachable |= reachable_from_receivers | reachable_from_senders
        return reachable, _

    # Non-newton case:
    # Relevant sender atoms are all atoms that are reachable via two times
    # the message passing interactions from a local atom of the domain.
    # Additional edges within the cutoff are required to correctly encode
    # the environment. We need the correct energy even for some ghost atoms
    # to compute forces without communication between domains.
    reachable, _ = lax.scan(_update, local, jnp.arange(nbr_order))
    mask = reachable[list.senders] & reachable[list.receivers]
    mask &= (list.senders < local.size) & (list.receivers < local.size)

    senders = jnp.where(mask, list.senders, local.size)
    receivers = jnp.where(mask, list.receivers, local.size)
    n_valid = jnp.sum(mask)

    # Reduce the size of the neighbor list
    mask, select = lax.top_k(mask, k=max_edges)
    senders = senders[select]
    receivers = receivers[select]

    return SimpleSparseNeighborList(senders, receivers, mask), n_valid


def prune_neighbor_list_dense(list, local, nbr_order: int):
    """Prunes a dense neighbor list.

    Args:
        list: Sparse neighbor list to prune.
        local: Mask specifying the local atoms.
        nbr_order: Maximum order of neighbors required for the force computation.

    Returns:
        Returns the pruned neighbor list, the number of valid edges, and the
        number of triplets from the valid edges.

    """

    def _update(reachable, _):
        # Send reachable messages to neighbors. Any connection to a reachable
        # node makes the node itself reachable
        print(f"Shape of reachable: {reachable[list.nbrs].shape}")
        reachable = jnp.any(
            reachable[list.nbrs] & list.nbrs < local.size,
            axis=1, keepdims=False)
        print(f"Shape of reachable (later): {reachable.shape}")
        print(f"Shape due to {(reachable[list.nbrs] & list.nbrs < reachable.size).shape}")
        return reachable, _

    reachable, _ = lax.scan(_update, local, jnp.arange(nbr_order))

    # Every node unreachable does not send out edges (row will be zero).
    # Every node unreachable does not receive edges (check indices).
    nbrs = jnp.where(reachable[:, None], list.nbrs, local.size)
    nbrs = jnp.where(reachable[nbrs], nbrs, local.size)

    nbrs_per_atom = jnp.sum(nbrs < reachable.size, axis=1)
    max_edges = jnp.sum(nbrs_per_atom)
    max_triplets = jnp.sum(nbrs_per_atom * (nbrs_per_atom - 1))

    return list.set(nbrs=nbrs), (max_edges, max_triplets)

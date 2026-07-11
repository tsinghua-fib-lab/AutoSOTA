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

"""Utility functions helpful in designing new trainers."""
from functools import partial
from typing import Any, cast

import chex
import cloudpickle as pickle
# import h5py
import jax
from jax import tree_util, device_count, numpy as jnp, lax
from jax.tree_util import tree_map

import jax_md_mod
from jax_md import simulate, partition
import numpy as onp

try:
    import mpi4py.MPI as MPI
except Exception:
    MPI = None

try:
    import mpi4jax
except Exception:
    mpi4jax = None


def use_mpi():
    """Whether MPI is available."""
    if MPI is None:
        return False
    try:
        if hasattr(MPI, "Is_initialized") and not MPI.Is_initialized():
            return False
        return MPI.COMM_WORLD.Get_size() > 1
    except Exception as e:
        print(f"Initializing MPI failed with error: {e}")
        return False


def is_root():
    """Whether the current MPI process is the root process."""
    if not use_mpi():
        return True
    assert MPI is not None
    return MPI.COMM_WORLD.Get_rank() == 0


def get_mpi():
    """Returns the MPI communicator."""
    if not use_mpi():
        return None
    assert MPI is not None
    return MPI


def get_communicator():
    """Returns the MPI communicator."""
    if not use_mpi():
        return None
    assert MPI is not None
    return MPI.COMM_WORLD


def mpi_tree_slice(tree, dim=None):
    """Slices a pytree with disjoint subsets across MPI processes."""
    if not use_mpi():
        return tree, None

    leaves = tree_util.tree_leaves(tree)
    if dim is None:
        dim = leaves[0].shape[0]

    assert all(
        leaf.shape[0] == leaves[0].shape[0] for leaf in leaves
    ), 'Tree first dimension size is not equal.'

    comm = get_communicator()
    assert comm is not None
    rank = comm.Get_rank()
    size = comm.Get_size()
    return tree_util.tree_map(lambda x: x[rank::size], tree), dim


def mpi_tree_broadcast(tree, root: int = 0):
    """Broadcast a pytree from `root` to all MPI processes using mpi4jax."""
    if not use_mpi():
        return tree

    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_broadcast."
        )

    MPI_mod = cast(Any, MPI)
    mpi4jax_mod = cast(Any, mpi4jax)

    comm = get_communicator()
    assert comm is not None
    rank = comm.Get_rank()

    def _bcast_leaf(x):
        x = jnp.asarray(x)
        if rank != root:
            x = jnp.zeros_like(x)
        res = mpi4jax_mod.allreduce(
            x, MPI_mod.LOR if x.dtype == jnp.bool_ else MPI_mod.SUM, comm=comm
        )
        return res[0] if isinstance(res, tuple) else res

    return tree_util.tree_map(_bcast_leaf, tree)


def mpi_tree_bcast(tree, root: int = 0):
    return mpi_tree_broadcast(tree, root=root)


def mpi_tree_gather(tree, dim=None):
    """Gathers a pytree from all MPI processes."""
    if not use_mpi():
        return tree

    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_gather/mpi_tree_mean inside jitted code."
        )
    MPI_mod = cast(Any, MPI)
    mpi4jax_mod = cast(Any, mpi4jax)
    
    comm = get_communicator()
    assert comm is not None
    rank = comm.Get_rank()
    size = comm.Get_size()

    gathered_tree = tree_util.tree_map(
        lambda x: jnp.zeros(
            (size * x.shape[0] if dim is None else dim, *x.shape[1:]), dtype=x.dtype
        ).at[rank::size].set(x), tree
    )
    tree = tree_util.tree_map(
        lambda x: mpi4jax_mod.allreduce(
            x, MPI_mod.LOR if x.dtype == jnp.bool_ else MPI_mod.SUM, comm=comm
        ), gathered_tree
    )

    tree = tree_util.tree_map(lambda x: x[0] if isinstance(x, tuple) else x, tree)

    return tree


def mpi_tree_mean(tree, dim=None):
    """Mean of a pytree from all MPI processes."""
    if not use_mpi():
        return tree

    if mpi4jax is None:
        raise RuntimeError(
            "MPI is available (mpi4py), but mpi4jax is not installed. "
            "Install mpi4jax to use mpi_tree_gather/mpi_tree_mean inside jitted code."
        )
    MPI_mod = cast(Any, MPI)
    mpi4jax_mod = cast(Any, mpi4jax)
    
    comm = get_communicator()
    assert comm is not None
    rank = comm.Get_rank()
    size = comm.Get_size()

    if dim is None:
        slice_size = 1
        dim = size
    else:
        slice_size = onp.arange(dim)[rank::size].size

    tree = tree_util.tree_map(
        lambda x: mpi4jax_mod.allreduce(
            x * (slice_size / dim), MPI_mod.SUM, comm=comm
        ), tree
    )

    tree = tree_util.tree_map(lambda x: x[0] if isinstance(x, tuple) else x, tree)

    return tree


# freezing seems to give slight performance improvement
@partial(chex.dataclass, frozen=True)
class TrainerState:
    """Each trainer at least contains the state of parameter and
    optimizer.
    """
    params: Any
    opt_state: Any


def _get_box_kwargs_if_npt(state):
    kwargs = {}
    if is_npt_ensemble(state):
        box = simulate.npt_box(state)
        kwargs['box'] = box
    return kwargs


def neighbor_update(neighbors, state, **kwargs):
    """Update neighbor lists irrespective of the ensemble.

    Fetches the box to the neighbor list update function in case of the
    NPT ensemble.

    Args:
        neighbors: Neighbor list to be updated
        state: Simulation state

    Returns:
        Updated neighbor list
    """
    kwargs.update(_get_box_kwargs_if_npt(state))
    nbrs = neighbors.update(state.position, **kwargs)
    return nbrs


def neighbor_allocate(neighbor_fn, state, extra_capacity=0):
    """Re-allocates neighbor lost irrespective of ensemble. Not jitable.

    Args:
        neighbor_fn: Neighbor function to re-allocate neighbor list
        state: Simulation state
        extra_capacity: Additional capacity of new neighbor list

    Returns:
        Updated neighbor list
    """
    kwargs = _get_box_kwargs_if_npt(state)
    nbrs = neighbor_fn.allocate(state.position, extra_capacity, **kwargs)
    return nbrs


def is_npt_ensemble(state):
    """Whether a state belongs to the NPT ensemble."""
    return hasattr(state, 'box_position')


def kl(p, q):
    """Returns Kullback-Leibler divergence D(P || Q) for discrete distributions.

    Args:
        p: Discrete probability density function values (array-like, shape=n).
        q: Discrete probability density function values (array-like, shape=n).
    """
    p = jnp.asarray(p, dtype=float)
    q = jnp.asarray(q, dtype=float)
    return -jnp.sum(jnp.where(p != 0, p * jnp.log(q / p), 0))


def jenson_shannon(p, q, distance=False):
    """Returns the Jensen-Shannon distance between two discrete distributions.

    Args:
        p: Discrete probability density function values (array-like, shape=n).
        q: Discrete probability density function values (array-like, shape=n).
        distance: Default False returns JS divergence. True returns JS distance
                  sqrt(JS).
    """
    p = onp.array(p)
    q = onp.array(q)
    m = 0.5 * (p + q)
    js = 0.5 * (kl(p, m) + kl(q, m))
    if distance:
        return onp.sqrt(js)
    else:
        return js


def tree_combine(tree):
    """Combines the first two axes of `tree`, e.g. after batching."""
    return tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), tree)


def tree_norm(tree):
    """Returns the Euclidean norm of a PyTree."""
    leaves, _ = tree_util.tree_flatten(tree)
    return sum(jnp.vdot(x, x) for x in leaves)


def tree_get_single(tree, n=0):
    """Returns the n-th tree of a tree-replica, e.g. from pmap.
    By default, the first tree is returned.
    """
    single_tree = tree_map(lambda x: jnp.array(x[n]), tree)
    return single_tree


def tree_set(tree, new_data, end, start=0):
    """Overrides entries of a tree from index start:end along axis 0
    with new_data.
    """
    return tree_map(lambda leaf, new_data_leaf:
                    leaf.at[start:end, ...].set(new_data_leaf), tree, new_data)


def tree_get_slice(tree, idx_start, idx_stop, take_every=1, to_device=True):
    """Returns a slice of trees taken from a tree-replica along axis 0."""
    if to_device:
        return tree_map(lambda x: jnp.array(x[idx_start:idx_stop:take_every]),
                        tree)
    else:
        return tree_map(lambda x: x[idx_start:idx_stop:take_every], tree)


def tree_take(tree, indicies, axis=0, on_cpu=True):
    """Tree-wise application of numpy.take."""
    numpy = onp if on_cpu else jnp
    return tree_map(lambda x: numpy.take(x, indicies, axis), tree)


def tree_put(tree, indicies, values, axis=0, on_cpu=True):
    """Tree-wise application of numpy.put_along_axis."""
    if on_cpu:
        return tree_map(
            lambda x, y: onp.put_along_axis(x, indicies, y, axis), values)
    else:
        assert axis == 0, 'Only axis=0 is supported for jax.'
        return tree_map(
            lambda x, y: x.at[indicies, ...].set(y), tree, values)


def tree_delete(tree, indicies, axis=None, on_cpu=True):
    """Returns a tree, where entries at position indicies along axis are
    deleted from the original tree.
    """
    numpy = onp if on_cpu else jnp
    return tree_map(lambda leaf: numpy.delete(leaf, indicies, axis=axis), tree)


def tree_append(orig_tree, tree_to_append, axis=None, on_cpu=True):
    numpy = onp if on_cpu else jnp
    return tree_map(partial(numpy.append, axis=axis), orig_tree, tree_to_append)


def tree_replicate(tree):
    """Replicates a pytree along the first axis for pmap."""
    return tree_map(lambda x: jnp.array([x] * device_count()), tree)


def tree_axis_swap(tree, axis1=0, axis2=1):
    """Swaps axes of all leaves of a pytree."""
    return tree_map(lambda x: jnp.swapaxes(x, axis1, axis2), tree)


def tree_concat(tree):
    """For output computed in parallel via pmap, restacks all leaves such that
    the parallel dimension is again along axis 0 and the leading pmap dimension
    vanishes.
    """
    return tree_map(partial(jnp.concatenate, axis=0), tree)


def tree_pmap_split(tree, n_devices):
    """Splits the first axis of `tree` evenly across the number of devices for
     pmap batching (size of first axis is n_devices).
     """
    assert tree_util.tree_leaves(tree)[0].shape[0] % n_devices == 0, \
        'First dimension needs to be multiple of number of devices.'
    return tree_map(lambda x: jnp.reshape(x, (n_devices, x.shape[0]//n_devices,
                                              *x.shape[1:])), tree)


def tree_vmap_split(tree, batch_size):
    """Splits the first axis of a 'tree' with leaf sizes (N, X)`into
    (n_batches, batch_size, X) to allow straightforward vmapping over axis0.
    """
    if len(tree_util.tree_leaves(tree)) == 0:
        return tree

    assert tree_util.tree_leaves(tree)[0].shape[0] % batch_size == 0, \
        'First dimension of tree needs to be splittable by batch_size' \
        ' without remainder.'
    return tree_map(lambda x: jnp.reshape(x, (x.shape[0] // batch_size,
                                              batch_size, *x.shape[1:])),
                    tree)


def tree_sum(*tree_list, axis=0):
    """Computes the sum of equal-shaped leafs of a pytree."""
    @partial(partial, tree_map)
    def leaf_add(*leafs):
        return jnp.sum(jnp.stack(leafs, axis=axis), axis=axis)
    return leaf_add(*tree_list)


def tree_mean(tree_list):
    """Computes the mean a list of equal-shaped pytrees."""
    @partial(partial, tree_map)
    def tree_add_imp(*leafs):
        return jnp.mean(jnp.stack(leafs), axis=0)

    return tree_add_imp(*tree_list)


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]
    Useful for turning the output of a vmapped function into normal objects.

    From: https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
    """
    leaves, treedef = tree_util.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(l) for l in new_leaves]
    return new_trees


def tree_multiplicity(tree):
    """Returns the number of stacked trees along axis 0."""
    leaves, _ = tree_util.tree_flatten(tree)
    return leaves[0].shape[0]


def assert_distributable(total_samples, n_devies, vmap_per_device):
    assert total_samples % (n_devies * vmap_per_device) == 0, (
        'For parallelization, the samples need to be evenly distributed'
        'over the devices and vmap, i.e. be a multiple of n_devices * n_vmap.')


def load_trainer(file_path):
    """Returns the trainer saved via 'trainer.save_trainer'.

    Args:
        file_path: Path of pickle file containing trainer.

    """
    with open(file_path, 'rb') as pickle_file:
        trainer = pickle.load(pickle_file)
    trainer.move_to_device()
    return trainer


def format_not_recognized_error(file_format):
    raise ValueError(f'File format {file_format} not recognized. '
                     f'Expected ".hdf5" or ".pkl".')

def batch_map(f, xs, batch_size: int = 1):
    """Maps a function over an array in batches.

    Substitute for ``lax.map`` with batch size argument from later jax versions.

    Args:
        f: Function to map.
        xs: List of arguments to map over.
        batch_size: Size of each batch.

    Returns:
        Returns results of f evaluated at element entry of xs.

    """

    f_vmapped = jax.vmap(f)
    tree_leaves, tree_structure = tree_util.tree_flatten(xs)

    # Ensure that the batch size is not larger than the number of samples
    batch_size = onp.min([batch_size, tree_leaves[0].shape[0]])

    # First, we split the pytree into batch and remainder part
    batches = []
    remainders = []

    for leave in tree_leaves:
        n_batches = leave.shape[0] // batch_size
        remainder = leave.shape[0] % batch_size

        if n_batches > 0:
            batches.append(jnp.reshape(leave[:n_batches * batch_size], (n_batches, batch_size, *leave.shape[1:])))
        if remainder > 0:
            remainders.append(leave[-remainder:])

    # Then, we map over the batches and compute the remainder in a single pass
    batch_results = lax.map(
        f_vmapped, tree_util.tree_unflatten(tree_structure, batches))
    # We are done if we can split the data into batches without remainder
    if len(remainders) == 0:
        return tree_concat(batch_results)

    remainder_results = f_vmapped(
        tree_util.tree_unflatten(tree_structure, remainders))

    # Concatenate remainder results and batches
    results = tree_util.tree_map(
        lambda x, y: jnp.concat([x, y], axis=0),
        tree_concat(batch_results), remainder_results
    )

    return results

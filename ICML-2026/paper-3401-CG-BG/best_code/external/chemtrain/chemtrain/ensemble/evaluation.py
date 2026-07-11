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
from typing import Any, Dict, Callable, NamedTuple

import numpy as onp
from jax import tree_util, numpy as jnp, vmap, lax, Array
from jax._src.basearray import ArrayLike

from jax_md_mod import custom_partition
from jax_md import simulate
from jax_md.partition import NeighborList

from chemtrain import util
from chemtrain.typing import QuantityDict

from typing import Protocol

"""Functions to compute snapshots quantities for many samples."""

class State(Protocol):
    """State of a molecular system.

    All states must at least prescribe the particle positions.
    Other attributes, such as velocities, forces, etc., might be necessary
    for some instantaneous quantities.

    Attributes:
        position: Particle positions

    """
    position: ArrayLike


class SimpleState(NamedTuple):
    """Simplest state of a molecular system.

    Args:
        position: Particle positions
    """
    position: ArrayLike


def quantity_map(states: State,
                 quantities: QuantityDict,
                 nbrs: NeighborList = None,
                 state_kwargs: Dict[str, Array] = None,
                 constant_state_kwargs: Dict[str, Array] = None,
                 energy_params: Any = None,
                 batch_size: int = 1,
                 feature_extract_fns: Dict[str, Callable] = None):
    """Computes quantities of interest for all states in a trajectory.

    Arbitrary quantity functions can be provided via the quantities-dict.
    The quantities dict provides the function to compute the quantity on
    a single snapshot. The resulting quantity trajectory will be saved in
    a dict under the same key as the input quantity function.

    Example usage:
        .. code-block:: python

            def custom_compute_fn(state, neighbor=None, feature=None, **kwargs):
                ...
                return quantity_snapshot


            quantities = {
                'energy': custom_quantity.energy_wrapper(energy_template_fn),
                'custom_quantity': custom_compute_fn
            }

            # Results will be available to all snapshot compute functions
            feature_extract_fns = {
                'feature': custom_feature_compute_fn
            }

            quantity_trajs = quantity_map(
                trajectory, quantities, reference_nbrs, dynamic_kwargs,
                energy_params, feature_extract_fns=feature_extract_fns
            )
            custom_quantity = quantity_trajs['custom_quantity']


    Args:
        states: System states, concatenated along the first dimensions of the
            arrays.
        quantities: The quantity dict containing for each target quantity
            the snapshot compute function
        nbrs: Reference neighbor list to compute new neighbor list
        state_kwargs: Kwargs to supply reference ``'kT'`` and/or ``'pressure'``
            to the energy function or the quantity functions.
        constant_state_kwargs: Kwargs to supply information to the energy
            function that is constant over all states.
        energy_params: Energy params for energy_fn_template to initialize
            the current energy_fn
        batch_size: Number of batches for vmap
        feature_extract_fns: Callables to compute features accessible to all
            snapshot compute functions.

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    return quantity_multimap(
       states, quantities=quantities, nbrs=nbrs,
        state_kwargs=state_kwargs, constant_state_kwargs=constant_state_kwargs,
        energy_params=energy_params, batch_size=batch_size,
        feature_extract_fns=feature_extract_fns)


def quantity_multimap(*states: State,
                      quantities: QuantityDict,
                      nbrs: NeighborList = None,
                      state_kwargs: Dict[str, Array] = None,
                      constant_state_kwargs: Dict[str, Array] = None,
                      energy_params: Any = None,
                      batch_size: int = 1,
                      feature_extract_fns: Dict[str, Callable] = None):
    """Computes quantities of interest for all states in a trajectory.

    This function extends :func:`quantity_map`
    to quantities with respect to multiple reference states.
    Therefore, the quantity function signature changes to

    .. code-block:: python

            def quantity_fn(*states, neighbor=None, energy_params=None, **kwargs):
                ...

    The keywords arguments, i.e. the neighbor list, are with respect to the
    first state of `*states`.

    Args:
        states: System states, concatenated along the first dimensions of the
            arrays.
        quantities: The quantity dict containing for each target quantity
            the snapshot compute function
        nbrs: Reference neighbor list to compute new neighbor list
        state_kwargs: Kwargs to supply reference ``'kT'`` and/or ``'pressure'``
            to the energy function or the quantity functions.
        constant_state_kwargs: Kwargs to supply information to the energy
            function that is constant over all states.
        energy_params: Energy params for energy_fn_template to initialize
            the current energy_fn
        batch_size: Number of batches for vmap
        feature_extract_fns: Callables to compute features accessible to all
            snapshot compute functions.

    Returns:
        A dict of quantity trajectories saved under the same key as the
        input quantity function.
    """
    nbrs_update_fn = nbrs.update_fn

    # Check that all states have the same format
    if state_kwargs is None:
        state_kwargs = {}
    if constant_state_kwargs is None:
        constant_state_kwargs = {}

    assert len(states) > 0, 'Need at least one trajectory.'
    ref_leaves, ref_struct = tree_util.tree_flatten(states[0])
    for traj in states:
        assert ref_struct == tree_util.tree_structure(traj), (
            "All trajectory states must have the same tree structure."
        )
        assert onp.all([
            jnp.shape(l) == jnp.shape(r)
            for r, l in zip(ref_leaves, tree_util.tree_leaves(traj))
        ]), "All trajectory state leaves must be of identical shape."

    # Extract additional features, making them accessible to all snapshot
    # compute functions. For example, when predicting molecular properties
    # using a neural network.
    if feature_extract_fns is None:
        feature_extract_fns = {}
    else:
        assert len(states) == 1, (
            "Feature extraction functions are only supported for single "
            "trajectory."
        )

    @vmap
    def single_state_quantities(single_snapshot):
        states, kwargs = single_snapshot

        kwargs.update(energy_params=energy_params)
        kwargs.update(constant_state_kwargs)

        # Add a masked neighbor list if masked and neighbor list are provided
        if util.is_npt_ensemble(states):
            box = simulate.npt_box(states[0])
            kwargs['box'] = box
        if nbrs is not None:
            new_nbrs = nbrs_update_fn(states[0].position, nbrs, **kwargs)
            mask = kwargs.get(
                "mask", jnp.ones(new_nbrs.reference_position.shape[0]))
            kwargs["neighbor"] = custom_partition.mask_neighbor_list(
                new_nbrs, mask)

        # Extract additional features to all snapshot computation functions,
        # e.g., the neighbor list graph. Next features can be beased on
        # previously computed features.
        for key in feature_extract_fns.keys():
            kwargs[key] = feature_extract_fns[key](states[0], **kwargs)

        if len(states) == 1:
            computed_quantities = {
                quantity_fn_key: quantities[quantity_fn_key](states[0], **kwargs)
                for quantity_fn_key in quantities
            }
        else:
            computed_quantities = {
                quantity_fn_key: quantities[quantity_fn_key](*states, **kwargs)
                for quantity_fn_key in quantities
            }
        return computed_quantities

    # If the batch size is larger than the number of samples, reduce the
    # batch size to compute all samples in one batch.
    # If the batch size does not divide the number of samples, we compute
    # the remainder in a separate call to avoid padding.

    ipt = (states, state_kwargs)
    if states[0].position.shape[0] < batch_size:
        batch_size = states[0].position.shape[0]

    remainder = states[0].position.shape[0] % batch_size

    if remainder > 0:
        rmd = tree_util.tree_map(lambda x: x[-remainder:], ipt)
        ipt = tree_util.tree_map(lambda x: x[:-remainder], ipt)

    batched_samples = util.tree_vmap_split(ipt, batch_size)
    batched_quantity_trajs = lax.map(single_state_quantities, batched_samples)
    quantity_trajs = util.tree_combine(batched_quantity_trajs)

    if remainder > 0:
        rmd_trajs = single_state_quantities(rmd)
        quantity_trajs = tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=0),
            quantity_trajs, rmd_trajs
        )

    return quantity_trajs


# Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich
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

"""Utilities to connect models to chemtrain."""

import numpy as onp

import jax
import jax.numpy as jnp
from jax import lax

from typing import Protocol, Any, Tuple


class ApplyFn(Protocol):
    """GNN apply function protocol."""
    def __call__(
        self,
        params: Any,
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        edge_features: Tuple[jnp.ndarray],
        node_features: Tuple[jnp.ndarray],
    ) -> jnp.ndarray: ...



def batch_apply_fn(_apply_fn: ApplyFn) -> ApplyFn:
    """Write custom vmap rules for the apply function.

    Instead of batching over graphs, combines all graphs into a supergraph.
    This step avoids vmapping operations in the neural network.

    Args:
        apply_fn: Mapping from (params, vectors, senders, receivers, species, mask)
            to per-particle energies.


    Returns:
        A wrapped apply function with custom vmap rules that avoids vmapping
        in the neural network.

    """

    def apply_fn(*args, **kwargs):
        res = _apply_fn(*args, **kwargs)
        return res


    def wrapped(params, senders, receivers, edge_features, node_features):
        # Check inputs. Node features are important to correctly rewrite invalid
        # indices in the supergraph.
        assert len(node_features) > 0, "At least one node feature array is required."
        
        return wrapped_apply_fn(
            params, senders, receivers, edge_features, node_features
        )


    @jax.custom_vjp
    @jax.custom_batching.custom_vmap
    def wrapped_apply_fn(params, senders, receivers, edge_features, node_features):
        return apply_fn(params, senders, receivers, edge_features, node_features)
        
    def wrapped_fun_fwd(*args):
        y = wrapped_apply_fn(*args)
        return y, args

    @jax.custom_batching.custom_vmap
    def wrapped_fun_bwd(res, y_bar):        
        _, vjp_fn = jax.vjp(apply_fn, *res)

        return vjp_fn(y_bar)
    
    @wrapped_fun_bwd.def_vmap
    def wrapped_fun_bwd_batch(
            axis_size, in_batched, res, y_bar):
        
        args_batched, y_bar_batched = in_batched
        # bparams, *_, bedge_features, bnode_features = args_batched
 
        if not y_bar_batched:
            y_bar = jnp.tile(y_bar[jnp.newaxis, :], (axis_size, 1))

        # The batched function figures out batching of parameters and graphs
        _, vjp_fn = jax.vjp(
            lambda *args: wrapped_fun_batch(
                axis_size, args_batched, *args, compute_vjp=True
            )[0], *res
        )

        grads = vjp_fn(y_bar)

        return grads, args_batched

    def wrapped_fun_batch(
            axis_size, in_batched, params, senders, receivers, edge_features,
            node_features, compute_vjp=False):

        bparams, *inputs_batched = in_batched

        # We must avoid the custom_vmap decorated function, as no vjp rule
        # is defined for it.
        if compute_vjp:
            fn = apply_fn
        else:
            fn = wrapped_apply_fn

        # Find out whether any of the parameters is batched
        batched_params = jax.tree.reduce(
            onp.logical_or, bparams, onp.bool(False)
        )
 
        # If the parameters are batched, the super-graph strategy does not work.
        # Thus, we fall back to a sequential evaluation via lax.map.

        if batched_params:
            # Ensure that all params are in the batched shape
            args_tiled = jax.tree.map(
                lambda l, b: jnp.tile(l, (axis_size,) + (1,) * (l.ndim - 1))
                if not b else l,
                (params, senders, receivers, edge_features, node_features),
                (bparams, *inputs_batched),
            )

            energies = lax.map(
                lambda args: fn(*args), args_tiled
            )

            return energies, True
        else:
            
            flattened_graph = flatten_graph(
                axis_size, inputs_batched, senders, receivers, edge_features,
                node_features
            )

            energies_flat = fn(params, *flattened_graph)

            # Unflatten the results
            return energies_flat.reshape((axis_size, -1)), True

    wrapped_apply_fn.defvjp(wrapped_fun_fwd, wrapped_fun_bwd)
    wrapped_apply_fn.def_vmap(wrapped_fun_batch)

    return wrapped


def flatten_graph(axis_size, in_batched, senders, receivers, edge_features, node_features):
        """Flattens batched graphs into a supergraph."""

        bsenders, breceivers, bedge_features, bnode_features = in_batched

        num_graphs = axis_size
        num_edges = (
            senders.shape[1] if bsenders
            else senders.shape[0]
        )
        natoms = (
            node_features[0].shape[1] if bnode_features[0]
            else node_features[0].shape[0]
        )

        if bsenders:
            assert breceivers, (
                "If vectors are batched, senders and receivers must be batched."
            )

        else:
            assert not breceivers, (
                "If vectors are not batched, senders and receivers must not be batched."
            )

            senders = jnp.tile(
                senders[None, :], (num_graphs, 1) + (1,) * (senders.ndim -1)
            )
            receivers = jnp.tile(
                receivers[None, :], (num_graphs, 1) + (1,) * (receivers.ndim -1)
            )            

        # Relabel the senders and receiver indices. Offset the senders
        # by the number of atoms in previous graphs.
        senders = jnp.where(
            senders.ravel() < natoms,
            senders.ravel() + natoms * jnp.repeat(jnp.arange(num_graphs), num_edges),
            num_graphs * natoms
        )
        receivers = receivers.reshape((-1, 2))
        receivers = jnp.where(
            receivers.ravel() < natoms,
            receivers.ravel() + natoms * jnp.repeat(jnp.arange(num_graphs), num_edges),
            num_graphs * natoms
            )

        edge_features_flat = ()
        for b, feat in zip(bedge_features, edge_features):
            if b:
                edge_features_flat += (jnp.reshape(feat, (-1,) + feat.shape[2:]),)
            else:
                edge_features_flat += (jnp.tile(
                    feat[None, :], (num_graphs, 1) + (1,) * (feat.ndim -1)
                ).ravel(),)

        node_features_flat = ()
        for b, feat in zip(bnode_features, node_features):
            if b:
                node_features_flat += (jnp.reshape(feat, (-1,) + feat.shape[2:]),)
            else:
                node_features_flat += (jnp.tile(
                    feat[None, :], (num_graphs, 1) + (1,) * (feat.ndim -1)
                ).ravel(),)

        return senders, receivers, edge_features_flat, node_features_flat

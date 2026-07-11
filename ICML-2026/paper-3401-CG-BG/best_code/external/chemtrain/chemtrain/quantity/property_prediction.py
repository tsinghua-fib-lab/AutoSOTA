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

"""This module contains molecular properties, computed from features of a
neural network used for potential energy prediction.
"""
import functools
import typing
from functools import wraps
from typing import Tuple, Callable, Any

import jax
from jax import numpy as jnp, Array

from jax_md import partition

from chemtrain.typing import ComputeFn

def readout_wrapper(energy_fn_template, mode="energy"):
    """Utility function to read out additional predictions from energy function.

    The energy function returned by the energy function template must have
    a keyword argument to select the return mode, e.g.

    >>> def energy_fn(pos, neighbor, mode=None, **kwargs):
    ...     ...
    ...     if mode is None:
    ...         return pot
    ...     if mode == "some_property":
    ...         return some_property

    The energy function can then be used to predict also other quantities:
    >>> quantity_dict = {
    ...     "some_property": readout_wrapper(energy_fn_template, mode="some_property")
    ... }

    """
    def snapshot_fn(state, neighbor=None, energy_params=None, **kwargs):
        assert energy_params is not None, "Energy parameters must be provided."

        prediction = energy_fn_template(energy_params)(state.position, neighbor, mode=mode, **kwargs)
        return prediction
    return snapshot_fn


def molecular_property_predictor(model, n_per_atom=0):
    """Wraps models that predict per-atom quantities to predict both global and
    per-atom quantities.

    Args:
        model: Initialized model predicting per-atom quantities, e.g. DimeNetPP.
        n_per_atom: Number of per-atom quantities to predict. Remaining
            predictions are assumed to be global.

    Returns:
        A tuple (global_properties, per_atom_properties) being a (n_globals,)
        and a (n_particles, n_per_atom) array of predictions.
    """
    @wraps(model)
    def property_wrapper(*args, **kwargs):
        per_atom_quantities = model(*args, **kwargs)
        n_predicted = per_atom_quantities.shape[1]
        n_global = n_predicted - n_per_atom
        per_atom_properties = per_atom_quantities[:, n_global:]
        global_properties = jnp.sum(per_atom_quantities[:, :n_global], axis=0)
        return global_properties, per_atom_properties
    return property_wrapper


class PropertyPredictor(typing.Protocol):

    def __call__(self, params, graph: Any, **kwargs) -> Tuple[Array, Array]:
        """Predicts molecular properties from a molecular graph.

        The form of the graph depends on the underlying model. E.g, for
        property predictions with
        :class:`jax_md_mod.model.neural_networks.DimeNetPP`, the graph is of
        the form :class:`jax_md_mod.model.sparse_graph.SparseDirectionalGraph`.

        Args:
            graph: Molecular graph containing neighborhood information.
            **kwargs: Additional arguments to the potential model,
                extracting features from the molecular graph.

        Returns:
            A tuple of global and per-atom properties.

        """


class SinglePropertyPredictor(typing.Protocol):

    @typing.overload
    def __call__(self, params, graph: Any, **kwargs): ...
    @typing.overload
    def __call__(self, features, **kwargs): ...
    @staticmethod
    def __call__(params=None, graph=None, features=None, **kwargs):
        """Predicts a molecular property from a molecular graph.

        The form of the graph depends on the underlying model. E.g, for
        property predictions with
        :class:`jax_md_mod.model.neural_networks.DimeNetPP`, the graph is of
        the form :class:`jax_md_mod.model.sparse_graph.SparseDirectionalGraph`.

        Args:
            graph: Molecular graph containing neighborhood information.
                Required, if features should be computed from a model within
                the predictor.
            features: Features derived from the molecular graph.
                Required if no model is provided to compute the features.
            **kwargs: Additional arguments to the potential model,
                extracting features from the molecular graph.

        Returns:
            Returns a single per-atom or molecular property.
        """


def apply_model(model: PropertyPredictor = None):
    """Initializes a molecular property predictor.

    If no model is provided, the global and per-atom features must be
    pre-computed. The property predictor can then be called as::

        property = property_predictor(global_features, per_atom_features, **kwargs)

    Otherwise, the features are computed via calling the model within the
    property predictor. Then, the property predictor must be called with
    a molecular graph as input::

        property = property_predictor(graph, **kwargs)

    Args:
        model: Optional model to compute the molecular features within the
            property predictor.

    """
    def decorator(fn) -> SinglePropertyPredictor:
        def predictor(params=None, graph=None, features=None, **kwargs):
            if model is not None:
                assert graph is not None and params is not None, (
                    "A graph is required to compute molecular features"
                )
                global_features, per_atom_features = model(params, graph, **kwargs)
                features = {
                    "global_features": global_features,
                    "per_atom_features": per_atom_features
                }
            else:
                assert features is not None, (
                    "If molecular features are not pre-computed, a model must "
                    "be provided to compute them."
                )

            return fn(features, **kwargs)
        return predictor
    return decorator


def snapshot_quantity(property_predictor: SinglePropertyPredictor,
                      graph_from_neighbor_list: Callable = None,
                      features_key = "features") -> ComputeFn:
    """Transforms a single property predictor to a snapshot compute function.

    Args:
        property_predictor: Function to predict a property from a molecular
            graph or from pre-extracted features.
        graph_from_neighbor_list: Function to build a molecular graph from
            a neighbor list. Only necessary, when the features should be
            extracted within the property predictor. Otherwise, pre-extracted
            global and per-atom features are required as kwargs.
        features_key: Key to the pre-computed features if no model is provided.

    Returns:
        Returns a snapshot compute function for the corresponding property.

    """

    def compute_fn(state,
                   neighbor=None,
                   energy_params=None,
                   **kwargs):

        if graph_from_neighbor_list is None:
            assert features_key in kwargs, (
                f"Features {features_key} must be pre-computed if model is not "
                f"provided."
            )

            return property_predictor(kwargs[features_key], **kwargs)

        else:
            assert neighbor is not None, (
                "A neighbor list is required to build a molecular graph."
            )

            # Enables to remove processed arguments
            mol_graph, kwargs = graph_from_neighbor_list(
                state.position, neighbor, **kwargs)
            return property_predictor(energy_params, mol_graph, **kwargs)

    return compute_fn


def init_feature_pre_computation(model: PropertyPredictor,
                                 graph_from_neighbor_list: Callable = None
                                 ) -> ComputeFn:
    """Initializes a function to compute global and per-atom features.

    Args:
        model: Model to compute the features from a molecular graph.
        graph_from_neighbor_list: Function to construct a molecular graph
            from the neighbor list.

    Returns:
        Returns a function to compute the features from a molecular graph.

    """

    def feature_computation_fn(state, neighbor=None, energy_params=None, **kwargs):
        graph = graph_from_neighbor_list(state.position, neighbor, **kwargs)
        global_features, per_atom_features = model(energy_params, graph, **kwargs)
        return {
            "global_features": global_features,
            "per_atom_features": per_atom_features
        }

    return feature_computation_fn


def potential_energy_prediction(model: PropertyPredictor = None,
                                feature_number: int = 0
                                ) -> SinglePropertyPredictor:
    """Initializes a prediction of the potential energy.

    This wrapper allows to use the same features for the prediction of the
    potential energy for a simulation and for other molecular properties.

    Args:
        model: Particle property prediction model.
        feature_number: Number of the global features to interpret as
            potential energy.

    Example::

        init_property_predictor, property_predictor = neural_networks.dimenetpp_property_prediction(
            r_cutoff = 1.0, n_targets = 2, n_species = 2, n_per_atom = 0)

        # Initialize the prediction of potential energy (the first global property)
        potential_energy_predictor = property_prediction.potential_energy_prediction(
            model=property_predictor, feature_number=0
        )

        # Initialize a function to compute the potential energy for a simulator
        # snapshot. The snapshot function first constructs a molecular graph
        # from a provided neighbor list.
        energy_snapshot_fn = property_prediction.snapshot_quantity(
            potential_energy_predictor, graph_from_neighbor_list
        )

        # The snapshot function can be used as learnable model or as
        # compute function for traj_util.quantity_traj
        def energy_fn_template(energy_params):
            def energy_fn(position, neighbor=None, **kwargs):
                # Wrap positions in pseudo simulator state
                state = force_matching.State(position)
                return energy_snapshot_fn(position, neighbor, energy_params=energy_params)
            return energy_fn

    Returns:
        Returns a function to predict the potential energy from a molecular graph.

    """

    @apply_model(model)
    def potential_energy(features, **kwargs) -> Array:
        return features["global_features"][feature_number]
    return potential_energy


def partial_charge_prediction(model: PropertyPredictor = None,
                              feature_number: int = 1,
                              total_charge: Array = 0.0,
                              ) -> SinglePropertyPredictor:
    """Initializes a prediction of partial charges.

    For a usage with or without model, see the decorator
    :func:`apply_model`.

    Args:
        model: Model extracting particle properties. If not provided, the
            global and per-atom features must be pre-computed.
        feature_number: Number of the per-atom features to base the prediction on.
        total_charge: Total charge of the system. By default, the system should
            be charge neutral.

    Returns:
        Returns a function to predict partial charges from a molecular graph.

    """
    @apply_model(model)
    def partial_charge(features, **kwargs) -> Array:
        raw_partial_charges = features["per_atom_features"][:, feature_number]

        # If masked particles are present, remove their partial charges
        mask = kwargs.get("mask", jnp.ones_like(raw_partial_charges))

        # Correct the charges to ensure charge_neutrality
        charge_correction = jnp.sum(raw_partial_charges * mask)
        charge_correction -= total_charge
        charge_correction /= jnp.sum(mask)

        partial_charges = (raw_partial_charges - charge_correction) * mask
        return partial_charges
    return partial_charge


def init_dipole_moment(displacement_fn: Callable,
                       model: PropertyPredictor = None,
                       graph_from_neighbor_list: Callable = None,
                       partial_charge_feature: int = 1,
                       reference_position_fn: Callable = None,
                       features_key: str = "features"):
    """Computes the dipole moment from partial charges of a molecule.

    Args:
        displacement_fn: Function to compute displacement between reference
            point and particle positions.
        model: Model to predict the partial charges.
        graph_from_neighbor_list: Function to create molecular graph from
            neighbor list.
        partial_charge_feature: Number of the per-atom feature corresponding
            to the partial charge.
        reference_position_fn: Function to compute the reference position for
            the dipole moment. If None, the origin of the box is used.
        features_key: Key to the pre-computed features if no model is provided.

    Returns:
        Returns a function to compute dipole moment snapshots.

    """

    # Initialize the partial charge as a snapshot function
    partial_charge_predictor = partial_charge_prediction(
        model, partial_charge_feature)

    partial_charge_snapshot_fn = snapshot_quantity(
        partial_charge_predictor, graph_from_neighbor_list, features_key
    )

    def dipole_moment_snapshot(state, mask=None, **kwargs):
        if mask is None:
            mask = jnp.ones(state.position.shape[0])

        # Use the origin as reference position
        if reference_position_fn is None:
            ref_position = jnp.zeros(state.position.shape[1])
        else:
            ref_position = reference_position_fn(state, **kwargs)

        dynamic_displacement = functools.partial(displacement_fn, **kwargs)

        # Compute the dipole moment with respect to a user-defined reference
        # point.
        partial_charges = partial_charge_snapshot_fn(state, mask=mask, **kwargs)
        displacements = jax.vmap(
            dynamic_displacement, (0, None)
        )(state.position, ref_position)

        moment = jnp.sum(
            partial_charges[:, None] * displacements * mask[:, None],
            axis=0
        )

        return moment
    return dipole_moment_snapshot

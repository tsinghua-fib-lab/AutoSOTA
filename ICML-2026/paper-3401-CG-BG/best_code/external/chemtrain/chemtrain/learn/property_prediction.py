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

"""This module contains methods to learn a molecular properties from features
of a neural network used for potential energy prediction.
"""

from jax import numpy as jnp, vmap

from jax_md_mod.model import sparse_graph

def build_dataset(targets, graph_dataset):
    """Builds dataset in format that is used for data loading and throughout
    property predictions.

    Args:
        targets: Dict containing all targets to be predicted. Can be retrieved
            in error_fn under the respective key.
        graph_dataset: Dataset of graphs, e.g. as obtained from
            :func:`jax_md_mod.model.sparse_graph.convert_dataset_to_graphs`.

    Returns:
        A dictionary containing the combined dataset and a list of target keys
    """
    target_keys = list(targets)
    target_keys.append('species_mask')
    return {**targets, **graph_dataset.to_dict()}, target_keys


def init_model(prediction_model):
    """Initializes a model that returns predictions for a single observation."""
    def mol_prediction_model(params, observation):
        graph = sparse_graph.SparseDirectionalGraph.from_dict(observation)
        predictions = prediction_model(params, graph)
        return predictions
    return mol_prediction_model


def init_loss_fn(error_fn):
    """Returns a loss function to optimize model parameters.

    Signature of error_fn::

       error = error_fn(predictions, batch, mask)

    where mask has the same shape as species to mask padded particles.

    Args:
        model: Molecular property prediction model (Haiku apply_fn).
        error_fn: Error model quantifying the discrepancy between predictions
            and respective targets.
    """
    def loss_fn(predictions, batch):
        mask = jnp.ones_like(predictions) * batch['species_mask']
        return error_fn(predictions, batch, mask)
    return loss_fn


def per_species_results(species, per_atom_quantities, species_idxs):
    """Sorts per-atom results by species and returns a per-species mean.

    Only real (non-masked) particles should be input.

    Args:
        species: An array storing for each atom the corresponding species.
        per_atom_quantities: An array with the same shape as species, storing
                             per-atom quantities to be evaluated per-species.
        species_idxs: A (species,) array storing species-types for evaluation,
                      e.g. jnp.unique(species).

    Returns:
        A (species,) array of per-species quantities.
    """
    @vmap
    def process_single_species(species_idx):
        species_mask = (species == species_idx)
        species_members = jnp.count_nonzero(species_mask)
        screened_results = jnp.where(species_mask, per_atom_quantities, 0.)
        mean_if_species_exists = jnp.sum(screened_results) / species_members
        return jnp.where(species_members == 0, 0., mean_if_species_exists)
    return process_single_species(species_idxs)


def per_species_box_errors(dataset, per_atom_errors):
    """Computes for each snapshot in the provided graph dataset, the per-species
    error.

    Args:
        dataset: Graph dataset containing the snapshots of interest.
        per_atom_errors: Per-atom error for each atom in the dataset.
            Has same shape as ``dataset['species']``.

    Returns:
        Mean per-species error for each snapshot in the dataset.
    """
    mask = dataset['species_mask']
    species = dataset['species']
    real_species = species[mask]
    unique_species = jnp.unique(real_species)
    species_masked = jnp.where(mask, species, 1000)  # species 1000 nonexistant
    per_box_and_species_fn = vmap(per_species_results, in_axes=(0, 0, None))
    per_box_species_errors = per_box_and_species_fn(
        species_masked, per_atom_errors, unique_species)
    distinct_per_box_species = jnp.sum(per_box_species_errors > 0., axis=1)
    mean_per_box_species_errors = (jnp.sum(per_box_species_errors, axis=1)
                                   / distinct_per_box_species)
    return mean_per_box_species_errors

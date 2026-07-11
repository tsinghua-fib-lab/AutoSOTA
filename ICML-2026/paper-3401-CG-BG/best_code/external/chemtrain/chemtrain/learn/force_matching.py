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

"""Functions for direct learning of per-snapshot quantities.

Directly learnable quantities are, for example, energy, forces, or virial
pressure.

"""
import collections
from typing import Callable, TypedDict, Tuple, List, Dict, DefaultDict, Union
from typing_extensions import Required

from jax import vmap, value_and_grad, numpy as jnp

from jax_sgmc.data.numpy_loader import NumpyDataLoader

from chemtrain.learn import max_likelihood
from chemtrain.ensemble import evaluation
from jax_md_mod import custom_quantity

from chemtrain.typing import EnergyFnTemplate, ArrayLike, NeighborList, ErrorFn, \
    ComputeFn


# Note:
#  Computing the neighbor list in each snapshot is not efficient for DimeNet++,
#  which constructs a sparse graph representation afterward. However, other
#  models such as the tabulated potential are inefficient if used without
#  neighbor list as many cut-off interactions are otherwise computed.
#  For the sake of a simpler implementation, the slight inefficiency
#  in the case of DimeNet++ is accepted for now.


class AtomisticDataset(TypedDict, total=False):
    """Atomistic data for force-matching.

    Args:
        R: Particle positions
        U: Potential energies
        F: Forces
        p: Pressures
        kT: Temperatures
    """
    R: Required[ArrayLike]
    U: ArrayLike
    F: ArrayLike
    p: ArrayLike
    kT: ArrayLike


def build_dataset(position_data: ArrayLike,
                  energy_data: ArrayLike = None,
                  force_data: ArrayLike = None,
                  virial_data: ArrayLike = None,
                  kt_data: ArrayLike = None,
                  **extra_data) -> Tuple[AtomisticDataset]:
    """Builds the force-matching dataset depending on available data.

    Example:

        For force matching, the reference data constist of particle positions
        and target forces.

        >>> from chemtrain.learn.force_matching import build_dataset
        >>> position_data = [...]
        >>> force_data = [...]

        The dataset for training is can be created via:

        >>> dataset = build_dataset(
        ...     position_data=position_data, force_data=force_data)
        >>> print(dataset)
        {'R': [Ellipsis], 'F': [Ellipsis]}

    Args:
        position_data: Reference particle positions
        energy_data: Reference potential energies
        force_data: Reference forces
        virial_data: Reference virials
        kt_data: Reference temperatures

    Returns:
        Returns the canonicalized dataset and a list of keys specifying the
        trainable targets.

    """
    dataset = {'R': position_data}
    if energy_data is not None:
        dataset['U'] = energy_data
    if force_data is not None:
        dataset['F'] = force_data
    if virial_data is not None:
        dataset['p'] = virial_data
    if kt_data is not None:
        dataset['kT'] = kt_data

    dataset.update(extra_data)

    return dataset


def _split_targets_inputs(observation, quantities):
    dynamic_kwargs, targets = {}, {}
    for key in observation.keys():
        if key in quantities:
            targets[key] = observation[key]
        else:
            dynamic_kwargs[key] = observation[key]

    assert set(quantities.keys()) == set(targets.keys()), (
        'All trainig targets must be present in the observation data.'
    )

    return dynamic_kwargs, targets


def state_from_positions(input_dict: Dict[str, ArrayLike]):
    """Extracts the state of the system from the particle positions.

    Args:
        input_dict: Dictionary containing particle positions under 'R'.

    Returns:
        State of the system.
    """
    state = evaluation.SimpleState(input_dict.pop('R'))
    return state, input_dict


# TODO: Initialize predictions for all kinds of quantities

def init_model(nbrs_init: NeighborList,
               quantities: Dict[str, ComputeFn],
               state_from_input: Callable = None,
               feature_extract_fns: Dict[str, Callable] = None):
    """Initialize prediction function for a single snapshot.

    The prediction function computed the energy, force, and virial (if provided)
    based on a single conformation and returns the results in a canonical
    format.

    Note:
        The prediction function does not check whether the neighbor list
        overflowed.

    Args:
        nbrs_init: Initial neighbor list.
        quantities: Dictionary of snapshot functions, e.g., energy and forces.
        state_from_input: Function to build a system state from the input data.
            Not necessary, if the state is already a key in the observations.
        feature_extract_fns: Additional quantities, computed before the
            snapshots and available to all snapshot compute functions.

    Returns:
        Returns a function that computes snapshots given energy parameters and
        observations (inputs).

    """
    if feature_extract_fns is None:
        feature_extract_fns = {}
    if state_from_input is None:
        state_from_input = state_from_positions

    def fm_model(energy_params, observations):
        # Remove default arguments if not provided in dataset
        if 'F' not in observations.keys():
            quantities.pop('F', None)
        if 'U' not in observations.keys():
            quantities.pop('U', None)

        dynamic_kwargs, _ = _split_targets_inputs(observations, quantities)

        # Provides the possibility to add a more detailed state of the
        # system, i.e., with velocities, box, etc.
        if 'state' in dynamic_kwargs:
            states = dynamic_kwargs.pop('state')
        else:
            states, dynamic_kwargs = vmap(state_from_input)(dynamic_kwargs)

        batch_size = states.position.shape[0]

        predictions = evaluation.quantity_map(
            states, quantities, nbrs_init, dynamic_kwargs, {},
            energy_params, batch_size, feature_extract_fns
        )

        return predictions
    return fm_model


def init_loss_fn(error_fns: Union[ErrorFn, dict[str, ErrorFn]] = None,
                 individual: bool = True,
                 gammas: dict[str, float] = None,
                 weights_keys: Dict[str, str] = None):
    """Initializes loss function for energy/force matching.

    Args:
        error_fns: Functions quantifying the deviation of the model and the
            targets. By default, mean-squared error functions.
        individual: Return the loss values for the individual targets, e.g., for
            testing purposes. If False, the loss function returns a scalar loss
            value from the individual loss contributions, weighted by the
            ``gamma_`` coefficients.
        gammas: Weights for the per-target losses in the total loss.
        weights_keys: Dictionary specifying weight keys in the dataset for
            individual targets. The weights determine the per-sample
            contribution for the specific target.

    Returns:
        Returns a function ``loss_fn(predictions, targets)``, which returns a
        scalar loss value for a batch of predictions and targets.
    """
    if gammas is None:
        gammas = {}
    if weights_keys is None:
        weights_keys = {}

    # Preserve old behaviour if error_fns is a single function.
    if isinstance(error_fns, collections.abc.Callable):
        _error_fns = collections.defaultdict(lambda: error_fns)
    else:
        _error_fns = collections.defaultdict(lambda: max_likelihood.mse_loss)

    if isinstance(error_fns, dict):
        _error_fns.update(error_fns)

    def loss_fn(predictions, targets):
        errors = {}
        loss_val = 0.

        # Always present.
        if 'U' in targets.keys():
            weights = targets.get(weights_keys.get('U'))
            errors['U'] = _error_fns['U'](predictions['U'], targets['U'], weights=weights)
            loss_val += gammas.get('U', 1.0) * errors['U']
        if 'F' in targets.keys():
            weights = targets.get(weights_keys.get('F'))
            errors['F'] = _error_fns['F'](predictions['F'], targets['F'], weights=weights)
            loss_val += gammas.get('F', 1.0) * errors['F']

        for key, gamma in gammas.items():
            if key in ['U', 'F']: continue

            weights = None
            if key in weights_keys.keys():
                weights = targets[weights_keys[key]]

            errors[key] = _error_fns[key](predictions[key], targets[key], weights=weights)
            loss_val += gamma * errors[key]

        if individual:
            return loss_val, errors
        else:
            return loss_val

    return loss_fn

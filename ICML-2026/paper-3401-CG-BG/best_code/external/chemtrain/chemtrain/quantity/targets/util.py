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

"""Create quantity targets for training. """
import functools

from jax.typing import ArrayLike

from jax_md.partition import NeighborList

from typing import TypedDict, Callable, Dict, Tuple, Protocol, List
from chemtrain.typing import ComputeFn, EnergyFnTemplate, TargetDict

InitReturn = Tuple[Dict[str, TargetDict], Dict[str, ComputeFn]]
ComputeFnInit = Callable[..., ComputeFn]
ComputeFnInitDict = Dict[str, Callable[..., ComputeFnInit]]
ComputeFnDict = Dict[str, ComputeFn]

class TargetInit(Protocol):

    required_args: List[str]
    optional_args: List[str]

    def __call__(self,
                 key,
                 target_dicts,
                 compute_fns,
                 system) -> Tuple[TargetDict, ComputeFnDict]:
        ...


class InitArguments(TypedDict, total=False):
    """Non-exhaustive dictionary of arguments to initialize the compute functions.

    Arguments:
        displacement_fn: Function to compute the displacement from particles.
            Initialized via :mod:`jax_md.space`.
        energy_fn_template: Template to build energy function from a set of
            parameters.
        r_init: Initial positions of the system, e.g., to infer the number of
            particles in the system.
        nbrs_init: Initial neighbor list of the system, e.g., to infer the
            number of close triplets in the system.
        reference_box: Reference box of the system.
        kT: Fixed temperature of the system.
        pressure: Fixed pressure of the system

    """
    displacement_fn: Callable
    energy_fn_template: EnergyFnTemplate
    nbrs_init: NeighborList
    r_init: ArrayLike
    reference_box: ArrayLike
    kT: ArrayLike
    pressure: ArrayLike


class TargetBuilder:
    """Class to simplify the initialization of DiffTRe targets.

    For each added target, the target builder will check whether compute
    functions for all required snapshots are provided.
    Additionally, the target builder delays the initialization of the compute
    functions until all information about the system is available.

    Args:
        system: Dictionary containing properties of the system, e.g., the
            displacement between particles.
        strict: If True, the target builder will check whether all required
            system properties are provided for each target. Otherwise,
            the properties of the system can be provided when building the
            targets.

    Example:

        First, we add the RDF function to the target builder:

        .. code::

              target_builder = TargetBuilder()

              target_builder["rdf"] = quantity.structure.init_radial_distribution_target(
                  target, rdf_start=0.00, rdf_cut=1.0, nbins=50
              )

        Then, as soon as the box and displacement function of the system are known,
        the compute functions and observables can be initialized:

        .. code::

           targets, compute_fns = target_builder.build({
               'displacement_fn': displacement_fn,
               'reference_box': box
           })

    """

    def __init__(self,
                 system: InitArguments = None,
                 strict: bool = False):
        if system is None:
            system = {}

        self._strict = strict
        self._system = system
        self._targets = {}

    def __setitem__(self, key, value: TargetInit):
        assert key not in self._targets, (
            f"Duplicate target {key}."
        )

        if self._strict:
            for required_arg in value.required_args:
                assert required_arg in self._system.keys(), (
                    f"Required argument {required_arg} not provided."
                )

        self._targets[key] = value

    def build(self,
              system: InitArguments = None,
              ) -> Tuple[TargetDict, ComputeFnDict]:
        """Initializes the targets and compute functions.

        Args:
            system: Dictionary containing properties of the system, e.g., the
                displacement between particles.

        Returns:
            Returns a dictionary of observables and a dictionary of functions
            to compute instantaneous properties from the simulator states.

        """
        if system is not None:
            self._system.update(system)

        target_dicts = {}
        compute_fns = {}
        for key, init_fn in self._targets.items():
            target_dicts, compute_fns = init_fn(
                key, target_dicts, compute_fns, self._system)
        return target_dicts, compute_fns


def split_target_dict(target_dict: TargetDict
                      ) -> Tuple[Dict[str, Callable],
                                 Dict[str, Callable],
                                 Dict[str, Dict[str, ArrayLike]]]:
    """Splits the target dictionary into observable functions, loss functions,
    and target values.

    Args:
        target_dict: Dictionary of targets.

    Returns:
        Returns a tuple of observable functions, loss functions, and target
        values.

    """
    observables = {
        key: target['traj_fn'] for key, target in target_dict.items()
    }
    target_loss_fns = {
        key: target['loss_fn'] for key, target in target_dict.items()
        if 'loss_fn' in target
    }
    targets = {
        key: {k: v for k, v in target.items() if k in ['gamma', 'target']}
        for key, target in target_dict.items()
    }

    return observables, target_loss_fns, targets


def target_quantity(required: list = None, optional: list = None):
    """Initializes a decorator defining required and optional arguments to
    initialize a target.
    """

    if required is None:
        required = []
    if optional is None:
        optional = []

    def quantity_init_wrapper(init_fn):
        @functools.wraps(init_fn)
        def quantity_init(key: str,
                          targets_dict: TargetDict,
                          compute_fns: ComputeFnInitDict,
                          init_args: InitArguments
                          ) -> Tuple[TargetDict,  ComputeFnDict]:
            # Assert that all required arguments are provided and collect all
            # optional arguments
            init_kwargs = {
                init_key: init_args[init_key] for init_key in required
            }
            optional_kwargs = {
                init_key: value for init_key in optional
                if (value := init_args.get(init_key)) is not None
            }
            init_kwargs.update(optional_kwargs)

            # Initialize the compute function and target function with the
            # provided arguments
            target_dict, compute_fn = init_fn(
                key, compute_fns, init_kwargs)

            # Add the initialized compute function and target dict
            if target_dict is not None:
                assert key not in target_dict.keys(), (
                    f"Duplicate target {key} provided.")
                targets_dict.update({key: target_dict})
            if compute_fn is not None:
                assert key not in compute_fns.keys(), (
                    f"Duplicate quantitiy {key} initialized.")
                compute_fns.update({key: compute_fn})

            return targets_dict, compute_fns

        # Enables checking whether system arguments are already provided
        quantity_init.required_args = required
        quantity_init.optional_args = optional
        return quantity_init

    return quantity_init_wrapper

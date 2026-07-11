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

"""Initialize thermodynamic quantities."""

__all__ = (
    "init_pressure_target",
    "init_reference_energy_target",
    "init_relative_entropy_target",
    "init_volume_target",
    "init_heat_capacity_nvt",
    "init_heat_capacity_npt"
)


from jax_md_mod import custom_quantity

from chemtrain.learn import max_likelihood
from chemtrain.quantity import observables
from chemtrain.typing import ArrayLike, EnergyFnTemplate, Any

from .util import target_quantity, TargetInit

def init_pressure_target(energy_fn_template: EnergyFnTemplate,
                         include_kinetic: bool = True,
                         gamma: float = 1.0e-7,
                         target: float = 0.06022137
                         ) -> TargetInit:
    """Initializes pressure target."""

    @target_quantity([], ['reference_box'])
    def initialize(key, compute_fns, init_args):
        compute_fn = custom_quantity.init_pressure(
            energy_fn_template=energy_fn_template,
            include_kinetic=include_kinetic, **init_args)

        target_dict = {
            'target': target, 'gamma': gamma,
            'traj_fn': observables.init_traj_mean_fn(key)
        }
        return target_dict, compute_fn
    return initialize


def init_reference_energy_target(energy_fn_template: EnergyFnTemplate,
                                 energy_params: Any = None,
                                 gamma: float = 1.0,
                                 target_energy: ArrayLike = None
                                 ) -> TargetInit:
    """Initializes the computation of an energy for a trajectory.

    TODO: Simplify the initialization...

    """
    @target_quantity()
    def initialize(key, compute_fns, init_args):
        del init_args, compute_fns

        compute_fn = custom_quantity.energy_wrapper(
            energy_fn_template=energy_fn_template,
            fixed_energy_params=energy_params)

        # Skip of not target provided
        if target_energy is None:
            return None, compute_fn

        target_dict = {
            'target': target_energy, 'gamma': gamma,
            'traj_fn': observables.init_traj_mean_fn(key)
        }
        return target_dict, compute_fn
    return initialize


def init_relative_entropy_target(reference_energy_key: str,
                                 ref_kT = None,
                                 gamma: float = 1.0,
                                 target: float = 0.0,
                                 ) -> TargetInit:
    """Initializes the computation of a relative entropy.

     Args:
         reference_energy_key: Key of the reference potential.
         ref_kT: Reference temperature. Value of 1.0 corresponds to the
             relative entropy of information theory.
         gamma: Scale constant for the target. A positive value maximizes
             the negative relative entropy.
         target: Target in the loss.

     """
    @target_quantity(optional=['reference_kT'])
    def initialize(key, compute_fns, init_args):
        assert reference_energy_key in compute_fns.keys(), (
            f"Computing the entropy requires a reference energy, but no "
            f"compute function for the quantity {reference_energy_key} "
            f"was found.")

        nonlocal ref_kT
        if ref_kT is None:
            ref_kT = init_args.get('reference_kT', 1.0)

        # Flip the sign of gamma to maximize the relative entropy
        target_dict = {
            'target': target, 'gamma': - 1.0 * gamma,
            'traj_fn': observables.init_relative_entropy_traj_fn(
                ref_kT, reference_key=reference_energy_key
            ), 'loss_fn': max_likelihood.identity_loss
        }

        return target_dict, None
    return initialize

def init_volume_target(gamma: float = 1.0,
                       target: float = None,
                       ) -> TargetInit:

    @target_quantity()
    def initialize(key, compute_fns, init_args):
        assert key == 'volume', (
            f"Please initialize the volume with the key 'volume' "
            f"instead using the provided key '{key}'.")

        if target is not None:
            target_dict = {
                'target': target, 'gamma': gamma,
                'traj_fn': observables.init_traj_mean_fn(key)
            }
        else:
            target_dict = None

        compute_fns = custom_quantity.volume_npt

        return target_dict, compute_fns
    return initialize

def init_heat_capacity_nvt(gamma: float = 1.0,
                           target: float = 1.0,
                           linearized: bool = False,
                           ) -> TargetInit:
    @target_quantity(['kT'], ['dof', 'r_init'])
    def initialize(key, compute_fns, init_args):
        dof = init_args.get('dof')
        if dof is None:
            assert 'r_init' in init_args.keys(), (
                "Cv Requires one of 'dof' or 'r_init' as init args.")

            # Get the degrees of freedom via the number of particles
            dof = 3 * init_args['r_init'].shape[-2]

        traj_fn = observables.init_heat_capacity_nvt(
            kbt=init_args['kbt'], dof=dof, linearized=linearized)

        target_dict = {
            'target': target, 'gamma': gamma,
            'traj_fn': traj_fn
        }

        return target_dict, None
    return initialize


def init_heat_capacity_npt(gamma: float = 1.0,
                           target: float = 1.0,
                           linearized: bool = False,
                           ) -> TargetInit:
    @target_quantity(['kT', 'pressure'], ['dof', 'r_init'])
    def initialize(key, compute_fns, init_args):
        dof = init_args.pop('dof', None)
        r_init = init_args.pop('r_init', None)
        if dof is None:
            assert r_init is not None, (
                "Cv Requires one of 'dof' or 'r_init' as init args.")

            # Get the degrees of freedom via the number of particles
            dof = 3 * r_init.shape[-2]
        init_args['dof'] = dof

        traj_fn = observables.init_heat_capacity_nvt(
            **init_args, linearized=linearized)

        target_dict = {
            'target': target, 'gamma': gamma,
            'traj_fn': traj_fn
        }

        return target_dict, None
    return initialize
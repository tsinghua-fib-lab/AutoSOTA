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

"""Molecular dynamics observable functions acting on trajectories rather than
single snapshots.

Builds on the TrajectoryState object defined in traj_util.py.
"""
import functools

from jax import vmap, numpy as jnp, lax
from jax_md import simulate, quantity

from chemtrain.quantity import constants

from typing import Dict
from jax.typing import ArrayLike
from chemtrain.typing import TrajFn


def dynamic_statepoint(keys_with_defaults: Dict[str, ArrayLike] = None):
    """Initializes a decorator enabeling a dynamically provided statepoint definition.

    Args:
        keys_with_defaults: List of keys that can be provided dynamically with
            default values.

    Returns:
        Returns a decorator to filter the dynamically provided arguments.

    """
    if keys_with_defaults is None:
        keys_with_defaults = {}

    def decorator(obs_fn):
        @functools.wraps(obs_fn)
        def wrapper(quantity_trajs, weights=None, **state_kwargs):
            if len(keys_with_defaults) == 0:
                return obs_fn(quantity_trajs, weights)

            state_dict = {
                key: state_kwargs.pop(key, default)
                for key, default in keys_with_defaults.items()
            }

            return obs_fn(quantity_trajs, state_dict, weights)
        return wrapper
    return decorator


def init_identity_fn(quantity_key) -> TrajFn:
    """Initialize a wrapper that returns the snapshots unchanged.

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns the snapshots of the quantity.

    """

    @dynamic_statepoint()
    def identity(quantity_trajs, weights=None):
        return quantity_trajs[quantity_key]

    return identity


def init_traj_mean_fn(quantity_key) -> TrajFn:
    """Initialize the computation of a (perturbed) ensemble average.

    This function builds the ``'traj_fn'`` of the DiffTRe ``target`` dict for the
    common case of target observables that simply consist of a
    trajectory-average of instantaneous quantities, such as RDF, ADF, pressure
    or density.

    .. math::

       a = \\left\\langle w(\\mathbf r) a(\\mathbf r)\\right\\rangle_{\\tilde U},\\quad \\text{where} \\\\
       w(\\mathbf r) = \\frac{e^{-\\beta (U(\\mathbf r) - \\tilde U(\\mathbf r))}}{\\left\\langle e^{-\\beta (U(\\mathbf r) - \\tilde U(\\mathbf r))}\\right\\rangle}

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns a function to compute an ensemble average when given a
        dictionary of snapshots.
        Returns a perturbed ensemble average if weights are provided.

    """
    @dynamic_statepoint()
    def traj_mean(quantity_trajs, weights=None):
        quantity_traj = quantity_trajs[quantity_key]
        return _traj_mean(quantity_traj, weights, linearized=False)
    return traj_mean


def _weighted_mean(quantity_traj, weights):
    if weights is not None:
        weights *= weights.size
        quantity_traj = (quantity_traj.T * weights).T
    return jnp.mean(quantity_traj, axis=0)


def _linearized_mean(quantity_traj, weights):
    # Mask out zero weights
    mask = weights > 1e-30
    weights = jnp.where(mask, weights, 1.0)
    n_nonzero = jnp.sum(mask)

    # Calculate the masked means
    quantity_mean = jnp.mean(quantity_traj, axis=0)
    correction = jnp.sum((quantity_traj.T * jnp.log(weights) * mask).T, axis=0)
    correction -= quantity_mean * jnp.sum(jnp.log(weights) * mask)
    return quantity_mean + correction / n_nonzero


def _traj_mean(quantity_traj, weights=None, linearized=False, **kwargs):
    del kwargs

    if linearized:
        assert weights is not None, (
            "The linearized trajectory average requires weights to be "
            "specified.")
        return _linearized_mean(quantity_traj, weights)
    else:
        return _weighted_mean(quantity_traj, weights)


def init_linear_traj_mean_fn(quantity_key) -> TrajFn:
    """Initializes the linear approximation of an observable average.

    This function approximates ensemble averages of instantaneous quantities
    for a perturbed ensemble based on a cumulant expansion [#imbalzano2021]_.


    .. [#imbalzano2021] Giulio Imbalzano, Yongbin Zhuang, Venkat Kapil,
       Kevin Rossi, Edgar A. Engel, Federico Grasselli, Michele Ceriotti;
       _Uncertainty estimation for molecular dynamics and sampling._
       J. Chem. Phys. 21 February 2021; 154 (7): 074102.
       https://doi.org/10.1063/5.0036522

    Args:
        quantity_key: Key referencing the quantity in the computed snapshots.

    Returns:
        Returns a function to approximate a perturbed ensemble average when
        given a dictionary of snapshots and weights.

    """
    @dynamic_statepoint()
    def traj_mean(quantity_trajs, weights=None):
        quantity_traj = quantity_trajs[quantity_key]
        return _traj_mean(quantity_traj, weights, linearized=True)
    return traj_mean


def init_relative_entropy_traj_fn(kT: float, reference_key = 'ref_energy') -> TrajFn:
    r"""Initializes the computation of the relative entropy difference between
    the current canonical distribution and reference distribution.

    The relative entropy is given as

    .. math::
        S_\text{rel} = -\int p(x)\log\frac{p(x)}{q(x)}dx.

    For two canonical distributions defined by the potentials U_p and U_q,
    this relative entropy computes as

    .. math::

        S_\text{rel} = \beta\left\langle\left(U_p - U_q\right)\right\rangle_{U_p} - \log\left\langle e^{-\beta(U_q - U_p)}\right\rangle_{U_p}.

    Args:
        kT: Reference temperature.
        reference_key: Key corresponding to the reference energy quantity.

    Returns:
        Returns a function that computes the relative entropy given a
        dictionary of quantities along a trajectory.

    """
    @dynamic_statepoint({'kT': kT})
    def relative_entropy_traj_fn(quantity_trajs, state_dict, weights=None):
        beta = 1.0 / state_dict['kT']

        energies = quantity_trajs['energy']
        ref_energies = quantity_trajs[reference_key]

        if weights is None:
            weights = jnp.ones_like(energies) / energies.size

        energy_difference = jnp.sum(weights * (energies - ref_energies))

        # Compute the free energy from the current to the reference potential
        # via the forward perturbation and reverse the sign
        exponents = -beta * (ref_energies - energies)
        max_exponent = jnp.max(exponents)
        exp = jnp.exp(exponents - max_exponent)

        # We use this method of free energy calculation only for the value but
        # not for the gradient.
        # The quality of the estimation via the perturbation formula depends
        # on the difference between the potentials, which is perhaps large.
        # Instead, the gradients of the free energy are quite easily computable
        # with respect to any other reference state.

        free_energy_difference = jnp.log(jnp.sum(weights * exp))
        free_energy_difference += max_exponent
        free_energy_difference *= -1.0 / beta
        free_energy_difference = lax.stop_gradient(free_energy_difference)

        # By using stop_gradient, we can ensure the contribution to the gradient
        # without a contribution to the absolute value.
        free_energy_difference -= quantity_trajs['free_energy']
        free_energy_difference += lax.stop_gradient(
            quantity_trajs['free_energy'])

        entropy_difference = energy_difference + free_energy_difference
        entropy_difference *= beta * constants.kb

        return entropy_difference
    return relative_entropy_traj_fn


def init_free_energy(kT, pressure=None, **kwargs):
    # TODO: Documentation

    assert pressure is None, (
        "The free energy calculation is not yet implemented for the NPT ensemble."
    )

    @dynamic_statepoint({'kT': kT})
    def compute_fn(quantity_trajs, state_dict, weights=None):
        # Free energy already computed
        logZ = quantity_trajs['free_energy']
        free_energy = -state_dict['kT'] * logZ
        return free_energy

    return compute_fn


def init_heat_capacity_nvt(kT, dof, **kwargs):
    """Returns the specific heat capacity of a system in the NPT ensemble via
     the fluctuation formula.

     The specific heat capacity is returned in units of the Boltzmann constant.

    Args:
        kT: Reference temperature
        dof: Number of degrees of freedom in the system
        kwargs: Additional arguments determining the calculation of trajectory
            averages.

    Returns:
        Returns a function to compute the (perturbed) specific heat capacity
        based on potential energy snapshots.

    References:
        `<https://journals-aps-org.eaccess.tum.edu/pre/pdf/10.1103/PhysRevE.99.012139>`

    """
    @dynamic_statepoint({'kT': kT, 'dof': dof})
    def specific_heat_capacity_fn(quantity_trajs, state_dict, weights=None):
        energy = quantity_trajs['energy']
        if weights is None:
            weights = jnp.ones_like(energy) / energy.size

        fluctuation = _traj_mean(energy ** 2, weights, **kwargs)
        fluctuation -= _traj_mean(energy, weights, **kwargs) ** 2

        c_v = (0.5 * state_dict['dof'] + fluctuation / state_dict['kT'] ** 2)
        return c_v
    return specific_heat_capacity_fn


def init_heat_capacity_npt(kT, dof, pressure, **kwargs):
    """Initializes the isobaric heat capacity of a system in the NPT ensemble.

    The heat capacity of the system depends on the fluctuation of the
    conformational enthalpy [#stroeker2021]_.

    Args:
        kT: Thermostat temperature
        dof: Number of degrees of freedom in the system
        kwargs: Extra arguments to the function computing the ensemble
            averages

    Returns:
        Returns a function to compute the (perturbed) specific heat capacity
        based on potential energy snapshots and volume snapshots.

    References:
        .. [#stroeker2021] P. Ströker, R. Hellmann, und K. Meier, „Systematic formulation of thermodynamic properties in the N p T ensemble“, Phys. Rev. E, Bd. 103, Nr. 2, S. 023305, Feb. 2021, doi: 10.1103/PhysRevE.103.023305.


    """
    @dynamic_statepoint({'kT': kT, 'pressure': pressure, 'dof': dof})
    def cp_fn(quantity_traj, state_dict, weights=None):
        volume_traj = quantity_traj['volume']
        energy_traj = quantity_traj['energy']

        enthalpy = energy_traj + volume_traj * state_dict['pressure']

        fluctuation = _traj_mean(enthalpy ** 2, weights, **kwargs)
        fluctuation -= _traj_mean(enthalpy, weights, **kwargs) ** 2

        cp = (0.5 * state_dict['dof'] + fluctuation / state_dict['kT'] ** 2)
        return cp
    return cp_fn


def init_born_stiffness_tensor(reference_box, dof, kT, elastic_constant_function=None):
    """Computes the elastic stiffness tensor via the stress fluctuation method
    in the NVT ensemble.

    Args:
        reference_box: Default box used to compute the volume
        dof: Default number of degrees of freedom
        kT: Default temperature
        elastic_constant_function: Function to compute the elastic constants
            from the stiffness tensor. If None, the stiffness tensor is returned.

    Returns:
        born_stiffness: Either the stiffness tenosr or the elastic constants
        if the elastic_constant_function is provided.
    """

    def stiffness_tensor_fn(mean_born, mean_sigma, mean_sig_ij_sig_kl, state_dict):
        """Computes the stiffness tensor given ensemble averages of
        C^B_ijkl, sigma^B_ij and sigma^B_ij * sigma^B_kl.
        """
        spatial_dim = mean_born.shape[0]
        volume = quantity.volume(spatial_dim, state_dict['box'])

        delta_ij = jnp.eye(spatial_dim)
        delta_ik_delta_jl = jnp.einsum('ik,jl->ijkl', delta_ij, delta_ij)
        delta_il_delta_jk = jnp.einsum('il,jk->ijkl', delta_ij, delta_ij)
        sigma_prod = jnp.einsum('ij,kl->ijkl', mean_sigma, mean_sigma)

        # Note: maybe use real kinetic energy of trajectory rather than target
        #       kbt?
        kinetic_term = delta_ik_delta_jl + delta_il_delta_jk
        kinetic_term *= state_dict['dof'] * state_dict['kT']
        kinetic_term /= spatial_dim * volume

        sigma_term = mean_sig_ij_sig_kl - sigma_prod
        sigma_term *= volume / state_dict['kT']

        return mean_born - sigma_term + kinetic_term

    @dynamic_statepoint({'kT': kT, 'dof': dof, 'box': reference_box})
    def born_stiffness(quantity_trajs, state_dict, weights=None):
        sigma = quantity_trajs['born_stress']
        born_stress_prods = jnp.einsum('nij,nkl->nijkl', sigma, sigma)

        stress_product_born_mean = _weighted_mean(born_stress_prods, weights)
        born_stress_tensor_mean = _weighted_mean(
            quantity_trajs['born_stress'], weights)
        born_stiffness_mean = _weighted_mean(
            quantity_trajs['born_stiffness'], weights)

        stiffness_tensor = stiffness_tensor_fn(
            born_stiffness_mean, born_stress_tensor_mean,
            stress_product_born_mean, state_dict)

        if elastic_constant_function is None:
            return stiffness_tensor
        else:
            return elastic_constant_function(stiffness_tensor)

    return born_stiffness


def stiffness_tensor_components_cubic_crystal(stiffness_tensor):
    """Computes the 3 independent elastic stiffness components of a cubic
    crystal from the whole stiffness tensor.

    The number of independent components in a general stiffness tensor is 21
    for isotropic pressure. For a cubic crystal, these 21 parameters only take
    3 distinct values: c11, c12 and c44. We compute these values from averages
    using all 21 components for variance reduction purposes.

    Args:
        stiffness_tensor: The full (3, 3, 3, 3) elastic stiffness tensor

    Returns:
        A (3,) ndarray containing (c11, c12, c44)
    """
    # TODO likely there exists a better formulation via Einstein notation
    c = stiffness_tensor
    c11 = (c[0, 0, 0, 0] + c[1, 1, 1, 1] + c[2, 2, 2, 2]) / 3.
    c12 = (c[0, 0, 1, 1] + c[1, 1, 0, 0] + c[0, 0, 2, 2] + c[2, 2, 0, 0]
           + c[1, 1, 2, 2] + c[2, 2, 1, 1]) / 6.
    c44 = (c[0, 1, 0, 1] + c[1, 0, 0, 1] + c[0, 1, 1, 0] + c[1, 0, 1, 0] +
           c[0, 2, 0, 2] + c[2, 0, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 2, 0] +
           c[2, 1, 2, 1] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[1, 2, 1, 2]) / 12.
    return jnp.array([c11, c12, c44])


def stiffness_tensor_components_hexagonal_crystal(stiffness_tensor):
    """ Computes 5 independent elastic stiff components of a hexagonal
    crystal from the whole stiffness tensor

    For a hexagonal structure the 5 components are c11, c33, c44, c12, c13

    Args:
        stiffness_tensor: The full(3,3,3,3) elastic stiffness tensor

    Returns:
          A (5,) ndarray containing (c11, c33, c44, c12, c13)
        """
    c = stiffness_tensor
    c11 = (c[0, 0, 0, 0] + c[1, 1, 1, 1]) / 2.
    c33 = c[2, 2, 2, 2]
    c44 = (c[0, 2, 0, 2] + c[2, 0, 0, 2] + c[0, 2, 2, 0] + c[2, 0, 2, 0] +
           c[2, 1, 2, 1] + c[1, 2, 2, 1] + c[2, 1, 1, 2] + c[1, 2, 1, 2]) / 8.
    c12 = (c[0, 0, 1, 1] + c[1, 1, 0, 0]) / 2.
    c13 = (c[0, 0, 2, 2] + c[2, 2, 0, 0] + c[1, 1, 2, 2] + c[2, 2, 1, 1]) / 4.

    return jnp.array([c11, c33, c44, c12, c13])



# TODO: Update the remaining compute functions (see reference above for npt
#       formulations)

def volumes(traj_state):
    """Returns array of volumes for all boxes in a NPT trajectory.

    Args:
        traj_state: TrajectoryState containing the NPT trajectory
    """
    dim = traj_state.sim_state.sim_state.position.shape[-1]
    boxes = vmap(simulate.npt_box)(traj_state.trajectory)
    return vmap(quantity.volume, (None, 0))(dim, boxes)


def isothermal_compressibility_npt(volume_traj, kbt):
    """Returns isothermal compressibility of a system in the NPT ensemble
    via the fluctuation formula.

    Args:
        volume_traj: Trajectory of box volumes of a NPT simulation,
                     e.g. computed via traj_quantity.volumes.
        kbt: Temperature * Boltzmann constant
    """
    mean_volume = jnp.mean(volume_traj)
    kappa = (jnp.mean(volume_traj**2) - mean_volume**2) / (mean_volume * kbt)
    return kappa


def thermal_expansion_coefficient_npt(volume_traj, energy_traj, temperature,
                                      kbt, pressure):
    """Returns the thermal expansion coefficient of a system in the NPT ensemble
    via the fluctuation formula.

    Args:
        volume_traj: Trajectory of box volumes of a NPT simulation,
                     e.g. computed via traj_quantity.volumes.
        energy_traj: Trajectory of potential energies
        temperature: Thermostat temperature
        kbt: Temperature * Boltzmann constant
        pressure: Barostat pressure
    """
    mean_volume = jnp.mean(volume_traj)
    fluctuation = (jnp.mean(volume_traj * energy_traj) - jnp.mean(energy_traj)
                   * mean_volume
                   + pressure * (jnp.mean(volume_traj**2) - mean_volume**2))
    alpha = fluctuation / temperature / kbt / mean_volume
    return alpha

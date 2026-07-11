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

"""Functions build around the Differentiable Trajectory Reweighting (DiffTRe)
algorithm [#Thaler2021]_. The DiffTRe algorithm builds
on umbrella sampling to efficiently compute gradients of ensemble observables.

Chemtrain implements umbrella sampling approaches in the module
:mod:`chemtrain.trajectory.reweighting`.

References:
   .. [#Thaler2021] Thaler, S.; Zavadlav, J. Learning Neural Network
      Potentials from Experimental Data via Differentiable Trajectory
      Reweighting. Nat Commun **2021**, 12 (1), 6884.
      https://doi.org/10.1038/s41467-021-27241-4.
   .. [#Shell2008] Shell, M. S. The Relative Entropy Is Fundamental to
      Multiscale and Inverse Thermodynamic Problems. J. Chem. Phys. 2008,
      129 (14), 144108. https://doi.org/10.1063/1.2992060.

"""

import functools
from typing import Dict, Any, Callable, Tuple

import numpy as onp

import jax
from jax import jit, numpy as jnp, lax
from jax.typing import ArrayLike

from jax_md_mod import custom_quantity

from chemtrain.learn import max_likelihood
from chemtrain.typing import TargetDict, EnergyFnTemplate, ComputeFn
from chemtrain.ensemble import reweighting, evaluation, sampling
from chemtrain import util

from chemtrain.typing import TrajFn


def init_default_loss_fn(observables: Dict[str, TrajFn],
                         loss_fns: Dict[str, Callable]
                         ):
    """Initializes the MSE loss function for DiffTRe.

    The default loss for the Differentiable Trajectory Reweighting (DiffTRe)
    algorithm [#Thaler2021]_ is the mean squared error (MSE) between an observable
    :math:`\\mathcal A` and an (experimental) reference :math:`\hat a`

    .. math::

       \\mathcal L(\\theta) = \\gamma \\left(
           \\hat a - \\mathcal A(\\mathbf w_N, \\mathbf r_N) \\right)^2.

    Some observables are simple ensemble averages of instantaneous quantities
    :math:`a`

    .. math::

       \\mathcal A(\\mathbf w, \\mathbf r^n) =
           \\sum_{n=1}^N w^{(n)} a\\left(\mathbf r^{(n)}\\right).

    However, some quantities, e.g., the heat capacity :math:`c_V`,
    relate to multiple ensemble averages or even multiple quantities.
    Therefore, each target specifies a ``traj_fn`` with access to all
    instantaneous quantities.

    For a list of implemented ensemble observables, refer to the module
    :mod:`chemtrain.quantity.observables`.


    Args:
        observables: Dictionary containing functions to compute ensemble
            observables from snapshots.
        loss_fns: Dictionary containing loss functions for the individual
            targets.

    Returns:
        Returns a DiffTRe loss_fn. The loss function accepts a dictionary of
        snapshots for each sample, the weights for each sample, a dict
        definition properties of the statepoint and the targets of the training.

    """
    def loss_fn(quantity_trajs, weights, state_dict, targets):
        predictions = {
            key: obs_fn(quantity_trajs, weights=weights, **state_dict)
            for key, obs_fn in observables.items()
        }

        # MSE loss for the remaining targets
        loss = 0.
        for target_key, target in targets.items():
            loss_fn = loss_fns.get(target_key, max_likelihood.mse_loss)
            gamma = target.get('gamma', 1.0)

            loss += gamma * loss_fn(predictions[target_key], target['target'])

        return loss, predictions
    return loss_fn


def init_difftre_gradient_and_propagation(
    reweight_fns: Tuple[Callable, Callable, Callable],
    loss_fn,
    quantities: Dict[str, ComputeFn],
    energy_fn_template: EnergyFnTemplate,
    wrapped: bool = True,
    batched: bool = False,
    ):
    """Initializes the function to compute the DiffTRe loss and its gradients.

    The Differentiable Trajectory Reweighing (DiffTRe) algorithm [#Thaler2021]_
    computes gradients of ensemble averages via a perturbation approach,
    initialized in
    :func:`chemtrain.trajectory.reweighting.init_pot_reweight_propagation_fns`.

    Args:
        reweight_fns: Functions to perform and evaluate umbrella-sampling
            simulations.
        loss_fn: DiffTRe compatible loss function, e.g.,
            initialized via :func:`init_default_loss_fn`.
        quantities: Dictionary specifying how to compute instantaneous
            quantities from the simulator states.
        energy_fn_template: Template to initialize the energy function that
            is required to compute the weights.
        wrapped: Return separate weight, propagation, and gradient functions
        batched: Computes loss for multiple ensembles.

    Returns:
        Returns a function to propagate the current trajectory state,
        compute the loss and gradient, and predict observations.

    """

    weights_fn, propagate_fn, safe_propagate = reweight_fns

    quantities['energy'] = custom_quantity.energy_wrapper(
        energy_fn_template)
    reweighting.checkpoint_quantities(quantities)

    def _difftre_loss(params, traj_state, state_dict, targets):
        """Computes the loss using the DiffTRe formalism and
        additionally returns predictions of the current model.
        """
        weights, _, entropy, free_energy = weights_fn(
            params, traj_state, entropy_and_free_energy=True)

        quantity_trajs = sampling.quantity_traj(
            traj_state, quantities, params)
        quantity_trajs.update(entropy=entropy, free_energy=free_energy)

        loss, predictions = loss_fn(
            quantity_trajs, weights, state_dict, targets)

        # Always save free energy and entropy even if they are not part of
        # the loss.
        predictions.update(entropy=entropy, free_energy=free_energy)

        return loss, predictions

    # TODO: Maybe separate the functions like for force matching

    def difftre_weights_fn(params, traj_state, reduction="min"):
        partial_weights = functools.partial(weights_fn, params)

        if not batched:
            return partial_weights(traj_state, reduction=reduction)

        return jax.vmap(partial_weights)(traj_state)

    def difftre_loss_fn(params, traj_state, state_dict, targets):
        partial_loss = functools.partial(_difftre_loss, params)


        if not batched:
            return partial_loss(traj_state, state_dict, targets)

        batched_loss, batched_predictions = jax.vmap(partial_loss)(
            traj_state, state_dict, targets)

        return jnp.mean(batched_loss), batched_predictions

    def difftre_propagation(params, traj_state, state_dict):
        """The main DiffTRe function that recomputes trajectories
        when needed and computes gradients of the loss wrt. energy function
        parameters for a single state point.
        """
        partial_propagation = functools.partial(
            propagate_fn, params, recompute=True)

        if not batched:
            return partial_propagation(traj_state, **state_dict)

        return jax.vmap(partial_propagation)(traj_state, **state_dict)

    if not wrapped:
        return difftre_loss_fn, difftre_propagation, difftre_weights_fn

    assert not batched, "Batched computation requires 'wrapped=False'."

    loss_grad_fn = jax.value_and_grad(difftre_loss_fn, has_aux=True, argnums=0)

    # TODO: There is more opportunity to make this general.
    #       We could extend the propagation and gradient function to take
    #       additional args besides the first two, e.g., batch state and
    #       output additional args besides the two mandatory traj state and
    #       loss grad

    @safe_propagate
    @jit
    def difftre_grad_and_propagation(params, traj_state, state_dict, targets):
        """The main DiffTRe function that recomputes trajectories
        when needed and computes gradients of the loss wrt. energy function
        parameters for a single state point.
        """
        traj_state = propagate_fn(params, traj_state, **state_dict)

        (loss_val, predictions), loss_grad = loss_grad_fn(
            params, traj_state, state_dict, targets)
        return traj_state, loss_val, loss_grad, predictions

    return difftre_grad_and_propagation


def init_rel_entropy_loss_fn(energy_fn_template, compute_weights, kbt, vmap_batch_size=10):
    """Initializes a function to computes the relative entropy loss.

    The relative entropy between a fine-grained (FG) reference system
    with :math:`U^\\mathrm{FG}(\\mathbf r)` coarse-grained (CG) reference system
    with :math:`U_\\theta^\\mathrm{CG}(\\mathbf R)` is [#Shell2008]_

    .. math::

       S_\\text{rel} = \\beta\\langle
          U_\\theta^\\mathrm{CG}(\\mathcal M(\\mathbf r))
          - U^\\mathrm{FG}(\\mathbf r) \\rangle_\\text{FG}
          -\\beta(A^\\mathrm{CG}_\\theta - A^\\mathrm{FG}) + S_\\text{map}.

    This relative entropy depends on the free energies of the models and
    a mapping entropy.

    However, using free-energy perturbation approaches, one can create
    a replacement loss functions that has the same gradients

    .. math::

       \\mathcal L(\\theta) = \\beta \\langle
          U_\\theta^\\mathrm{CG}(\\mathcal M(\\mathbf r))\\rangle_\\text{FG}
          -\\beta A^\\mathrm{CG}_\\theta.


    Args:
        energy_fn_template: Energy function template
        compute_weights: compute_weights function as initialized from
            init_pot_reweight_propagation_fns.
        kbt: Temperature of the statepoint.
        vmap_batch_size: Batch size for computing the potential energies on
            the reference positions.

    Returns:
        A function that returns the relative entropy loss, i.e., the
        contributions to the relative entropy that depend on the parameters
        of the model.

    """

    ref_quantities = {
        "ref_energy": custom_quantity.energy_wrapper(energy_fn_template)
    }

    reweighting.checkpoint_quantities(ref_quantities)

    def loss_fn(params, traj_state, reference_batch):
        # Compute the free energy difference with respect to the initial state
        *_, free_energy = compute_weights(
            params, traj_state, entropy_and_free_energy=True)
        free_energy += traj_state.free_energy_diff

        # Compute the potential predictions on the reference data
        ref_states = evaluation.SimpleState(position=reference_batch['R'])

        nbrs = traj_state.sim_state.nbrs
        if nbrs.reference_position.ndim > 2:
            nbrs = util.tree_get_single(traj_state.sim_state.nbrs)

        ref_energies = evaluation.quantity_map(
            ref_states, ref_quantities, nbrs, {},
            {"kT": kbt}, params, vmap_batch_size,
        )["ref_energy"]

        return (jnp.mean(ref_energies) - free_energy) / kbt

    return loss_fn


def init_rel_entropy_gradient_and_propagation(reference_dataloader,
                                              reweight_fns,
                                              energy_fn_template,
                                              kbt,
                                              vmap_batch_size=10):
    """Initialize function to compute the relative entropy gradients.

    This implementation of the Relative Entropy Minimization algorithm
    [#Shell2008]_ computes the gradients of the free energy similar to the
    Differentiable Trajectory Reweighting (DiffTRe) algorithm [#Thaler2021]_
    via a perturbation approach, initialized in
    :func:`chemtrain.trajectory.reweighting.init_pot_reweight_propagation_fns`.

    The computation of the gradient is batched to increase computational
    efficiency.

    Args:
        reference_dataloader: Dataloader containing the mapped atomistic
            reference positions.
        reweight_fns: Functions to perform and evaluate umbrella-sampling
            simulations to estimate the free energy gradients.
            Initialized via
            :func:`chemtrain.trajectory.reweighting.init_pot_reweight_propagation_fns`.
        energy_fn_template: Template to initialize the energy function that
            is required to compute the weights.
        kbt: Temperature of the statepoint
        vmap_batch_size: Batch for computing the potential energies on the
            reference positions.

    Returns:
        Returns the gradient and propagation function for the relative entropy
        minimization algorithm.

    """

    weights_fn, propagate_fn, safe_propagate = reweight_fns

    rel_entropy_loss = init_rel_entropy_loss_fn(
        energy_fn_template, weights_fn, kbt, vmap_batch_size)

    value_and_grad = jax.value_and_grad(rel_entropy_loss, argnums=0)

    @safe_propagate
    @jax.jit
    def safe_propagation_and_grad(params, traj_state, reference_batch):
        """Propagates the trajectory, if necessary, and computes the
        gradient via the relative entropy formalism.
        """
        traj_state = propagate_fn(params, traj_state)

        loss, grad = value_and_grad(params, traj_state, reference_batch)
        return traj_state, loss, grad

    def propagation_and_grad(params, traj_state, batch_state):
        new_batch_state, reference_batch = reference_dataloader(batch_state)
        outs = safe_propagation_and_grad(params, traj_state, reference_batch)
        return *outs, new_batch_state

    return propagation_and_grad


def init_step_size_adaption(weight_fn: Callable,
                            allowed_reduction: ArrayLike = 0.5,
                            interior_points: int = 10,
                            step_size_scale: float = 1e-7
                            ) -> Callable:
    """Initializes a line search to tune the step size in each iteration.

    This method interpolates linearly between the old parameters
    :math:`\\theta^{(i)}` and the paremeters :math:`\\tilde\\theta`
    proposed by the optimizer to find the optimal update

    .. math ::
        \\theta^{(i + 1)} = (1 - \\alpha) \\theta^{(i)} + \\alpha\\tilde\\theta

    that reduces the effective sample size to a predefined constant

    .. math ::
        N_\\text{eff}(\\theta^{(i+1)}) = r\cdot N_\\text{eff}(\\theta^{(i)}).

    This method uses a vectorized bisection algorithm with fixed number of
    iterations. At each iteration, the algorithm computes the effective
    sample size for a predefined number of interior points and updates the
    search interval boundaries to include the two closest points bisecting
    the residual.

    The number of required iterations computes from the number of interior
    points :math:`n_i` and the desired accuracy :math:`a` via

    .. math ::
        N = \\left\\lceil -\\log(a) / \\log(n_i + 1)\\right\\rceil.

    Args:
        weight_fn: Function computing a tuple (weights, N_eff) from the
            parameter states.
        allowed_reduction: Target reduction of the effective sample size
        interior_points: Number of interiour points
        step_size_scale: Accuracy of the found optimal interpolation
            coefficient

    Returns:
        Returns the interpolation coefficient :math:`\\alpha`.

    """

    # TODO: Makes this more general to work on an arbitrary measure,
    #       not only the effective sample size.
    #       Then, it is possible to move out the step size adaption
    #       into a more general trainer.

    iterations = int(onp.ceil(-onp.log(step_size_scale) / onp.log(interior_points + 1)))
    print(f"[Step size] Use {iterations} iterations for {interior_points} interior points.")

    @functools.partial(jax.vmap, in_axes=(0, None, None, None, None, None))
    def _residual(alpha, params, N_eff, batch_grad, proposal, traj_state):
        # Find the biggest reduction among the statepoints

        new_params = jax.tree_util.tree_map(
            lambda old, new: old * (1 - alpha) + new * alpha,
            params, proposal
        )

        _, N_eff_new = weight_fn(new_params, traj_state)
        reduction = jnp.log(N_eff_new) - jnp.log(N_eff)
        # Allow a reduction of the current effective sample size
        # The minimum reduction must still be larger than the allowed reduction
        # i.e. the residual of the final alpha must be greater than zero
        return reduction - jnp.log(allowed_reduction)

    def _step(state, _, params=None, N_effs=None, batch_grad=None, proposal=None, traj_states=None):
        a, b, res_a, res_b = state

        # Do not re-evaluate the residual for the already computed interval
        # boundaries
        c = jnp.reshape(jnp.linspace(a, b, interior_points + 2)[1:-1], (-1,))
        res_c = _residual(c, params, N_effs, batch_grad, proposal, traj_states)

        # debug.print("[Step Size] Residuals are {res}", res=res_c)

        # Add bondary points to the possible candidates
        c = jnp.concatenate((jnp.asarray([a, b]), c))
        res_c = jnp.concatenate((jnp.asarray([res_a, res_b]), res_c))

        # Find the smallest point bigger than zero and the biggest point
        # smaller than zero
        all_positive = jnp.where(res_c < 0, jnp.max(res_c), res_c)
        all_negative = jnp.where(res_c > 0, jnp.min(res_c), res_c)
        a_idx = jnp.argmin(all_positive)
        b_idx = jnp.argmax(all_negative)
        a, res_a = c[a_idx], res_c[a_idx]
        b, res_b = c[b_idx], res_c[b_idx]

        # debug.print("[Step Size] Search interval [{a}, {b}] with residual in [{res_a}, {res_b}]", a=a, b=b, res_a=res_a, res_b=res_b)

        return (a, b, res_a, res_b), None

    @jit
    def _adaptive_step_size(params, batch_grad, proposal, traj_state):
        _, N_eff = weight_fn(params, traj_state)
        a, b = 1.0e-5, 1.0
        res_a, res_b = _residual(
            jnp.asarray([a, b]),
            params, N_eff, batch_grad, proposal, traj_state)

        # Check that minimum step size is sufficiently small, else just keep
        # the minimum step size
        b = jnp.where(res_a <= 0, a, b)

        # Check whether full step does not reduce the effective step size
        # below the threshold. If this is the case do the full step
        a = jnp.where(jnp.logical_and(res_a > 0, res_b > 0), b, a)

        # In the other case, do the bisection with the unchanged initial
        # values of a and b
        _step_fn = functools.partial(
            _step, N_effs=N_eff, batch_grad=batch_grad, proposal=proposal,
            traj_states=traj_state, params=params)
        (a, b, res_a, _), _ = lax.scan(
            _step_fn,
            (a, b, res_a, res_b), onp.arange(iterations)
        )
        return a, res_a

    return _adaptive_step_size
